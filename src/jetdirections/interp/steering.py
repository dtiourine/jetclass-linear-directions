import torch
from nnsight import NNsight


def extract_steering_vector(model, batch1, batch2, layer, chunk_size=512):
    """Difference of mean CLS-token activations at `layer`, computed as batch2 - batch1.

    Args:
        model: JetTagTransformer.
        batch1, batch2: (x, mask, labels) tuples; labels are ignored.
        layer: index into model.blocks; the vector is taken at that block's output.
        chunk_size: jets per forward pass, so GPU memory stays flat for large batches.

    Returns:
        Steering vector of shape (embed,), unnormalized, on CPU.
    """
    device = next(model.parameters()).device
    model.eval()
    nsm = NNsight(model)

    def cls_mean(batch):
        x, mask, _ = batch
        total = 0.0
        for i in range(0, len(x), chunk_size):
            with torch.no_grad(), nsm.trace(x[i:i + chunk_size].to(device), mask[i:i + chunk_size].to(device)):
                cls = nsm.blocks[layer].output[0][0].save()
            total = total + cls.sum(dim=0)
        return total / len(x)

    return (cls_mean(batch2) - cls_mean(batch1)).detach().cpu()


def apply_steering(model, batch, steering_vectors, scale=1.0, chunk_size=512):
    """Run `model` on `batch`, applying each steering vector in `steering_vectors` to its block.

    At every block in `steering_vectors`, `scale * steer_vec_cls` is added to the CLS
    token and `scale * steer_vec_part` to every particle token. (Steering padding tokens
    are harmless since they are masked in attention.)

    The notebooks in `experiments/` use this general function for CLS-only steering
    by passing zero particle-token vectors.

    Args:
        model: JetTagTransformer.
        batch: (x, mask, labels) tuple; labels are ignored.
        steering_vectors: dict mapping a block index to that block's steering
            vector pair (steer_vec_cls, steer_vec_part), each of shape (embed,).
        scale: coefficient multiplying every steering vector.
        chunk_size: jets per forward pass, so GPU memory stays flat.

    Returns:
        Logits of shape (batch, num_classes), on CPU.
    """
    for layer, (steer_vec_cls, steer_vec_part) in steering_vectors.items():
        assert 0 <= layer < len(model.blocks), \
            f"steering_vectors key {layer!r} is not a block index (model has {len(model.blocks)} blocks)"
        assert steer_vec_cls.dim() == 1 and steer_vec_cls.shape == steer_vec_part.shape, \
            f"steering_vectors[{layer}] must be two (embed,) vectors, got shapes " \
            f"{tuple(steer_vec_cls.shape)} and {tuple(steer_vec_part.shape)}"

    device = next(model.parameters()).device
    model.eval()
    nsm = NNsight(model)

    x, mask, _ = batch

    out = []
    for i in range(0, len(x), chunk_size):
        with torch.no_grad(), nsm.trace(x[i:i + chunk_size].to(device), mask[i:i + chunk_size].to(device)):
            for layer in sorted(steering_vectors):
                steer_vec_cls, steer_vec_part = steering_vectors[layer]
                layer_activations = nsm.blocks[layer].output[0]  # (seq, batch, embed)
                layer_activations[0] += scale * steer_vec_cls.to(device) # seq 0 is the CLS token
                layer_activations[1:] += scale * steer_vec_part.to(device)
            logits = nsm.output.save()
        out.append(logits.detach().float().cpu())
    return torch.cat(out)
