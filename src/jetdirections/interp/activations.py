import torch
from nnsight import NNsight


def extract_cls_activations(model, batch, layers, chunk_size=512, steering=None):
    """Per-jet CLS-token activations at each block in `layers`.

    Args:
        model: JetTagTransformer.
        batch: (x, mask, labels) tuple; labels are ignored.
        layers: iterable of indices into model.blocks; activations are read at each
            block's output.
        chunk_size: jets per forward pass, so GPU memory stays flat for large batches.
        steering: optional (vec, layer, scale) tuple; adds scale * vec to the CLS token
            at that block's output before reading. Reading
            at the steering layer itself returns the post-steering activation.

    Returns:
        Dict mapping each layer to a (batch, embed) tensor of CLS activations, on CPU.
    """
    device = next(model.parameters()).device
    model.eval()
    nsm = NNsight(model)

    layers = list(layers)
    x, mask, _ = batch
    if steering is not None:
        vec, steer_layer, scale = steering
        v = (scale * vec).to(device)

    chunks = {layer: [] for layer in layers}
    for i in range(0, len(x), chunk_size):
        saved = {}
        with (
            torch.no_grad(),
            nsm.trace(
                x[i : i + chunk_size].to(device), mask[i : i + chunk_size].to(device)
            ),
        ):
            if steering is not None:
                nsm.blocks[steer_layer].output[0][0] += v
            for layer in layers:
                saved[layer] = nsm.blocks[layer].output[0][0].save()
        for layer in layers:
            chunks[layer].append(saved[layer].detach().cpu())

    return {layer: torch.cat(c) for layer, c in chunks.items()}


def extract_cls_activations_with_steering(
    model, batch, layers, steering_vectors, scale=1.0, chunk_size=512
):
    """Per-jet CLS-token activations at each block in `layers`, with steering applied.

    Combines extract_cls_activations' per-jet granularity with the multi-layer
    steering dict: every vector in
    `steering_vectors` is added to the residual stream in the same forward pass, in
    block order, before any of `layers` is read. Reading at a layer that is also
    being steered returns the post-steering activation for that layer.

    Args:
        model: JetTagTransformer.
        batch: (x, mask, labels) tuple; labels are ignored.
        layers: iterable of indices into model.blocks; activations are read at each
            block's output.
        steering_vectors: dict mapping a block index to that block's steering
            vector pair (steer_vec_cls, steer_vec_part), each of shape (embed,), as
            in apply_steering.
        scale: coefficient multiplying every steering vector.
        chunk_size: jets per forward pass, so GPU memory stays flat for large batches.

    Returns:
        Dict mapping each layer in `layers` to a (batch, embed) tensor of CLS
        activations, on CPU.
    """
    device = next(model.parameters()).device
    model.eval()
    nsm = NNsight(model)

    layers = list(layers)
    x, mask, _ = batch

    chunks = {layer: [] for layer in layers}
    for i in range(0, len(x), chunk_size):
        saved = {}
        with (
            torch.no_grad(),
            nsm.trace(
                x[i : i + chunk_size].to(device), mask[i : i + chunk_size].to(device)
            ),
        ):
            # steering and reads interleaved in block order: nnsight requires
            # touching module outputs in execution order within a trace
            for layer in sorted(set(layers) | set(steering_vectors)):
                if layer in steering_vectors:
                    steer_vec_cls, steer_vec_part = steering_vectors[layer]
                    layer_activations = nsm.blocks[layer].output[0]  # (seq, batch, embed)
                    layer_activations[0] += scale * steer_vec_cls.to(device)
                    layer_activations[1:] += scale * steer_vec_part.to(device)
                if layer in layers:
                    saved[layer] = nsm.blocks[layer].output[0][0].save()
        for layer in layers:
            chunks[layer].append(saved[layer].detach().cpu())

    return {layer: torch.cat(c) for layer, c in chunks.items()}
