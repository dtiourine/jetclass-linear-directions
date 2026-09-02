import pytest
import torch

from jetdirections.model.transformer import JetTagTransformer
from jetdirections.interp import extract_steering_vector, extract_cls_activations

INPUT_DIM = 17
EMBED = 32
HEADS = 4
LAYERS = 3
SEQ = 16
BATCH = 8
NUM_CLASSES = 10


@pytest.fixture
def model():
    torch.manual_seed(0)
    return JetTagTransformer(embed_dim=EMBED, num_heads=HEADS, num_layers=LAYERS)


def make_batch(seed, n=BATCH, seq=SEQ, n_pad=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, INPUT_DIM, seq, generator=g)
    mask = torch.zeros(n, seq, dtype=torch.bool)
    if n_pad:
        mask[:, seq - n_pad:] = True
    labels = torch.zeros(n, dtype=torch.long)
    return x, mask, labels


def hook_reference(model, batch1, batch2, layer):
    """Independent extraction via a forward hook on the block (no nnsight)."""
    device = next(model.parameters()).device
    captured = {}

    def hook(_module, _inputs, output):
        captured["cls"] = output[0][0].detach()

    handle = model.blocks[layer].register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(batch2[0].to(device), batch2[1].to(device))
            mean2 = captured["cls"].mean(dim=0)
            model(batch1[0].to(device), batch1[1].to(device))
            mean1 = captured["cls"].mean(dim=0)
    finally:
        handle.remove()
        model.train(was_training)
    return (mean2 - mean1).cpu()


def test_shape_dtype_device(model):
    v = extract_steering_vector(model, make_batch(1), make_batch(2), layer=1)
    assert v.shape == (EMBED,)
    assert v.dtype == torch.float32
    assert v.device.type == "cpu"
    assert not v.requires_grad
    assert torch.isfinite(v).all()


def test_identical_batches_give_zero(model):
    batch = make_batch(1)
    v = extract_steering_vector(model, batch, batch, layer=1)
    assert torch.allclose(v, torch.zeros(EMBED), atol=1e-6)


def test_antisymmetric_in_batch_order(model):
    b1, b2 = make_batch(1), make_batch(2)
    v12 = extract_steering_vector(model, b1, b2, layer=2)
    v21 = extract_steering_vector(model, b2, b1, layer=2)
    assert torch.allclose(v12, -v21, atol=1e-6)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_matches_forward_hook_reference(model, layer):
    b1, b2 = make_batch(1), make_batch(2)
    v = extract_steering_vector(model, b1, b2, layer=layer)
    ref = hook_reference(model, b1, b2, layer)
    assert torch.allclose(v, ref, atol=1e-5)


def test_different_layers_give_different_vectors(model):
    b1, b2 = make_batch(1), make_batch(2)
    v0 = extract_steering_vector(model, b1, b2, layer=0)
    v2 = extract_steering_vector(model, b1, b2, layer=2)
    assert not torch.allclose(v0, v2)


def test_leaves_model_in_eval(model):
    model.train()
    extract_steering_vector(model, make_batch(1), make_batch(2), layer=1)
    assert not model.training


def test_runs_with_padding(model):
    b1 = make_batch(1, n_pad=4)
    b2 = make_batch(2, n_pad=6)
    v = extract_steering_vector(model, b1, b2, layer=1)
    assert torch.isfinite(v).all()


def test_chunked_matches_unchunked(model):
    b1, b2 = make_batch(1), make_batch(2)
    v = extract_steering_vector(model, b1, b2, layer=1)
    v_chunked = extract_steering_vector(model, b1, b2, layer=1, chunk_size=3)  # uneven chunks of BATCH=8
    assert torch.allclose(v, v_chunked, atol=1e-5)














def residuals_hook_reference(model, batch, layers, steering=None):
    """Independent CLS-residual extraction via forward hooks on the blocks (no nnsight)."""
    device = next(model.parameters()).device
    captured = {}
    handles = []

    if steering is not None:
        vec, steer_layer, scale = steering
        v = (scale * vec).to(device)

        def steer_hook(_module, _inputs, output):
            h = output[0].clone()
            h[0] = h[0] + v
            return (h,) + tuple(output[1:])

        handles.append(model.blocks[steer_layer].register_forward_hook(steer_hook))

    def make_reader(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output[0][0].detach().cpu()
        return hook

    # readers registered after the steering hook, so a reader at the steering layer
    # sees the post-steering residual — matching extract_cls_activations semantics
    for layer in layers:
        handles.append(model.blocks[layer].register_forward_hook(make_reader(layer)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x, mask, _ = batch
            model(x.to(device), mask.to(device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return captured


def test_residuals_shape_dtype_device(model):
    res = extract_cls_activations(model, make_batch(1), layers=[0, 2])
    assert set(res) == {0, 2}
    for t in res.values():
        assert t.shape == (BATCH, EMBED)
        assert t.dtype == torch.float32
        assert t.device.type == "cpu"
        assert not t.requires_grad
        assert torch.isfinite(t).all()


def test_residuals_match_forward_hook_reference(model):
    batch = make_batch(1)
    res = extract_cls_activations(model, batch, layers=[0, 1, 2])
    ref = residuals_hook_reference(model, batch, layers=[0, 1, 2])
    for layer in [0, 1, 2]:
        assert torch.allclose(res[layer], ref[layer], atol=1e-5)


def test_residuals_chunked_matches_unchunked(model):
    batch = make_batch(1)
    res = extract_cls_activations(model, batch, layers=[1])
    res_chunked = extract_cls_activations(model, batch, layers=[1], chunk_size=3)  # uneven chunks of BATCH=8
    assert torch.allclose(res[1], res_chunked[1], atol=1e-5)


def test_residuals_mean_diff_matches_steering_vector(model):
    b1, b2 = make_batch(1), make_batch(2)
    v = extract_steering_vector(model, b1, b2, layer=1)
    r1 = extract_cls_activations(model, b1, layers=[1])[1]
    r2 = extract_cls_activations(model, b2, layers=[1])[1]
    assert torch.allclose(r2.mean(dim=0) - r1.mean(dim=0), v, atol=1e-5)


def test_residuals_steered_match_forward_hook_reference(model):
    batch = make_batch(1)
    steering = (torch.randn(EMBED), 0, 2.5)
    res = extract_cls_activations(model, batch, layers=[1, 2], steering=steering)
    ref = residuals_hook_reference(model, batch, layers=[1, 2], steering=steering)
    for layer in [1, 2]:
        assert torch.allclose(res[layer], ref[layer], atol=1e-5)


def test_residuals_at_steer_layer_are_post_steering(model):
    batch = make_batch(1)
    vec = torch.randn(EMBED)
    base = extract_cls_activations(model, batch, layers=[1])[1]
    steered = extract_cls_activations(model, batch, layers=[1], steering=(vec, 1, 3.0))[1]
    assert torch.allclose(steered, base + 3.0 * vec, atol=1e-5)


def test_residuals_zero_scale_matches_unsteered(model):
    batch = make_batch(1)
    base = extract_cls_activations(model, batch, layers=[2])[2]
    steered = extract_cls_activations(model, batch, layers=[2], steering=(torch.randn(EMBED), 0, 0.0))[2]
    assert torch.allclose(base, steered, atol=1e-6)


def test_residuals_runs_with_padding(model):
    res = extract_cls_activations(model, make_batch(1, n_pad=5), layers=[1])
    assert torch.isfinite(res[1]).all()
