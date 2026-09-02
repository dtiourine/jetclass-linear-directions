import torch
from pathlib import Path

from jetdirections.data.sampling import load_contrast_batches

N_JETS = 9
SEQ = 16


class StubDataset:
    """Stands in for JetClassDataset with one QCD file of N_JETS jets."""

    def __init__(self, x, mask, labels):
        self.data_dir = Path("stub")
        self.files = [Path("stub/ZJetsToNuNu_000.root")]
        self._batch = (x, mask, labels)

    def load_file(self, filepath):
        return self._batch


def test_contrast_batches_split_by_score():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N_JETS, 17, SEQ, generator=g)
    mask = torch.zeros(N_JETS, SEQ, dtype=torch.bool)
    labels = torch.arange(N_JETS)  # track jet identity through the split
    dataset = StubDataset(x, mask, labels)

    score = lambda x, mask: x[:, 0, 0]  # noqa: E731
    low, high = load_contrast_batches(dataset, "QCD", score, batch_size=N_JETS, seed=0)

    ranked = torch.argsort(x[:, 0, 0])
    assert sorted(low[2].tolist()) == sorted(ranked[:3].tolist())
    assert sorted(high[2].tolist()) == sorted(ranked[-3:].tolist())

    # x, mask, labels stay aligned per jet
    for x_b, mask_b, labels_b in (low, high):
        assert torch.equal(x_b, x[labels_b])
        assert torch.equal(mask_b, mask[labels_b])
