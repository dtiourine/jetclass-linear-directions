import torch
from .dataset import JetClassDataset
from jetdirections.config import BATCH_SIZE

class JetClassDataLoader:
    """
    Yields shuffled, mixed-class batches from a JetClassDataset (each ROOT file
    holds one class, so files are buffered and shuffled together first).

    Usage:
        dataset = JetClassDataset('data/train')
        loader = JetClassDataLoader(dataset, batch_size=512, buffer_size=20)
        for x, mask, labels in loader:
            logits = model(x, mask)
    """

    def __init__(
        self,
        dataset: JetClassDataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        drop_last: bool = False,        # drop the final partial batch
        buffer_size: int = 10,          # ROOT files buffered & shuffled together
        jets_per_file: int = 100000,    # approximate 
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.jets_per_file = jets_per_file

    def __iter__(self):
        buffer_x = []
        buffer_mask = []
        buffer_labels = []
        files_in_buffer = 0

        for x, mask, labels in self.dataset:
            buffer_x.append(x)
            buffer_mask.append(mask)
            buffer_labels.append(labels)
            files_in_buffer += 1

            if files_in_buffer >= self.buffer_size:
                yield from self._yield_from_buffer(buffer_x, buffer_mask, buffer_labels)
                buffer_x = []
                buffer_mask = []
                buffer_labels = []
                files_in_buffer = 0

        if buffer_x:
            yield from self._yield_from_buffer(buffer_x, buffer_mask, buffer_labels)

    def _yield_from_buffer(self, buffer_x, buffer_mask, buffer_labels):
        x = torch.cat(buffer_x, dim=0)
        mask = torch.cat(buffer_mask, dim=0)
        labels = torch.cat(buffer_labels, dim=0)
        # free the per-file copies now that they're concatenated (caps peak RAM)
        buffer_x.clear()
        buffer_mask.clear()
        buffer_labels.clear()

        num_jets = x.size(0)
        # gather per batch instead of permuting the whole buffer (avoids a full-size copy)
        order = torch.randperm(num_jets) if self.shuffle else torch.arange(num_jets)

        for start_idx in range(0, num_jets, self.batch_size):
            idx = order[start_idx:start_idx + self.batch_size]
            if self.drop_last and idx.numel() < self.batch_size:
                break
            yield x[idx], mask[idx], labels[idx]

    def __len__(self):
        return (len(self.dataset.files) * self.jets_per_file) // self.batch_size  # approximate
