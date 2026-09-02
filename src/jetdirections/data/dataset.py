import uproot
import torch
from pathlib import Path
from typing import Tuple
import random

from .vectorize import vectorize


class JetClassDataset:
    """
    Iterable over JetClass ROOT files, yielding one file's tensors at a time.

    Usage:
        dataset = JetClassDataset('data/train')
        for x, mask, labels in dataset:
            output = model(x, mask)
    """

    def __init__(self, data_dir: str | Path, max_particles: int = 128, shuffle_files: bool = True):
        self.data_dir = Path(data_dir)
        self.max_particles = max_particles
        self.shuffle_files = shuffle_files

        self.files = sorted(list(self.data_dir.glob("*.root")))
        if not self.files:
            raise FileNotFoundError(f"No .root files found in {self.data_dir}")


    def __iter__(self):
        """Iterate through all files, yielding (x, mask, labels) for each."""
        file_order = list(range(len(self.files)))
        if self.shuffle_files:
            random.shuffle(file_order)

        for idx in file_order:
            yield self.load_file(self.files[idx])

    def load_file(self, filepath: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns CPU tensors for one ROOT file:
            x:      (num_jets, 17, max_particles) - particle features
            mask:   (num_jets, max_particles)     - True = padding
            labels: (num_jets,)                   - class indices
        """
        with uproot.open(filepath) as file:
            return vectorize(file["tree"], max_particles=self.max_particles)
