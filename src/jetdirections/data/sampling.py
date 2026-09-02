import random
from pathlib import Path
from typing import Optional, Tuple

import torch

FILENAME_TO_CLASS = {
    "HToBB": "Hbb",
    "HToCC": "Hcc",
    "HToGG": "Hgg",
    "HToWW4Q": "H4q",
    "HToWW2Q1L": "Hqql",
    "ZToQQ": "Zqq",
    "WToQQ": "Wqq",
    "TTBar": "Tbqq",
    "TTBarLep": "Tbl",
    "ZJetsToNuNu": "QCD",
}

CLASS_TO_FILENAME = {cls: prefix for prefix, cls in FILENAME_TO_CLASS.items()}


def _file_class(path: Path) -> Optional[str]:
    prefix = path.stem.rsplit("_", 1)[0]  # "HToBB_000" -> "HToBB"
    return FILENAME_TO_CLASS.get(prefix)


def load_class_batch(dataset, class_name: str, batch_size: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns x (B, 17, P), mask (B, P; True=padding), labels (B,)
        B = batch_size, 17 = features, P = max_particles
    """
    class_files = [f for f in dataset.files if _file_class(f) == class_name]
    if not class_files:
        raise ValueError(f"No files for class '{class_name}' in {dataset.data_dir}")
    random.Random(seed).shuffle(class_files)  

    xs, ms, ls, total = [], [], [], 0
    for f in class_files:
        x, mask, labels = dataset.load_file(f)
        xs.append(x); ms.append(mask); ls.append(labels)
        total += len(x)
        if total >= batch_size:
            break

    x, mask, labels = torch.cat(xs), torch.cat(ms), torch.cat(ls)
    return x[:batch_size], mask[:batch_size], labels[:batch_size]


def load_contrast_batches(dataset, class_name: str, score, batch_size: int, seed: Optional[int] = None):
    """
    Load one batch of a single class and split it into bottom/top tertiles of `score`.

    Args:
        score: callable(x, mask) -> (B,) per-jet scalar to sort by.

    Returns:
        (low_batch, high_batch), each an (x, mask, labels) tuple of batch_size // 3 jets.
    """
    x, mask, labels = load_class_batch(dataset, class_name, batch_size, seed=seed)
    order = torch.argsort(score(x, mask))
    n = batch_size // 3
    low, high = order[:n], order[-n:]
    return (x[low], mask[low], labels[low]), (x[high], mask[high], labels[high])


CLASS_NAMES = list(FILENAME_TO_CLASS.values())


def load_stratified_batch(dataset, per_class: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ms, ls = [], [], []
    for class_name in CLASS_NAMES:
        x, mask, labels = load_class_batch(dataset, class_name, per_class, seed=seed)
        xs.append(x); ms.append(mask); ls.append(labels)
    return torch.cat(xs), torch.cat(ms), torch.cat(ls)
