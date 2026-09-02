from typing import Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from jetdirections.data.vectorize import LABEL_NAMES

# Class names in class-index order (QCD=0, Hbb=1, ...), derived from the same
# source of truth used to build the labels so the two can never drift apart.
CLASS_NAMES = [name.replace("label_", "") for name in LABEL_NAMES]
NUM_CLASSES = len(CLASS_NAMES)
BACKGROUND = "QCD"
BACKGROUND_INDEX = CLASS_NAMES.index(BACKGROUND)
# Every class except the QCD background is treated as a signal to be rejected
# against QCD, following the JetClass convention.
SIGNAL_INDICES = [i for i in range(NUM_CLASSES) if i != BACKGROUND_INDEX]

# Signal efficiencies at which background rejection is reported (ParT Table 2).
DEFAULT_EFFICIENCIES = (0.5, 0.99)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy."""
    return float((probs.argmax(axis=1) == labels).mean())


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Sample-weighted cross-entropy with no label smoothing."""
    return float(log_loss(labels, probs, labels=list(range(NUM_CLASSES))))


def per_class_auc(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """One-vs-rest ROC AUC for each class (NaN if a class is absent)."""
    aucs = {}
    for idx, name in enumerate(CLASS_NAMES):
        y_true = (labels == idx).astype(int)
        if y_true.min() == y_true.max():
            # Only one outcome present -> AUC undefined for this class.
            aucs[name] = float("nan")
        else:
            aucs[name] = float(roc_auc_score(y_true, probs[:, idx]))
    return aucs


def macro_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged one-vs-rest ROC AUC across all classes."""
    return float(
        roc_auc_score(
            labels,
            probs,
            multi_class="ovr",
            average="macro",
            labels=list(range(NUM_CLASSES)),
        )
    )


def background_rejection(
    probs: np.ndarray,
    labels: np.ndarray,
    signal: int,
    efficiency: float,
    background: int = BACKGROUND_INDEX,
) -> float:
    """
    Background rejection 1 / eps_B at a fixed signal efficiency eps_S.

    Restricts to the signal-vs-background two-class problem and scores it with
    the discriminant D = P(signal) / (P(signal) + P(background)), then reads the
    background efficiency off the ROC curve at the requested signal efficiency.

    Returns inf if no background passes the threshold (eps_B == 0).
    """
    keep = (labels == signal) | (labels == background)
    y_true = (labels[keep] == signal).astype(int)

    p_sig = probs[keep, signal]
    p_bkg = probs[keep, background]
    denom = p_sig + p_bkg
    # denom is 0 only if the model assigns zero mass to both classes; treat as a
    # maximally ambiguous score rather than dividing by zero.
    discriminant = np.divide(
        p_sig, denom, out=np.full_like(p_sig, 0.5), where=denom > 0
    )

    fpr, tpr, _ = roc_curve(y_true, discriminant)
    # tpr is monotonically non-decreasing, so np.interp gives eps_B at eps_S.
    eps_b = float(np.interp(efficiency, tpr, fpr))
    return float("inf") if eps_b == 0.0 else 1.0 / eps_b


def confusion(probs: np.ndarray, labels: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Confusion matrix (rows = true class). Row-normalized by default."""
    cm = _sk_confusion_matrix(
        labels, probs.argmax(axis=1), labels=list(range(NUM_CLASSES))
    ).astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    efficiencies=DEFAULT_EFFICIENCIES,
) -> dict:
    """
    Compute the full evaluation suite.

    Args:
        probs: (N, NUM_CLASSES) softmax probabilities.
        labels: (N,) integer class labels.
        efficiencies: signal efficiencies for background rejection.

    Returns a dict with accuracy, cross_entropy, macro_auc, per_class_auc,
    rejection (nested {signal_name: {eff: 1/eps_B}}) and confusion_matrix.
    """
    rejection = {}
    for sig in SIGNAL_INDICES:
        rejection[CLASS_NAMES[sig]] = {
            eff: background_rejection(probs, labels, sig, eff)
            for eff in efficiencies
        }

    return {
        "accuracy": accuracy(probs, labels),
        "cross_entropy": cross_entropy(probs, labels),
        "macro_auc": macro_auc(probs, labels),
        "per_class_auc": per_class_auc(probs, labels),
        "rejection": rejection,
        "confusion_matrix": confusion(probs, labels),
    }


def format_metrics(metrics: dict, efficiencies=DEFAULT_EFFICIENCIES) -> str:
    """Render the metrics dict as a human-readable table for stdout/logs."""
    lines = []
    lines.append(f"Accuracy:       {metrics['accuracy']:.4f}")
    lines.append(f"Cross-entropy:  {metrics['cross_entropy']:.4f}")
    lines.append(f"Macro AUC:      {metrics['macro_auc']:.4f}")
    lines.append("")

    eff_headers = "  ".join(f"Rej{int(e * 100)}%".rjust(10) for e in efficiencies)
    lines.append(f"{'Class':<8}{'AUC':>8}   {eff_headers}")
    lines.append("-" * (8 + 8 + 3 + len(eff_headers)))
    for name in CLASS_NAMES:
        auc = metrics["per_class_auc"][name]
        if name in metrics["rejection"]:
            rej = "  ".join(
                f"{metrics['rejection'][name][e]:10.1f}" for e in efficiencies
            )
        else:
            rej = "  ".join("-".rjust(10) for _ in efficiencies)
        lines.append(f"{name:<8}{auc:>8.4f}   {rej}")

    return "\n".join(lines)
