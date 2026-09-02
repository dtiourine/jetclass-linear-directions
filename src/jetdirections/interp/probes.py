from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV


DEFAULT_RIDGE_ALPHAS = np.logspace(-2, 4, 13)


@dataclass(frozen=True)
class LinearProbeResult:
    """A fitted ridge probe and its train/held-out performance."""

    probe: RidgeCV
    train_r2: float
    test_r2: float
    alpha: float


def fit_linear_probe(
    activations,
    targets,
    train_indices,
    test_indices,
    *,
    alphas=DEFAULT_RIDGE_ALPHAS,
) -> LinearProbeResult:
    """Fit a ridge probe and evaluate it on fixed train and held-out indices.

    The split is supplied by the caller so the same examples can be used for
    every layer and concept. RidgeCV chooses its regularization strength using
    only the training partition.

    Args:
        activations: Array-like of shape (n_examples, representation_dim).
        targets: Array-like of shape (n_examples,).
        train_indices: Integer indices used to fit the probe and select alpha.
        test_indices: Disjoint integer indices used only for held-out scoring.
        alphas: Candidate ridge penalties passed to sklearn.linear_model.RidgeCV.

    Returns:
        LinearProbeResult containing the fitted estimator, train and test R^2,
        and the selected ridge penalty.
    """
    x = np.asarray(activations)
    y = np.asarray(targets)
    train = np.asarray(train_indices)
    test = np.asarray(test_indices)

    if x.ndim != 2:
        raise ValueError(f"activations must be 2D, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {y.shape}")
    if len(x) != len(y):
        raise ValueError(
            f"activations and targets must have equal length, got {len(x)} and {len(y)}"
        )
    if np.intersect1d(train, test).size:
        raise ValueError("train_indices and test_indices must be disjoint")

    probe = RidgeCV(alphas=np.asarray(alphas)).fit(x[train], y[train])
    return LinearProbeResult(
        probe=probe,
        train_r2=float(probe.score(x[train], y[train])),
        test_r2=float(probe.score(x[test], y[test])),
        alpha=float(probe.alpha_),
    )
