import numpy as np


def matched_tertile_contrast(target, confounders, n_bins=6, seed=0):
    """Indices of the bottom vs top `target` tertile, matched on `confounders`.

    Splits jets into the bottom and top tertiles of `target`, bins both on a shared
    quantile grid over the confounders, and keeps min(n_low, n_high) jets per cell from
    each side. The two returned groups then share the same joint confounder distribution
    (identical per-cell counts) while keeping most of the target gap, so a difference of
    means across them isolates `target` rather than the confounders it correlates with.

    Args:
        target: (N,) values whose tertiles define the contrast (e.g. jet mass). Tensor or
            array; ties are broken by original order.
        confounders: sequence of (N,) arrays to match on (e.g. [multiplicity, jet_pt]).
        n_bins: quantile bins per confounder for the matching grid. Integer-valued
            confounders whose quantiles coincide simply yield fewer bins.
        seed: RNG seed for the per-cell downsampling.

    Returns:
        (keep_low, keep_high): equal-length int index arrays into the N jets. Empty when
        the tertiles share no confounder cell.
    """
    target = np.asarray(target)
    confs = [np.asarray(c) for c in confounders]

    order = np.argsort(target, kind="stable")
    tertile = len(target) // 3
    idx_low, idx_high = order[:tertile], order[-tertile:]

    # One set of quantile edges per confounder, shared across both tertiles.
    pooled = np.concatenate([idx_low, idx_high])
    inner = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(c[pooled], inner)) for c in confs]

    def cells(idx):  # mixed-radix cell id from each confounder's bin
        cell = np.zeros(len(idx), dtype=np.intp)
        for c, e in zip(confs, edges):
            cell = cell * (len(e) + 1) + np.digitize(c[idx], e)
        return cell

    rng = np.random.default_rng(seed)
    cells_low, cells_high = cells(idx_low), cells(idx_high)
    keep_low, keep_high = [], []
    for c in np.unique(np.concatenate([cells_low, cells_high])):
        in_low, in_high = idx_low[cells_low == c], idx_high[cells_high == c]
        k = min(len(in_low), len(in_high))
        if k:
            keep_low.append(rng.choice(in_low, size=k, replace=False))
            keep_high.append(rng.choice(in_high, size=k, replace=False))

    empty = np.empty(0, dtype=np.intp)
    return (np.concatenate(keep_low) if keep_low else empty,
            np.concatenate(keep_high) if keep_high else empty)
