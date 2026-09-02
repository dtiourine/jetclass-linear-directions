from collections import Counter

import numpy as np
import pytest
import torch

from jetdirections.interp import matched_tertile_contrast


# --- independent reference for the cell grid (plain-Python restatement of the spec) ---

def reference_cells(target, confounders, n_bins):
    """Tertile indices and a per-jet cell id, computed independently of the implementation.

    Mirrors only the *definition* (quantile edges over the pooled tertiles, mixed-radix
    cell id); the grouping and downsampling under test are re-derived here from scratch.
    """
    target = np.asarray(target)
    confs = [np.asarray(c) for c in confounders]
    order = np.argsort(target, kind="stable")
    t = len(target) // 3
    idx_low, idx_high = order[:t], order[-t:]

    pooled = np.concatenate([idx_low, idx_high])
    inner = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(c[pooled], inner)) for c in confs]

    cell = np.full(len(target), -1, dtype=int)
    for j in range(len(target)):
        cid = 0
        for c, e in zip(confs, edges):
            cid = cid * (len(e) + 1) + int(np.digitize(c[j], e))
        cell[j] = cid
    return idx_low, idx_high, cell


def correlated_data(n=5000, rho=0.7, seed=7):
    """target and a single confounder with correlation ~rho; target also varies freely."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    target = a
    conf = rho * a + np.sqrt(1 - rho**2) * b
    return target, conf


# --- shape / typing ---

def test_returns_equal_length_int_index_arrays():
    target, conf = correlated_data()
    low, high = matched_tertile_contrast(target, [conf])
    assert low.ndim == high.ndim == 1
    assert len(low) == len(high) > 0
    for idx in (low, high):
        assert np.issubdtype(idx.dtype, np.integer)
        assert idx.min() >= 0 and idx.max() < len(target)


def test_accepts_torch_and_numpy_identically():
    target, conf = correlated_data()
    low_np, high_np = matched_tertile_contrast(target, [conf], seed=3)
    low_t, high_t = matched_tertile_contrast(
        torch.from_numpy(target), [torch.from_numpy(conf)], seed=3
    )
    assert np.array_equal(low_np, low_t)
    assert np.array_equal(high_np, high_t)
    # returned indices actually index a (seq-first-style) torch batch
    x = torch.randn(len(target), 3)
    assert x[low_t].shape == (len(low_t), 3)


# --- tertile membership (independent argsort) ---

def test_keep_is_subset_of_correct_tertiles():
    target, conf = correlated_data()
    order = np.argsort(target, kind="stable")
    t = len(target) // 3
    bottom, top = set(order[:t].tolist()), set(order[-t:].tolist())

    low, high = matched_tertile_contrast(target, [conf])
    assert set(low.tolist()) <= bottom
    assert set(high.tolist()) <= top


def test_tertiles_are_strictly_separated_in_target():
    target, conf = correlated_data()
    low, high = matched_tertile_contrast(target, [conf])
    # distinct-valued target => bottom tertile strictly below top tertile
    assert target[low].max() < target[high].min()


def test_no_duplicate_indices():
    target, conf = correlated_data()
    low, high = matched_tertile_contrast(target, [conf])
    assert len(set(low.tolist())) == len(low)
    assert len(set(high.tolist())) == len(high)
    assert set(low.tolist()).isdisjoint(high.tolist())


# --- the core guarantee: perfect joint balance on the matching grid ---

@pytest.mark.parametrize("n_confounders", [1, 2, 3])
def test_perfect_joint_balance(n_confounders):
    rng = np.random.default_rng(1)
    n = 6000
    a = rng.standard_normal(n)
    target = a
    confs = [0.6 * a + 0.8 * rng.standard_normal(n) for _ in range(n_confounders)]

    low, high = matched_tertile_contrast(target, confs, n_bins=5)

    idx_low, idx_high, cell = reference_cells(target, confs, n_bins=5)
    hist_low, hist_high = Counter(cell[low]), Counter(cell[high])

    # low and high occupy identical cells with identical multiplicity
    assert hist_low == hist_high
    # and each kept count is exactly min(available_low, available_high) for that cell
    pool_low, pool_high = Counter(cell[idx_low]), Counter(cell[idx_high])
    for c in set(pool_low) | set(pool_high):
        assert hist_low[c] == min(pool_low[c], pool_high[c])
    # every kept jet really lives in the tertile it was drawn from
    assert set(low.tolist()) <= set(idx_low.tolist())
    assert set(high.tolist()) <= set(idx_high.tolist())


def test_integer_confounder_with_heavy_ties():
    # multiplicity-like: integer confounder whose quantiles collapse via np.unique
    rng = np.random.default_rng(2)
    n = 4000
    a = rng.standard_normal(n)
    target = a
    conf = np.rint(3 * a + rng.standard_normal(n)).astype(int)  # few distinct values

    low, high = matched_tertile_contrast(target, [conf], n_bins=8)
    _, _, cell = reference_cells(target, [conf], n_bins=8)
    assert Counter(cell[low]) == Counter(cell[high])
    assert len(low) > 0


# --- determinism and the role of the seed ---

def test_reproducible_for_same_seed():
    target, conf = correlated_data()
    a = matched_tertile_contrast(target, [conf], seed=11)
    b = matched_tertile_contrast(target, [conf], seed=11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_seed_changes_draw_but_not_balance():
    target, conf = correlated_data()
    _, _, cell = reference_cells(target, [conf], n_bins=6)

    low0, high0 = matched_tertile_contrast(target, [conf], seed=0)
    low1, high1 = matched_tertile_contrast(target, [conf], seed=1)

    # counts per cell are a deterministic min(), independent of the seed ...
    assert Counter(cell[low0]) == Counter(cell[low1])
    assert Counter(cell[high0]) == Counter(cell[high1])
    # ... but which jets get drawn from over-full cells does change
    assert not np.array_equal(low0, low1)


# --- what the matching is for: kill the confound, keep the contrast ---

def test_matching_removes_confound_and_keeps_target_gap():
    target, conf = correlated_data(rho=0.7)
    order = np.argsort(target, kind="stable")
    t = len(target) // 3
    idx_low, idx_high = order[:t], order[-t:]

    raw_conf_gap = conf[idx_high].mean() - conf[idx_low].mean()
    raw_target_gap = target[idx_high].mean() - target[idx_low].mean()

    low, high = matched_tertile_contrast(target, [conf], n_bins=6)
    matched_conf_gap = conf[high].mean() - conf[low].mean()
    matched_target_gap = target[high].mean() - target[low].mean()

    # the confounder gap is largely closed ...
    assert abs(matched_conf_gap) < 0.25 * abs(raw_conf_gap)
    # ... while a clear target contrast survives
    assert matched_target_gap > 0.4 * raw_target_gap


# --- hand-computed micro example: pins edges + selection end to end ---

def test_hand_computed_example():
    # target strictly increasing => idx_low=[0,1,2], idx_high=[6,7,8]; middle discarded
    target = np.arange(9)
    #             0   1   2   3  4  5    6   7   8
    conf = np.array([10, 11, 40, 0, 0, 0, 12, 41, 42])
    # pooled conf over [0,1,2,6,7,8] = [10,11,40,12,41,42]; n_bins=2 => single edge at
    # the median 26.0. Bins: {10,11,12}->0, {40,41,42}->1.
    #   cell of idx_low  [0,1,2] = [0,0,1]   (conf 10,11 -> 0 ; 40 -> 1)
    #   cell of idx_high [6,7,8] = [0,1,1]   (conf 12 -> 0 ; 41,42 -> 1)
    # cell 0: low {0,1} vs high {6} -> keep 1 of {0,1} and 6
    # cell 1: low {2}   vs high {7,8} -> keep 2 and 1 of {7,8}
    low, high = matched_tertile_contrast(target, [conf], n_bins=2, seed=0)

    assert len(low) == len(high) == 2
    # deterministic members (the singleton side of each cell)
    assert 2 in low and 6 in high
    # downsampled members: exactly one from each over-full cell
    assert len(set(low) & {0, 1}) == 1
    assert len(set(high) & {7, 8}) == 1
    # nothing leaked in from the wrong cell/tertile
    assert set(low) <= {0, 1, 2} and set(high) <= {6, 7, 8}


def test_empty_when_tertiles_share_no_cell():
    # confounder == target => bottom and top tertiles fall in disjoint confounder bins,
    # so no cell overlaps and matching yields nothing
    target = np.arange(60)
    low, high = matched_tertile_contrast(target, [target.copy()], n_bins=6)
    assert len(low) == 0 and len(high) == 0
    assert np.issubdtype(low.dtype, np.integer)
