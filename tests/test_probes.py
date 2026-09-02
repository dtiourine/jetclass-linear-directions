import numpy as np
import pytest

from jetdirections.interp.probes import fit_linear_probe


def test_fit_linear_probe_recovers_linear_target():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    y = x @ np.array([1.5, -2.0, 0.0, 0.5, 3.0]) + rng.normal(scale=0.01, size=200)

    result = fit_linear_probe(x, y, np.arange(160), np.arange(160, 200))

    assert result.test_r2 > 0.999
    assert result.train_r2 > 0.999
    assert result.alpha > 0


def test_fit_linear_probe_rejects_overlapping_split():
    x = np.ones((5, 2))
    y = np.arange(5)

    with pytest.raises(ValueError, match="disjoint"):
        fit_linear_probe(x, y, [0, 1, 2], [2, 3, 4])
