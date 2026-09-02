"""Sanity checks for jetdirections.training.metrics on perfect / uninformative inputs."""

import numpy as np
import pytest

from jetdirections.training import metrics
from jetdirections.training.metrics import (
    BACKGROUND_INDEX,
    CLASS_NAMES,
    NUM_CLASSES,
    SIGNAL_INDICES,
)


def _labels(per_class=100):
    """Balanced labels: per_class samples of every class, in order."""
    return np.repeat(np.arange(NUM_CLASSES), per_class)


def _one_hot(labels):
    probs = np.zeros((len(labels), NUM_CLASSES))
    probs[np.arange(len(labels)), labels] = 1.0
    return probs


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_class_names_match_label_order():
    # QCD is the background and must be class index 0 (see vectorize.LABEL_NAMES).
    assert NUM_CLASSES == 10
    assert CLASS_NAMES[0] == "QCD"
    assert BACKGROUND_INDEX == 0
    assert BACKGROUND_INDEX not in SIGNAL_INDICES
    assert len(SIGNAL_INDICES) == NUM_CLASSES - 1


def test_perfect_classifier():
    labels = _labels()
    probs = _one_hot(labels)

    assert metrics.accuracy(probs, labels) == 1.0
    assert metrics.macro_auc(probs, labels) == 1.0
    # log_loss clips probabilities, so a perfect classifier is ~0, not exactly 0.
    assert metrics.cross_entropy(probs, labels) < 1e-6

    for name, auc in metrics.per_class_auc(probs, labels).items():
        assert auc == 1.0, name

    # No background ever passes a perfect signal cut -> infinite rejection.
    for sig in SIGNAL_INDICES:
        for eff in (0.5, 0.99):
            assert metrics.background_rejection(probs, labels, sig, eff) == float("inf")


def test_confusion_matrix_perfect_is_identity():
    labels = _labels()
    probs = _one_hot(labels)
    cm = metrics.confusion(probs, labels, normalize=True)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    np.testing.assert_allclose(cm, np.eye(NUM_CLASSES))


def test_uniform_cross_entropy_is_log_num_classes():
    labels = _labels()
    probs = np.full((len(labels), NUM_CLASSES), 1.0 / NUM_CLASSES)
    assert metrics.cross_entropy(probs, labels) == pytest.approx(np.log(NUM_CLASSES))


def test_uninformative_classifier():
    # Scores independent of labels -> AUC ~ 0.5 and rejection ~ 1/efficiency.
    rng = np.random.default_rng(0)
    n = 20000
    labels = rng.integers(0, NUM_CLASSES, size=n)
    probs = _softmax(rng.standard_normal((n, NUM_CLASSES)))

    assert metrics.accuracy(probs, labels) == pytest.approx(1.0 / NUM_CLASSES, abs=0.02)
    assert metrics.macro_auc(probs, labels) == pytest.approx(0.5, abs=0.03)

    sig = SIGNAL_INDICES[0]
    assert metrics.background_rejection(probs, labels, sig, 0.5) == pytest.approx(2.0, rel=0.25)


def test_compute_and_format_run():
    # End-to-end smoke test: compute_metrics + format_metrics don't raise.
    rng = np.random.default_rng(1)
    n = 2000
    labels = rng.integers(0, NUM_CLASSES, size=n)
    probs = _softmax(rng.standard_normal((n, NUM_CLASSES)))

    result = metrics.compute_metrics(probs, labels)
    assert set(result) == {
        "accuracy",
        "cross_entropy",
        "macro_auc",
        "per_class_auc",
        "rejection",
        "confusion_matrix",
    }
    assert len(result["rejection"]) == len(SIGNAL_INDICES)
    text = metrics.format_metrics(result)
    assert "Accuracy" in text and "Rej50%" in text
