"""Metric values against scikit-learn, batch semantics and edge cases."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn import metrics as skm

from evalstats import metrics as em
from evalstats._validation import FloatArray

TOL = 1e-12

LabelFn = Callable[..., float]

# (ours, scikit-learn's) for metrics on 0/1 predictions.
LABEL_PAIRS: list[tuple[LabelFn, LabelFn]] = [
    (em.accuracy, skm.accuracy_score),
    (em.precision, skm.precision_score),
    (em.recall, skm.recall_score),
    (em.f1, skm.f1_score),
]
SCORE_PAIRS: list[tuple[LabelFn, LabelFn]] = [
    (em.roc_auc, skm.roc_auc_score),
    (em.average_precision, skm.average_precision_score),
]
PROBABILITY_PAIRS: list[tuple[LabelFn, LabelFn]] = [
    (em.log_loss, skm.log_loss),
    (em.brier_score, skm.brier_score_loss),
]
REGRESSION_PAIRS: list[tuple[LabelFn, LabelFn]] = [
    (em.mean_absolute_error, skm.mean_absolute_error),
    (em.root_mean_squared_error, skm.root_mean_squared_error),
    (em.r2_score, skm.r2_score),
]


def _random_case(
    rng: np.random.Generator, weighted: bool
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    """Labels with both classes, coarse scores (many ties), 0/1 predictions, optional weights."""
    n = int(rng.integers(5, 400))
    y = rng.integers(0, 2, n).astype(np.float64)
    y[:2] = [0.0, 1.0]
    rng.shuffle(y)
    decimals = int(rng.integers(1, 4))
    scores = np.round(np.clip(0.3 * y + 0.7 * rng.random(n), 0.0, 1.0), decimals)
    # Always include a tie between a positive and a negative.
    scores[np.argmax(y == 1)] = scores[np.argmax(y == 0)]
    labels = (rng.random(n) < 0.5).astype(np.float64)
    labels[:2] = [0.0, 1.0]
    weights = rng.random(n) * (rng.random(n) > 0.2) + 1e-3 if weighted else None
    return y, scores, labels, weights


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("ours", "theirs"), LABEL_PAIRS, ids=lambda f: f.__name__)
def test_label_metrics_match_sklearn(ours: LabelFn, theirs: LabelFn, weighted: bool) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        y, _, labels, w = _random_case(rng, weighted)
        expected = theirs(y, labels, sample_weight=w)
        assert ours(y, labels, sample_weight=w) == pytest.approx(expected, abs=TOL, rel=TOL)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("ours", "theirs"), SCORE_PAIRS, ids=lambda f: f.__name__)
def test_score_metrics_match_sklearn_with_ties(
    ours: LabelFn, theirs: LabelFn, weighted: bool
) -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        y, scores, _, w = _random_case(rng, weighted)
        assert len(np.unique(scores)) < len(scores)  # the case really has ties
        expected = theirs(y, scores, sample_weight=w)
        assert ours(y, scores, sample_weight=w) == pytest.approx(expected, abs=TOL, rel=TOL)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("ours", "theirs"), PROBABILITY_PAIRS, ids=lambda f: f.__name__)
def test_probability_metrics_match_sklearn(ours: LabelFn, theirs: LabelFn, weighted: bool) -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        y, scores, _, w = _random_case(rng, weighted)
        expected = theirs(y, scores, sample_weight=w)
        assert ours(y, scores, sample_weight=w) == pytest.approx(expected, abs=TOL, rel=TOL)


def test_log_loss_clips_certain_wrong_predictions_like_sklearn() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    prob = np.array([0.0, 1.0, 0.0, 1.0])
    value = em.log_loss(y, prob)
    assert np.isfinite(value)
    assert value == pytest.approx(skm.log_loss(y, prob), abs=TOL, rel=TOL)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("ours", "theirs"), REGRESSION_PAIRS, ids=lambda f: f.__name__)
def test_regression_metrics_match_sklearn(ours: LabelFn, theirs: LabelFn, weighted: bool) -> None:
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = int(rng.integers(3, 400))
        y = rng.normal(5.0, 3.0, n)
        pred = y + rng.normal(0.0, 2.0, n)
        w = rng.random(n) + 0.01 if weighted else None
        expected = theirs(y, pred, sample_weight=w)
        assert ours(y, pred, sample_weight=w) == pytest.approx(expected, abs=TOL, rel=TOL)


def test_auc_known_values() -> None:
    y = [0, 0, 1, 1]
    assert em.roc_auc(y, [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert em.roc_auc(y, [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert em.roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    # One positive tied with one negative: 3 of 4 pairs ordered, one tie worth one half.
    assert em.roc_auc(y, [0.1, 0.3, 0.3, 0.4]) == pytest.approx(3.5 / 4)


def test_auc_is_invariant_to_monotone_transforms() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    scores = rng.normal(size=300) + y
    assert em.roc_auc(y, scores) == pytest.approx(em.roc_auc(y, np.exp(3 * scores)), abs=TOL)


def test_integer_weights_equal_repeated_rows() -> None:
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 60).astype(np.float64)
    y[:2] = [0, 1]
    scores = np.round(rng.random(60), 1)
    counts = rng.integers(0, 4, 60)
    counts[:2] = 1
    repeated = np.repeat(np.arange(60), counts)
    for metric in (em.ROC_AUC, em.AVERAGE_PRECISION, em.BRIER, em.LOG_LOSS):
        weighted = metric(y, scores, counts)
        assert weighted == pytest.approx(metric(y[repeated], scores[repeated]), abs=TOL)


@pytest.mark.parametrize("name", sorted(em.METRICS))
def test_batch_rows_equal_separate_evaluations(name: str) -> None:
    metric = em.METRICS[name]
    rng = np.random.default_rng(7)
    n = 80
    if metric.is_classification:
        y = rng.integers(0, 2, n).astype(np.float64)
        y[:2] = [0, 1]
        pred = (rng.random((3, n)) < 0.5).astype(np.float64)
        if metric.kind == "score":
            pred = np.round(rng.random((3, n)), 1)
    else:
        y = rng.normal(size=n)
        pred = y + rng.normal(size=(3, n))
    weights = rng.integers(0, 3, (3, n)).astype(np.float64) + 0.5

    # Many weight rows, one prediction vector (the bootstrap case).
    expected = [metric.batch(y, pred[0], weights[r])[0] for r in range(3)]
    np.testing.assert_allclose(metric.batch(y, pred[0], weights), expected, rtol=TOL, atol=TOL)
    # Many prediction rows, one weight vector (the permutation case).
    expected = [metric.batch(y, pred[r], weights[0])[0] for r in range(3)]
    np.testing.assert_allclose(metric.batch(y, pred, weights[0]), expected, rtol=TOL, atol=TOL)


@pytest.mark.parametrize(
    ("fn", "y", "pred", "message"),
    [
        (em.roc_auc, [1, 1, 1], [0.1, 0.2, 0.3], "auc is undefined: y_true contains a single"),
        (em.roc_auc, [0, 0], [0.1, 0.2], "single class"),
        (em.average_precision, [0, 0, 0], [0.1, 0.2, 0.3], "no positive examples"),
        (em.precision, [0, 1, 1], [0, 0, 0], "no positive predictions"),
        (em.recall, [0, 0, 0], [0, 1, 0], "no positive examples"),
        (em.f1, [0, 0, 0], [0, 0, 0], "no positive labels or predictions"),
        (em.r2_score, [2.0, 2.0, 2.0], [1.0, 2.0, 3.0], "constant"),
    ],
)
def test_undefined_metrics_raise_clear_errors(
    fn: LabelFn, y: ArrayLike, pred: ArrayLike, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fn(y, pred)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        em.roc_auc([0, 2, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="predicted labels"):
        em.accuracy([0, 1, 1], [0.2, 0.7, 0.9])
    with pytest.raises(ValueError, match="probabilities"):
        em.brier_score([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError, match="same length"):
        em.roc_auc([0, 1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        em.mean_absolute_error([1.0, 2.0], [1.0, np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        em.mean_absolute_error([[1.0, 2.0]], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="non-negative"):
        em.accuracy([0, 1], [0, 1], sample_weight=[1.0, -1.0])
    with pytest.raises(ValueError, match="positive entry"):
        em.accuracy([0, 1], [0, 1], sample_weight=[0.0, 0.0])
    with pytest.raises(ValueError, match="empty"):
        em.accuracy([], [])
    with pytest.raises(TypeError, match="numeric"):
        em.accuracy(["a", "b"], [0, 1])


def test_boolean_inputs_are_accepted() -> None:
    assert em.accuracy([True, False, True], [True, True, True]) == pytest.approx(2 / 3)


def test_metric_registry() -> None:
    assert em.get_metric("roc_auc") is em.ROC_AUC
    assert em.get_metric("ap") is em.AVERAGE_PRECISION
    assert em.get_metric(em.RMSE) is em.RMSE
    with pytest.raises(ValueError, match="unknown metric 'auroc'"):
        em.get_metric("auroc")
