"""Metrics for binary classification and regression.

Each metric has a single implementation, a *batch* function

    batch(y_true, y_pred, weights) -> values

where ``y_true`` has shape ``(n,)`` and ``y_pred`` and ``weights`` have shape ``(n,)`` or
``(rows, n)``. Row ``r`` of the result is the metric computed with predictions ``y_pred[r]``
and weights ``weights[r]``; a 1-D argument is shared by every row. The same code therefore
serves three purposes:

* a point estimate is a batch of one row, with the sample weights (or ones);
* a bootstrap passes a matrix of resampling counts as ``weights``, so a resample is never
  materialised as a copy of the data;
* a permutation test passes a matrix of permuted predictions as ``y_pred``.

A batch returns NaN where the metric is undefined (for example ROC AUC on a resample with
one class). The public functions validate their inputs and raise ``ValueError`` instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from evalstats._validation import (
    FloatArray,
    IntArray,
    as_float_vector,
    as_weights,
    check_binary,
    check_probabilities,
    check_same_length,
)

MetricKind = Literal["label", "score", "regression"]
BatchFn = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]

# Probabilities are clipped to [EPS, 1 - EPS] before taking logarithms, the same convention
# scikit-learn uses for float64 input. A confident wrong prediction is therefore large
# (about 36 nats) rather than infinite.
LOG_LOSS_EPS = float(np.finfo(np.float64).eps)


def _rows(array: FloatArray) -> FloatArray:
    return array if array.ndim == 2 else array[np.newaxis, :]


def _sum_rows(array: FloatArray) -> FloatArray:
    return np.asarray(np.sum(array, axis=-1), dtype=np.float64)


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Elementwise ratio with NaN where the denominator is not positive."""
    num, den = np.broadcast_arrays(numerator, denominator)
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


# --------------------------------------------------------------------------------------------
# Batch implementations
# --------------------------------------------------------------------------------------------


def _accuracy_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    w2 = _rows(w)
    correct = (_rows(pred) == y).astype(np.float64)
    return _ratio(_sum_rows(w2 * correct), _sum_rows(w2))


def _precision_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    w2, p2 = _rows(w), _rows(pred)
    return _ratio(_sum_rows(w2 * y * p2), _sum_rows(w2 * p2))


def _recall_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    w2, p2 = _rows(w), _rows(pred)
    return _ratio(_sum_rows(w2 * y * p2), _sum_rows(w2 * y))


def _f1_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (actual positives + predicted positives).
    w2, p2 = _rows(w), _rows(pred)
    true_pos = _sum_rows(w2 * y * p2)
    return _ratio(2.0 * true_pos, _sum_rows(w2 * y) + _sum_rows(w2 * p2))


def _log_loss_batch(y: FloatArray, prob: FloatArray, w: FloatArray) -> FloatArray:
    w2 = _rows(w)
    p = np.clip(_rows(prob), LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return _ratio(_sum_rows(w2 * loss), _sum_rows(w2))


def _brier_batch(y: FloatArray, prob: FloatArray, w: FloatArray) -> FloatArray:
    w2 = _rows(w)
    return _ratio(_sum_rows(w2 * (y - _rows(prob)) ** 2), _sum_rows(w2))


def _mae_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    w2 = _rows(w)
    return _ratio(_sum_rows(w2 * np.abs(y - _rows(pred))), _sum_rows(w2))


def _rmse_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    w2 = _rows(w)
    return np.sqrt(_ratio(_sum_rows(w2 * (y - _rows(pred)) ** 2), _sum_rows(w2)))


def _r2_batch(y: FloatArray, pred: FloatArray, w: FloatArray) -> FloatArray:
    w2 = _rows(w)
    total = _sum_rows(w2)
    mean = _ratio(_sum_rows(w2 * y), total)
    # Two passes (centre, then square) to avoid cancellation in the total sum of squares.
    ss_tot = _sum_rows(w2 * (y - mean[:, np.newaxis]) ** 2)
    ss_res = _sum_rows(w2 * (y - _rows(pred)) ** 2)
    return 1.0 - _ratio(ss_res, ss_tot)


def _tie_bounds(sorted_scores: FloatArray) -> tuple[IntArray, IntArray]:
    """For each position of row-wise sorted scores, the first and last index of its tie group."""
    n = sorted_scores.shape[1]
    positions = np.arange(n)
    starts_group = np.ones(sorted_scores.shape, dtype=np.bool_)
    starts_group[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    ends_group = np.ones(sorted_scores.shape, dtype=np.bool_)
    ends_group[:, :-1] = starts_group[:, 1:]
    start = np.maximum.accumulate(np.where(starts_group, positions, 0), axis=1)
    end_reversed = np.where(ends_group, positions, n - 1)[:, ::-1]
    end = np.minimum.accumulate(end_reversed, axis=1)[:, ::-1]
    return start, end


def _gather(values: FloatArray, index: IntArray) -> FloatArray:
    """Row-wise ``values[r, index[r]]``, where ``index`` may have a single shared row."""
    if index.shape[0] == 1:
        return values[:, index[0]]
    return np.take_along_axis(values, index, axis=1)


def _sort_rows(
    y: FloatArray, scores: FloatArray, w: FloatArray, *, descending: bool
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Sort labels and weights by score within each row and locate the tie groups.

    When the scores are shared by every row (the bootstrap case) they are sorted once and only
    the weight matrix is permuted, which keeps a bootstrap at O(rows * n) after one sort.
    """
    s2, w2 = _rows(scores), _rows(w)
    key = -s2 if descending else s2
    if s2.shape[0] == 1:
        order = np.argsort(key[0], kind="stable")
        sorted_scores = s2[:, order]
        sorted_labels = y[order][np.newaxis, :]
        sorted_weights = w2[:, order]
    else:
        order_rows = np.argsort(key, axis=1, kind="stable")
        sorted_scores = np.take_along_axis(s2, order_rows, axis=1)
        sorted_labels = y[order_rows]
        sorted_weights = np.take_along_axis(np.broadcast_to(w2, s2.shape), order_rows, axis=1)
    start, end = _tie_bounds(sorted_scores)
    return sorted_labels, sorted_weights, start, end


def _roc_auc_batch(y: FloatArray, scores: FloatArray, w: FloatArray) -> FloatArray:
    # AUC = P(score of a positive > score of a negative) + 0.5 P(tie), with weights. This is
    # the area under the ROC curve computed with the trapezoidal rule.
    labels, weights, start, end = _sort_rows(y, scores, w, descending=False)
    positive = weights * labels
    negative = weights - positive
    rows, n = negative.shape
    negatives_before = np.zeros((rows, n + 1))
    np.cumsum(negative, axis=1, out=negatives_before[:, 1:])
    below = _gather(negatives_before, start)  # negative weight with a strictly lower score
    through = _gather(negatives_before, end + 1)  # negative weight with a lower or equal score
    numerator = _sum_rows(positive * 0.5 * (below + through))
    return _ratio(numerator, _sum_rows(positive) * _sum_rows(negative))


def _average_precision_batch(y: FloatArray, scores: FloatArray, w: FloatArray) -> FloatArray:
    # AP = sum over distinct thresholds of (recall increase) * precision at that threshold.
    # Every positive in a tie group shares the precision computed at the end of the group.
    labels, weights, _, end = _sort_rows(y, scores, w, descending=True)
    positive = weights * labels
    true_pos = _gather(np.cumsum(positive, axis=1), end)
    predicted_pos = _gather(np.cumsum(weights, axis=1), end)
    precision = np.zeros(true_pos.shape)
    np.divide(true_pos, predicted_pos, out=precision, where=predicted_pos > 0)
    return _ratio(_sum_rows(positive * precision), _sum_rows(positive))


# --------------------------------------------------------------------------------------------
# Metric objects and the registry
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A metric together with what it needs from its inputs.

    Attributes:
        name: Registry name, used in reports.
        kind: ``"label"`` for metrics on 0/1 predictions, ``"score"`` for metrics on
            continuous scores of a binary outcome, ``"regression"`` for real-valued targets.
        higher_is_better: Direction of improvement, used to word reports.
        batch: The vectorised implementation described in the module docstring.
        undefined_when: Human-readable condition under which the metric has no value.
        needs_probabilities: Scores must lie in [0, 1] (log loss, Brier score).
    """

    name: str
    kind: MetricKind
    higher_is_better: bool
    batch: BatchFn
    undefined_when: str
    needs_probabilities: bool = False

    @property
    def is_classification(self) -> bool:
        return self.kind != "regression"

    def check_inputs(self, y_true: FloatArray, y_pred: FloatArray, name: str = "y_pred") -> None:
        """Raise ``ValueError`` if validated float arrays do not suit this metric."""
        if self.is_classification:
            check_binary(y_true, "y_true")
        if self.kind == "label":
            check_binary(y_pred, f"{name} (predicted labels for {self.name})")
        if self.needs_probabilities:
            check_probabilities(y_pred, name)

    def __call__(
        self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> float:
        """Compute the metric on one set of predictions."""
        y = as_float_vector(y_true, "y_true")
        pred = as_float_vector(y_pred, "y_pred")
        n = check_same_length(y_true=y, y_pred=pred)
        weights = as_weights(sample_weight, n)
        self.check_inputs(y, pred)
        return self.value(y, pred, weights)

    def value(self, y: FloatArray, pred: FloatArray, weights: FloatArray) -> float:
        """Point value on validated arrays; raises ``ValueError`` if it is undefined."""
        result = float(self.batch(y, pred, weights)[0])
        if np.isnan(result):
            raise ValueError(f"{self.name} is undefined: {self.undefined_when}")
        return result


ACCURACY = Metric("accuracy", "label", True, _accuracy_batch, "the weights sum to zero")
PRECISION = Metric(
    "precision", "label", True, _precision_batch, "there are no positive predictions"
)
RECALL = Metric("recall", "label", True, _recall_batch, "y_true has no positive examples")
F1 = Metric("f1", "label", True, _f1_batch, "there are no positive labels or predictions")
ROC_AUC = Metric("auc", "score", True, _roc_auc_batch, "y_true contains a single class")
AVERAGE_PRECISION = Metric(
    "average_precision", "score", True, _average_precision_batch, "y_true has no positive examples"
)
LOG_LOSS = Metric(
    "log_loss",
    "score",
    False,
    _log_loss_batch,
    "the weights sum to zero",
    needs_probabilities=True,
)
BRIER = Metric(
    "brier", "score", False, _brier_batch, "the weights sum to zero", needs_probabilities=True
)
MAE = Metric("mae", "regression", False, _mae_batch, "the weights sum to zero")
RMSE = Metric("rmse", "regression", False, _rmse_batch, "the weights sum to zero")
R2 = Metric("r2", "regression", True, _r2_batch, "y_true is constant")

METRICS: Mapping[str, Metric] = {
    metric.name: metric
    for metric in (
        ACCURACY,
        PRECISION,
        RECALL,
        F1,
        ROC_AUC,
        AVERAGE_PRECISION,
        LOG_LOSS,
        BRIER,
        MAE,
        RMSE,
        R2,
    )
}
_ALIASES = {"roc_auc": "auc", "ap": "average_precision", "brier_score": "brier"}


def get_metric(metric: str | Metric) -> Metric:
    """Return a ``Metric`` from its registry name (or an alias), or the object itself."""
    if isinstance(metric, Metric):
        return metric
    key = _ALIASES.get(metric, metric)
    if key not in METRICS:
        known = ", ".join(sorted(METRICS))
        raise ValueError(f"unknown metric {metric!r}; choose one of: {known}")
    return METRICS[key]


# --------------------------------------------------------------------------------------------
# Function interface
# --------------------------------------------------------------------------------------------


def accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Fraction of correct 0/1 predictions."""
    return ACCURACY(y_true, y_pred, sample_weight)


def precision(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """TP / (TP + FP) for 0/1 predictions."""
    return PRECISION(y_true, y_pred, sample_weight)


def recall(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """TP / (TP + FN) for 0/1 predictions."""
    return RECALL(y_true, y_pred, sample_weight)


def f1(y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None) -> float:
    """Harmonic mean of precision and recall, 2 TP / (2 TP + FP + FN)."""
    return F1(y_true, y_pred, sample_weight)


def roc_auc(
    y_true: ArrayLike, y_score: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Area under the ROC curve; tied scores count one half."""
    return ROC_AUC(y_true, y_score, sample_weight)


def average_precision(
    y_true: ArrayLike, y_score: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Average precision, the step-wise area under the precision-recall curve."""
    return AVERAGE_PRECISION(y_true, y_score, sample_weight)


def log_loss(
    y_true: ArrayLike, y_prob: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Mean negative log-likelihood of the labels under predicted probabilities."""
    return LOG_LOSS(y_true, y_prob, sample_weight)


def brier_score(
    y_true: ArrayLike, y_prob: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Mean squared difference between predicted probability and outcome."""
    return BRIER(y_true, y_prob, sample_weight)


def mean_absolute_error(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    return MAE(y_true, y_pred, sample_weight)


def root_mean_squared_error(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    return RMSE(y_true, y_pred, sample_weight)


def r2_score(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot (weighted)."""
    return R2(y_true, y_pred, sample_weight)
