"""Bootstrap confidence intervals.

A resample is represented by its count vector: ``counts[i]`` is how many times example ``i``
was drawn. Counts for a block of resamples form a ``(rows, n)`` matrix that is passed to a
metric's batch function as weights (see :mod:`evalstats.metrics`), so no resampled copy of the
data is ever built and the work per resample is a few vectorised passes over ``n`` values.
Blocks are sized to about a million matrix entries, which bounds memory for any ``n``.

Intervals are percentile or BCa (bias-corrected and accelerated, Efron 1987). The BCa
acceleration comes from a jackknife: exact leave-one-out up to 1000 examples, and a delete-a-group
jackknife with 1000 random groups above that (see ``_jackknife``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from evalstats._validation import (
    FloatArray,
    IntArray,
    SeedLike,
    as_float_vector,
    as_generator,
    as_weights,
    check_confidence,
    check_positive_int,
    check_same_length,
)
from evalstats.metrics import Metric, get_metric

CIMethod = Literal["percentile", "bca"]
# Maps a (rows, n) matrix of resampling counts to one value per row, or to a (rows, k) matrix.
Statistic = Callable[[FloatArray], FloatArray]

_BLOCK_ELEMENTS = 1 << 20
_JACKKNIFE_GROUPS = 1000


@dataclass(frozen=True)
class BootstrapResult:
    """A metric's point estimate and bootstrap confidence interval.

    ``method`` is the interval actually reported: a BCa request falls back to percentile (with
    a warning) when the estimate lies outside the bootstrap distribution, where BCa is undefined.
    ``n_invalid`` counts resamples on which the metric was undefined (for example AUC on a
    resample with one class); they are excluded from the interval.
    """

    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    method: CIMethod
    standard_error: float
    n: int
    n_resamples: int
    n_invalid: int
    stratified: bool


@dataclass(frozen=True)
class _Summary:
    estimate: float
    ci_low: float
    ci_high: float
    method: CIMethod
    p_value: float
    standard_error: float
    n_invalid: int


def iter_blocks(total: int, n: int) -> Iterator[tuple[int, int]]:
    rows = max(1, _BLOCK_ELEMENTS // n)
    for start in range(0, total, rows):
        yield start, min(rows, total - start)


def _uniform_counts(rows: int, size: int, rng: np.random.Generator) -> FloatArray:
    """``rows`` resamples of ``size`` draws with replacement from ``size`` items, as counts."""
    draws = rng.integers(0, size, size=(rows, size))
    draws += (np.arange(rows) * size)[:, np.newaxis]
    counts = np.bincount(draws.ravel(), minlength=rows * size).reshape(rows, size)
    return counts.astype(np.float64)


def resample_counts(
    rows: int, n: int, strata: Sequence[IntArray] | None, rng: np.random.Generator
) -> FloatArray:
    """Draw ``rows`` bootstrap resamples of ``n`` examples as a ``(rows, n)`` matrix of counts.

    With ``strata`` (a list of disjoint index arrays covering the examples), each stratum is
    resampled on its own, as many draws as it has members, so every resample keeps the stratum
    sizes. Without it, all ``n`` examples form a single stratum.
    """
    if strata is None:
        return _uniform_counts(rows, n, rng)
    counts = np.zeros((rows, n))
    for members in strata:
        counts[:, members] = _uniform_counts(rows, len(members), rng)
    return counts


def bootstrap_replicates(
    statistic: Statistic,
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    strata: Sequence[IntArray] | None = None,
) -> FloatArray:
    """Evaluate ``statistic`` on ``n_resamples`` bootstrap resamples of ``n`` examples."""
    out: FloatArray | None = None
    for start, rows in iter_blocks(n_resamples, n):
        values = statistic(resample_counts(rows, n, strata, rng))
        if out is None:
            out = np.empty((n_resamples, *values.shape[1:]))
        out[start : start + rows] = values
    assert out is not None
    return out


def _jackknife(statistic: Statistic, n: int, rng: np.random.Generator) -> FloatArray:
    """Jackknife values of ``statistic``, one per deleted example or group.

    Leave-one-out costs O(n^2), so above 1000 examples the data are split at random into 1000
    groups and one group is deleted at a time. The BCa acceleration is a ratio of the third
    moment of the jackknife influence values to the 1.5 power of the second; summing the
    influence of the members of each group multiplies both by the group size and the ratio is
    unchanged, so the grouped estimate targets the same quantity.
    """
    if n <= _JACKKNIFE_GROUPS:
        group_of = np.arange(n)
        n_groups = n
    else:
        group_of = np.empty(n, dtype=np.intp)
        for g, members in enumerate(np.array_split(rng.permutation(n), _JACKKNIFE_GROUPS)):
            group_of[members] = g
        n_groups = _JACKKNIFE_GROUPS
    values = np.empty(n_groups)
    for start, rows in iter_blocks(n_groups, n):
        deleted = np.arange(start, start + rows)[:, np.newaxis]
        keep = (group_of[np.newaxis, :] != deleted).astype(np.float64)
        values[start : start + rows] = statistic(keep)
    return values


def _acceleration(jackknife: FloatArray) -> float:
    values = jackknife[np.isfinite(jackknife)]
    if len(values) < 3:
        return 0.0
    influence = values.mean() - values
    second = float(np.sum(influence**2))
    denominator = 6.0 * second * math.sqrt(second)
    if denominator == 0.0:
        return 0.0
    return float(np.sum(influence**3)) / denominator


def _rank_fraction(replicates: FloatArray, value: float) -> float:
    """Fraction of replicates below ``value``, counting ties as one half."""
    below = np.count_nonzero(replicates < value)
    ties = np.count_nonzero(replicates == value)
    return (below + 0.5 * ties) / len(replicates)


def _p_value_from_level(one_sided: float) -> float:
    return float(min(1.0, 2.0 * min(one_sided, 1.0 - one_sided)))


def summarize(
    statistic: Statistic,
    n: int,
    *,
    n_resamples: int,
    confidence: float,
    method: CIMethod,
    rng: np.random.Generator,
    strata: Sequence[IntArray] | None = None,
    label: str = "statistic",
) -> _Summary:
    """Bootstrap ``statistic`` and return its estimate, interval and a p-value for zero.

    The p-value is the smallest ``1 - confidence`` at which the reported interval would exclude
    zero (interval inversion). B resamples cannot resolve p-values below 2/(B+1): the share of
    replicates on either side of zero is kept inside [1/(B+1), B/(B+1)] and the p-value is
    reported as at least 2/(B+1).
    """
    estimate = float(statistic(np.ones((1, n)))[0])
    if not np.isfinite(estimate):
        raise ValueError(f"{label} is undefined on the full sample")
    replicates = bootstrap_replicates(statistic, n, n_resamples, rng, strata)
    finite = replicates[np.isfinite(replicates)]
    n_invalid = n_resamples - len(finite)
    if len(finite) < 2:
        raise ValueError(f"{label} is undefined on almost every resample; try stratify=True")
    if n_invalid:
        warnings.warn(
            f"{n_invalid} of {n_resamples} resamples gave an undefined {label} and were "
            "excluded; consider stratify=True",
            RuntimeWarning,
            stacklevel=3,
        )
    alpha = 1.0 - confidence
    count = len(finite)
    zero_level = float(np.clip(_rank_fraction(finite, 0.0), 1 / (count + 1), count / (count + 1)))
    used: CIMethod = method
    if method == "bca":
        position = _rank_fraction(finite, estimate)
        if position in (0.0, 1.0):
            warnings.warn(
                f"the {label} estimate lies outside its bootstrap distribution, where BCa is "
                "undefined; reporting a percentile interval",
                RuntimeWarning,
                stacklevel=3,
            )
            used = "percentile"
        else:
            bias = float(ndtri(position))
            accel = _acceleration(_jackknife(statistic, n, rng))
            z = ndtri(np.array([alpha / 2.0, 1.0 - alpha / 2.0]))
            levels = ndtr(bias + (bias + z) / (1.0 - accel * (bias + z)))
            low, high = np.quantile(finite, levels)
            # Invert the same mapping at zero for a p-value consistent with the interval.
            w = float(ndtri(zero_level)) - bias
            if 1.0 + accel * w > 0.0:
                zero_level = float(ndtr(w / (1.0 + accel * w) - bias))
    if used == "percentile":
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return _Summary(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        method=used,
        p_value=max(_p_value_from_level(zero_level), 2.0 / (count + 1)),
        standard_error=float(np.std(finite, ddof=1)),
        n_invalid=n_invalid,
    )


def class_strata(y: FloatArray) -> list[IntArray]:
    """Index arrays of the negatives and the positives (empty classes omitted)."""
    return [
        members for members in (np.flatnonzero(y == 0), np.flatnonzero(y == 1)) if len(members)
    ]


def check_bootstrap_args(n_resamples: int, confidence: float, method: str) -> None:
    check_positive_int(n_resamples, "n_resamples")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be at least 100, got {n_resamples}")
    check_confidence(confidence)
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")


def bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: str | Metric,
    *,
    sample_weight: ArrayLike | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    method: CIMethod = "bca",
    stratify: bool = False,
    seed: SeedLike = None,
) -> BootstrapResult:
    """Confidence interval for a metric by the nonparametric bootstrap.

    Args:
        y_true: True labels (0/1) or regression targets.
        y_pred: Predicted labels, scores or values, as the metric expects.
        metric: A registry name such as ``"auc"`` or a :class:`~evalstats.metrics.Metric`.
        sample_weight: Optional non-negative weights; a resample multiplies them by its counts.
        n_resamples: Number of bootstrap resamples (at least 100).
        confidence: Coverage of the interval, in (0, 1).
        method: ``"bca"`` (default) or ``"percentile"``.
        stratify: Resample positives and negatives separately, keeping the class counts fixed.
            Classification metrics only. Useful when a class is rare.
        seed: Seed or ``numpy.random.Generator`` for reproducible results.
    """
    spec = get_metric(metric)
    check_bootstrap_args(n_resamples, confidence, method)
    y = as_float_vector(y_true, "y_true")
    pred = as_float_vector(y_pred, "y_pred")
    n = check_same_length(y_true=y, y_pred=pred)
    weights = as_weights(sample_weight, n)
    spec.check_inputs(y, pred)
    spec.value(y, pred, weights)  # raises a clear error if undefined on the full sample
    if stratify and not spec.is_classification:
        raise ValueError("stratify=True needs a classification metric")
    strata = class_strata(y) if stratify else None

    def statistic(counts: FloatArray) -> FloatArray:
        return spec.batch(y, pred, counts * weights)

    summary = summarize(
        statistic,
        n,
        n_resamples=n_resamples,
        confidence=confidence,
        method=method,
        rng=as_generator(seed),
        strata=strata,
        label=spec.name,
    )
    return BootstrapResult(
        metric=spec.name,
        estimate=summary.estimate,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
        confidence=confidence,
        method=summary.method,
        standard_error=summary.standard_error,
        n=n,
        n_resamples=n_resamples,
        n_invalid=summary.n_invalid,
        stratified=stratify,
    )
