"""Slice analysis: a metric per value of a categorical column, compared with the rest.

For every slice with enough examples the report gives the metric with a bootstrap interval, the
metric on all other examples ("the rest"), the difference with its interval, and a p-value for
"this slice differs from the rest", adjusted across slices with Holm or Benjamini-Hochberg.

One bootstrap serves every slice: examples are resampled within each slice (and within each
class when ``stratify=True``), so slice sizes are fixed and a slice and its rest are resampled
independently. Intervals are percentile intervals and the p-value inverts the difference
interval, as in :func:`evalstats.paired.paired_bootstrap`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from evalstats._validation import (
    FloatArray,
    IntArray,
    SeedLike,
    as_float_vector,
    as_generator,
    as_weights,
    check_positive_int,
    check_same_length,
)
from evalstats.bootstrap import bootstrap_replicates, check_bootstrap_args, percentile_interval
from evalstats.metrics import Metric, get_metric
from evalstats.multiple import Adjustment, adjust_p_values


@dataclass(frozen=True)
class SliceResult:
    """One slice compared with the rest of the data."""

    name: str
    n: int
    estimate: float
    ci_low: float
    ci_high: float
    rest_estimate: float
    difference: float  # estimate - rest_estimate
    difference_ci_low: float
    difference_ci_high: float
    p_value: float
    p_adjusted: float


@dataclass(frozen=True)
class SkippedSlice:
    """A slice left out of the comparison, and why."""

    name: str
    n: int
    reason: str


@dataclass(frozen=True)
class SliceReport:
    """Result of :func:`slice_analysis`. ``slices`` are in sorted order of their names."""

    metric: str
    overall: float
    confidence: float
    adjustment: Adjustment
    n_resamples: int
    slices: tuple[SliceResult, ...]
    skipped: tuple[SkippedSlice, ...]

    def flagged(self, alpha: float = 0.05) -> tuple[SliceResult, ...]:
        """Slices whose adjusted p-value is at most ``alpha``."""
        return tuple(s for s in self.slices if s.p_adjusted <= alpha)


def _slice_codes(groups: ArrayLike, n: int) -> tuple[list[str], IntArray]:
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"groups must be one-dimensional with length {n}")
    names, codes = np.unique(np.array([str(v) for v in values]), return_inverse=True)
    return [str(name) for name in names], np.asarray(codes, dtype=np.intp)


def slice_analysis(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike,
    metric: str | Metric,
    *,
    min_support: int = 30,
    sample_weight: ArrayLike | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    adjust: Adjustment = "holm",
    stratify: bool = False,
    seed: SeedLike = None,
) -> SliceReport:
    """Metric per slice with intervals and adjusted p-values against the rest.

    Args:
        y_true, y_pred: Labels and predictions, as the metric expects.
        groups: Slice of each example; values are compared by their string form.
        metric: A registry name or :class:`~evalstats.metrics.Metric`.
        min_support: Slices (and rests) with fewer examples are skipped. They still count in
            the rest of the other slices.
        sample_weight: Optional non-negative weights.
        n_resamples: Bootstrap resamples shared by all slices.
        confidence: Coverage of the intervals.
        adjust: ``"holm"`` (family-wise error) or ``"bh"`` (false discovery rate).
        stratify: Also keep the class counts of each slice fixed (classification only).
        seed: Seed or ``numpy.random.Generator``.
    """
    spec = get_metric(metric)
    check_bootstrap_args(n_resamples, confidence, "percentile")
    check_positive_int(min_support, "min_support")
    if adjust not in ("holm", "bh"):
        raise ValueError(f"adjust must be 'holm' or 'bh', got {adjust!r}")
    if stratify and not spec.is_classification:
        raise ValueError("stratify=True needs a classification metric")
    y = as_float_vector(y_true, "y_true")
    pred = as_float_vector(y_pred, "y_pred")
    n = check_same_length(y_true=y, y_pred=pred)
    weights = as_weights(sample_weight, n)
    spec.check_inputs(y, pred)
    overall = spec.value(y, pred, weights)
    names, codes = _slice_codes(groups, n)

    eligible: list[int] = []
    skipped: list[SkippedSlice] = []
    for code, name in enumerate(names):
        members = codes == code
        size = int(np.count_nonzero(members))
        if size < min_support:
            skipped.append(SkippedSlice(name, size, f"fewer than {min_support} examples"))
            continue
        if n - size < min_support:
            skipped.append(SkippedSlice(name, size, f"the rest has fewer than {min_support}"))
            continue
        undefined = [
            part
            for part, mask in (("slice", members), ("rest", ~members))
            if np.isnan(spec.batch(y[mask], pred[mask], weights[mask])[0])
        ]
        if undefined:
            reason = f"{spec.name} is undefined on the {undefined[0]}: {spec.undefined_when}"
            skipped.append(SkippedSlice(name, size, reason))
            continue
        eligible.append(code)

    strata = [
        members
        for code in range(len(names))
        for members in (
            [np.flatnonzero((codes == code) & (y == c)) for c in (0.0, 1.0)]
            if stratify
            else [np.flatnonzero(codes == code)]
        )
        if len(members)
    ]
    masks = [codes == code for code in eligible]

    def statistic(counts: FloatArray) -> FloatArray:
        resampled = counts * weights
        columns = []
        for mask in masks:
            rest = ~mask
            inside = spec.batch(y[mask], pred[mask], resampled[:, mask])
            outside = spec.batch(y[rest], pred[rest], resampled[:, rest])
            columns.extend([inside, inside - outside])
        return np.column_stack(columns) if columns else np.empty((len(counts), 0))

    replicates = bootstrap_replicates(statistic, n, n_resamples, as_generator(seed), strata)
    raw: list[tuple[int, float, float, float, float, float, float, float]] = []
    n_invalid = 0
    for position, code in enumerate(eligible):
        mask = masks[position]
        slice_reps = replicates[:, 2 * position]
        diff_reps = replicates[:, 2 * position + 1]
        valid = np.isfinite(diff_reps)
        n_invalid += int(np.count_nonzero(~valid))
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                f"{spec.name} is undefined on almost every resample of slice {names[code]!r}; "
                "try stratify=True or a larger min_support"
            )
        low, high, _ = percentile_interval(slice_reps[valid], confidence)
        diff_low, diff_high, p_value = percentile_interval(diff_reps[valid], confidence)
        estimate = spec.value(y[mask], pred[mask], weights[mask])
        rest = spec.value(y[~mask], pred[~mask], weights[~mask])
        raw.append((code, estimate, low, high, rest, diff_low, diff_high, p_value))
    if n_invalid:
        warnings.warn(
            f"{n_invalid} slice resamples gave an undefined {spec.name} and were excluded; "
            "consider stratify=True",
            RuntimeWarning,
            stacklevel=2,
        )

    adjusted = adjust_p_values([row[-1] for row in raw], adjust)
    results = tuple(
        SliceResult(
            name=names[code],
            n=int(np.count_nonzero(codes == code)),
            estimate=estimate,
            ci_low=low,
            ci_high=high,
            rest_estimate=rest,
            difference=estimate - rest,
            difference_ci_low=diff_low,
            difference_ci_high=diff_high,
            p_value=p_value,
            p_adjusted=float(p_adj),
        )
        for (code, estimate, low, high, rest, diff_low, diff_high, p_value), p_adj in zip(
            raw, adjusted, strict=True
        )
    )
    return SliceReport(
        metric=spec.name,
        overall=overall,
        confidence=confidence,
        adjustment=adjust,
        n_resamples=n_resamples,
        slices=results,
        skipped=tuple(skipped),
    )
