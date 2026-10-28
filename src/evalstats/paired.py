"""Paired comparisons of two models evaluated on the same examples.

All functions return a :class:`PairedResult` whose ``difference`` is model A minus model B, so
a positive difference favours A for metrics where higher is better.

* :func:`paired_bootstrap`: any metric; CI of the difference and the p-value that inverts it.
* :func:`permutation_test`: any metric; exact-size test of "A and B are exchangeable".
* :func:`mcnemar`: accuracy of 0/1 predictions; exact binomial or mid-p test.
* :func:`delong_test`: ROC AUC; DeLong's asymptotic test for correlated AUCs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri
from scipy.stats import binom, rankdata

from evalstats._validation import (
    FloatArray,
    SeedLike,
    as_float_vector,
    as_generator,
    as_weights,
    check_binary,
    check_confidence,
    check_positive_int,
    check_same_length,
)
from evalstats.bootstrap import (
    CIMethod,
    check_bootstrap_args,
    class_strata,
    iter_blocks,
    summarize,
)
from evalstats.metrics import ACCURACY, ROC_AUC, Metric, get_metric


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired comparison.

    Attributes:
        method: Name of the procedure, for example ``"delong"``.
        metric: Metric compared.
        estimate_a, estimate_b: Metric values of the two models on the full sample.
        difference: ``estimate_a - estimate_b``.
        ci_low, ci_high: Confidence interval for the difference, or None when the procedure
            gives none (the permutation test).
        confidence: Coverage of the interval, or None without one.
        p_value: Two-sided p-value for "no difference".
        statistic: The test statistic where there is a standard one (z for DeLong).
        n: Number of examples.
        details: Procedure-specific quantities (counts, variances, resample counts).
    """

    method: str
    metric: str
    estimate_a: float
    estimate_b: float
    difference: float
    ci_low: float | None
    ci_high: float | None
    confidence: float | None
    p_value: float
    statistic: float | None
    n: int
    details: Mapping[str, float | int | str] = field(default_factory=dict)


def _paired_arrays(
    spec: Metric,
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    y = as_float_vector(y_true, "y_true")
    a = as_float_vector(pred_a, "pred_a")
    b = as_float_vector(pred_b, "pred_b")
    n = check_same_length(y_true=y, pred_a=a, pred_b=b)
    weights = as_weights(sample_weight, n)
    spec.check_inputs(y, a, "pred_a")
    spec.check_inputs(y, b, "pred_b")
    return y, a, b, weights


def paired_bootstrap(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    metric: str | Metric,
    *,
    sample_weight: ArrayLike | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    method: CIMethod = "bca",
    stratify: bool = False,
    seed: SeedLike = None,
) -> PairedResult:
    """Bootstrap CI of ``metric(A) - metric(B)``, resampling examples jointly for both models.

    Resampling the same examples for both models keeps the correlation between their errors,
    which is what makes a paired interval narrower than two separate ones. The p-value is the
    smallest ``1 - confidence`` at which the interval would exclude zero; with B resamples it
    cannot go below 2/(B+1). It is an approximation; :func:`permutation_test` has exact size.
    """
    spec = get_metric(metric)
    check_bootstrap_args(n_resamples, confidence, method)
    y, a, b, weights = _paired_arrays(spec, y_true, pred_a, pred_b, sample_weight)
    if stratify and not spec.is_classification:
        raise ValueError("stratify=True needs a classification metric")
    estimate_a = spec.value(y, a, weights)
    estimate_b = spec.value(y, b, weights)

    def statistic(counts: FloatArray) -> FloatArray:
        resampled = counts * weights
        return spec.batch(y, a, resampled) - spec.batch(y, b, resampled)

    summary = summarize(
        statistic,
        len(y),
        n_resamples=n_resamples,
        confidence=confidence,
        method=method,
        rng=as_generator(seed),
        strata=class_strata(y) if stratify else None,
        label=f"{spec.name} difference",
    )
    return PairedResult(
        method="paired_bootstrap",
        metric=spec.name,
        estimate_a=estimate_a,
        estimate_b=estimate_b,
        difference=summary.estimate,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
        confidence=confidence,
        p_value=summary.p_value,
        statistic=None,
        n=len(y),
        details={
            "ci_method": summary.method,
            "standard_error": summary.standard_error,
            "n_resamples": n_resamples,
            "n_invalid": summary.n_invalid,
            "stratified": int(stratify),
        },
    )


def permutation_test(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    metric: str | Metric,
    *,
    sample_weight: ArrayLike | None = None,
    n_permutations: int = 10_000,
    seed: SeedLike = None,
) -> PairedResult:
    """Paired permutation test of ``metric(A) - metric(B)``.

    Under the null hypothesis the two predictions for each example are exchangeable, so each
    pair is swapped with probability one half. The p-value counts permuted differences at least
    as large in absolute value as the observed one, plus one for the observed arrangement, over
    ``n_permutations + 1``; this Monte Carlo p-value has exact size (it never rejects a true null
    more often than the nominal level).
    """
    spec = get_metric(metric)
    check_positive_int(n_permutations, "n_permutations")
    y, a, b, weights = _paired_arrays(spec, y_true, pred_a, pred_b, sample_weight)
    estimate_a = spec.value(y, a, weights)
    estimate_b = spec.value(y, b, weights)
    observed = estimate_a - estimate_b
    # Equal statistics can differ in the last bits after summing in another order.
    threshold = abs(observed) - 1e-10 * max(1.0, abs(observed))
    rng = as_generator(seed)
    n = len(y)
    at_least_as_extreme = 0
    n_undefined = 0
    for _, rows in iter_blocks(n_permutations, n):
        swap = rng.random((rows, n)) < 0.5
        permuted = spec.batch(y, np.where(swap, b, a), weights) - spec.batch(
            y, np.where(swap, a, b), weights
        )
        n_undefined += int(np.count_nonzero(np.isnan(permuted)))
        at_least_as_extreme += int(np.count_nonzero(np.abs(permuted) >= threshold))
    # A permutation keeps y and the weights, so the metric stays defined; guard anyway.
    if n_undefined:
        raise ValueError(f"{spec.name} was undefined on {n_undefined} permutations")
    return PairedResult(
        method="permutation",
        metric=spec.name,
        estimate_a=estimate_a,
        estimate_b=estimate_b,
        difference=observed,
        ci_low=None,
        ci_high=None,
        confidence=None,
        p_value=(at_least_as_extreme + 1) / (n_permutations + 1),
        statistic=observed,
        n=n,
        details={"n_permutations": n_permutations},
    )


def mcnemar(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    *,
    variant: Literal["mid-p", "exact"] = "mid-p",
    confidence: float = 0.95,
) -> PairedResult:
    """McNemar's test for the accuracy of two sets of 0/1 predictions.

    Only discordant examples carry information: ``n10`` where A is right and B wrong, ``n01``
    the reverse. Under the null each discordant example is equally likely to go either way, so
    ``n10 ~ Binomial(n10 + n01, 1/2)``. ``"exact"`` doubles the smaller tail and is
    conservative; ``"mid-p"`` counts half the probability of the observed value and keeps the
    size close to nominal (Fagerland, Lydersen and Laake, 2013).

    The confidence interval for the accuracy difference is the Agresti-Min (2005) adjusted Wald
    interval: add one half to every cell of the 2x2 table, then use the Wald formula.
    """
    if variant not in ("mid-p", "exact"):
        raise ValueError(f"variant must be 'mid-p' or 'exact', got {variant!r}")
    check_confidence(confidence)
    y, a, b, _ = _paired_arrays(ACCURACY, y_true, pred_a, pred_b)
    n = len(y)
    correct_a = a == y
    correct_b = b == y
    n10 = int(np.count_nonzero(correct_a & ~correct_b))
    n01 = int(np.count_nonzero(~correct_a & correct_b))
    discordant = n10 + n01
    smaller = min(n10, n01)
    if discordant == 0:
        p_value = 1.0
    else:
        tail = float(binom.cdf(smaller, discordant, 0.5))
        if variant == "mid-p":
            tail -= 0.5 * float(binom.pmf(smaller, discordant, 0.5))
        p_value = min(1.0, 2.0 * tail)

    adj_n10, adj_n01, adj_n = n10 + 0.5, n01 + 0.5, n + 2.0
    centre = (adj_n10 - adj_n01) / adj_n
    variance = ((adj_n10 + adj_n01) - (adj_n10 - adj_n01) ** 2 / adj_n) / adj_n**2
    half_width = float(ndtri(0.5 + confidence / 2.0)) * np.sqrt(variance)
    return PairedResult(
        method=f"mcnemar_{variant.replace('-', '_')}",
        metric=ACCURACY.name,
        estimate_a=float(np.mean(correct_a)),
        estimate_b=float(np.mean(correct_b)),
        difference=(n10 - n01) / n,
        ci_low=max(-1.0, centre - half_width),
        ci_high=min(1.0, centre + half_width),
        confidence=confidence,
        p_value=p_value,
        statistic=float(n10),
        n=n,
        details={"a_right_b_wrong": n10, "a_wrong_b_right": n01},
    )


def delong_covariance(y_true: ArrayLike, scores: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """AUCs of several score vectors on the same examples and DeLong's covariance estimate.

    Args:
        y_true: 0/1 labels, length ``n``, with at least two of each class.
        scores: Array of shape ``(k, n)``, one row per model.

    Returns:
        ``(aucs, covariance)`` with shapes ``(k,)`` and ``(k, k)``.

    The AUC is the mean of the kernel psi(x, y) = 1{x > y} + 1/2 1{x = y} over all
    (positive, negative) pairs. DeLong, DeLong and Clarke-Pearson (1988) estimate the covariance
    of the AUCs from the placement values (the kernel averaged over the other class) as
    ``S10 / m + S01 / n0``, where S10 and S01 are the sample covariances of the placements of
    the m positives and the n0 negatives. The placements are computed from midranks in
    O(n log n) per model (Sun and Xu, 2014).
    """
    y = as_float_vector(y_true, "y_true")
    check_binary(y, "y_true")
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(y):
        raise ValueError(f"scores must have shape (k, {len(y)}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scores contain NaN or infinite values")
    positive = matrix[:, y == 1]
    negative = matrix[:, y == 0]
    m, n0 = positive.shape[1], negative.shape[1]
    if m < 2 or n0 < 2:
        raise ValueError("DeLong's covariance needs at least two positives and two negatives")
    rank_pos = rankdata(positive, axis=1)
    rank_neg = rankdata(negative, axis=1)
    rank_all = rankdata(np.concatenate([positive, negative], axis=1), axis=1)
    # Placement of a positive: share of negatives below it (ties count one half), and so on.
    placement_pos = (rank_all[:, :m] - rank_pos) / n0
    placement_neg = 1.0 - (rank_all[:, m:] - rank_neg) / m
    aucs = placement_pos.mean(axis=1)
    covariance = (
        np.atleast_2d(np.cov(placement_pos)) / m + np.atleast_2d(np.cov(placement_neg)) / n0
    )
    return np.asarray(aucs, dtype=np.float64), np.asarray(covariance, dtype=np.float64)


def delong_test(
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    *,
    confidence: float = 0.95,
) -> PairedResult:
    """DeLong's test for the difference between two correlated ROC AUCs.

    The difference of the AUCs is asymptotically normal with variance
    ``Var(A) + Var(B) - 2 Cov(A, B)`` from :func:`delong_covariance`; the test statistic is
    the difference over its standard error, and the interval is the matching Wald interval.
    """
    check_confidence(confidence)
    y, a, b, _ = _paired_arrays(ROC_AUC, y_true, score_a, score_b)
    aucs, cov = delong_covariance(y, np.vstack([a, b]))
    difference = float(aucs[0] - aucs[1])
    variance = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    # Rounding can leave a tiny negative variance when the two score vectors rank alike.
    variance = max(variance, 0.0)
    std_error = float(np.sqrt(variance))
    if std_error > 1e-15:
        z = difference / std_error
        p_value = float(2.0 * ndtr(-abs(z)))
    else:  # identical rankings, or perfect separation by both models
        z = 0.0 if difference == 0.0 else float(np.sign(difference) * np.inf)
        p_value = 1.0 if difference == 0.0 else 0.0
    half_width = float(ndtri(0.5 + confidence / 2.0)) * std_error
    sd_a, sd_b = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    correlation = float(cov[0, 1] / (sd_a * sd_b)) if sd_a > 0 and sd_b > 0 else float("nan")
    return PairedResult(
        method="delong",
        metric=ROC_AUC.name,
        estimate_a=float(aucs[0]),
        estimate_b=float(aucs[1]),
        difference=difference,
        ci_low=difference - half_width,
        ci_high=difference + half_width,
        confidence=confidence,
        p_value=p_value,
        statistic=z,
        n=len(y),
        details={"variance": variance, "auc_correlation": correlation},
    )
