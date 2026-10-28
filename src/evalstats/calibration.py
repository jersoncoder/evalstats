"""Calibration of predicted probabilities for a binary outcome.

Measurement:

* :func:`reliability_curve`: per-bin mean predicted probability against observed frequency.
* :func:`expected_calibration_error`: the bin-weighted mean gap between the two.

Bins are right-closed, ``(e_k, e_k+1]`` with the first bin also holding ``e_0``, the convention
of ``sklearn.calibration.calibration_curve``. ``"uniform"`` bins have equal width on [0, 1];
``"quantile"`` bins have edges at the quantiles of the predictions, so they hold about the same
number of examples (tied predictions can make some quantile bins empty; empty bins are dropped).

Recalibration, fitted by maximum likelihood on a validation split and applied to new data:

* :func:`fit_temperature`: one parameter, ``p' = sigmoid(logit(p) / T)``; keeps the ranking.
* :func:`fit_platt`: two parameters, ``p' = sigmoid(a logit(p) + b)``; also corrects a bias.

Both are logistic regressions on the logit of the input, which have a unique maximum when one
exists. When it does not (the classes are separated by the scores, so the likelihood keeps
improving as the scale grows) the fit raises ``ValueError`` instead of returning a huge value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import expit

from evalstats._validation import (
    FloatArray,
    IntArray,
    as_float_vector,
    check_binary,
    check_positive_int,
    check_probabilities,
    check_same_length,
)

BinStrategy = Literal["uniform", "quantile"]

# Probabilities are clipped to [CLIP, 1 - CLIP] before taking the logit (about +-27.6).
LOGIT_CLIP = 1e-12


def _binary_and_probabilities(
    y_true: ArrayLike, y_prob: ArrayLike, *, logits: bool = False
) -> tuple[FloatArray, FloatArray]:
    y = as_float_vector(y_true, "y_true")
    prob = as_float_vector(y_prob, "y_logit" if logits else "y_prob")
    check_same_length(y_true=y, y_prob=prob)
    check_binary(y, "y_true")
    if not logits:
        check_probabilities(prob, "y_prob")
    return y, prob


def logit(y_prob: ArrayLike) -> FloatArray:
    """Log-odds of probabilities, clipped to [1e-12, 1 - 1e-12] first."""
    p = np.clip(np.asarray(y_prob, dtype=np.float64), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)


# ------------------------------------------------------------------------------ measurement


@dataclass(frozen=True)
class ReliabilityCurve:
    """Calibration data per bin. Arrays list the non-empty bins only.

    Attributes:
        bin_edges: The ``n_bins + 1`` edges (quantile edges can repeat).
        bin_index: Position of each listed bin among the ``n_bins``.
        mean_predicted: Mean predicted probability in the bin.
        observed_frequency: Fraction of positives in the bin.
        counts: Number of examples in the bin.
    """

    bin_edges: FloatArray
    bin_index: IntArray
    mean_predicted: FloatArray
    observed_frequency: FloatArray
    counts: IntArray


def reliability_curve(
    y_true: ArrayLike, y_prob: ArrayLike, *, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> ReliabilityCurve:
    """Mean predicted probability and observed frequency of positives per bin."""
    check_positive_int(n_bins, "n_bins")
    y, prob = _binary_and_probabilities(y_true, y_prob)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        percentiles = np.linspace(0.0, 1.0, n_bins + 1) * 100.0
        edges = np.asarray(np.percentile(prob, percentiles), dtype=np.float64)
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
    bins = np.searchsorted(edges[1:-1], prob, side="left")
    counts = np.bincount(bins, minlength=n_bins)
    sums = np.bincount(bins, weights=prob, minlength=n_bins)
    positives = np.bincount(bins, weights=y, minlength=n_bins)
    occupied = np.flatnonzero(counts)
    return ReliabilityCurve(
        bin_edges=edges,
        bin_index=occupied,
        mean_predicted=sums[occupied] / counts[occupied],
        observed_frequency=positives[occupied] / counts[occupied],
        counts=counts[occupied],
    )


def expected_calibration_error(
    y_true: ArrayLike, y_prob: ArrayLike, *, n_bins: int = 15, strategy: BinStrategy = "uniform"
) -> float:
    """Expected calibration error: sum over bins of (bin share) * |mean prediction - frequency|.

    The binned estimate is biased upward when bins hold few examples, because sampling noise in
    each bin's frequency adds to the gap. Compare models with the same bins and similar sizes.
    """
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    gaps = np.abs(curve.mean_predicted - curve.observed_frequency)
    return float(np.average(gaps, weights=curve.counts))


# ---------------------------------------------------------------------------- recalibration


@dataclass(frozen=True)
class TemperatureScaling:
    """Fitted temperature scaling: ``sigmoid(z / temperature)`` with ``z`` the input logit."""

    temperature: float
    logits: bool = False

    def transform(self, scores: ArrayLike) -> FloatArray:
        """Recalibrated probabilities for probabilities (or logits, if fitted on logits)."""
        z = _as_logits(scores, self.logits)
        return np.asarray(expit(z / self.temperature), dtype=np.float64)


@dataclass(frozen=True)
class PlattScaling:
    """Fitted Platt scaling: ``sigmoid(slope * z + intercept)`` with ``z`` the input logit."""

    slope: float
    intercept: float
    logits: bool = False

    def transform(self, scores: ArrayLike) -> FloatArray:
        """Recalibrated probabilities for probabilities (or logits, if fitted on logits)."""
        z = _as_logits(scores, self.logits)
        return np.asarray(expit(self.slope * z + self.intercept), dtype=np.float64)


def _as_logits(scores: ArrayLike, logits: bool) -> FloatArray:
    values = as_float_vector(scores, "scores")
    if logits:
        return values
    check_probabilities(values, "scores")
    return logit(values)


def _negative_log_likelihood(t: FloatArray, y: FloatArray) -> float:
    # sum of log(1 + exp(t)) - y t, computed without overflow.
    return float(np.sum(np.logaddexp(0.0, t) - y * t))


def fit_temperature(
    y_true: ArrayLike, y_prob: ArrayLike, *, logits: bool = False
) -> TemperatureScaling:
    """Fit a temperature by maximum likelihood.

    Args:
        y_true: 0/1 labels of a validation split.
        y_prob: Predicted probabilities on the same split (or logits with ``logits=True``).
        logits: Treat ``y_prob`` as real-valued logits or decision scores.

    With ``u = 1 / T``, the negative log-likelihood is convex in ``u`` and its derivative
    ``sum((sigmoid(u z) - y) z)`` is increasing, so the maximum is the root of the derivative,
    found by bracketing. A root with ``u > 0`` exists only if the derivative is negative at
    zero (scores positively associated with the labels) and some example is on the wrong side
    of zero (otherwise the likelihood improves without bound as ``T -> 0``).
    """
    y, raw = _binary_and_probabilities(y_true, y_prob, logits=logits)
    z = raw if logits else logit(raw)

    def derivative(u: float) -> float:
        return float(np.sum((expit(u * z) - y) * z))

    if derivative(0.0) >= 0.0:
        raise ValueError(
            "the scores are not positively associated with the labels; "
            "temperature scaling needs scores that rank positives higher"
        )
    if not np.any(((z > 0) & (y == 0)) | ((z < 0) & (y == 1))):
        raise ValueError(
            "every example is on the correct side of probability 0.5, so the likelihood has no "
            "maximum (the fitted temperature would go to zero)"
        )
    upper = 1.0
    while derivative(upper) < 0.0:
        upper *= 2.0
    inverse_temperature = brentq(derivative, 0.0, upper, xtol=1e-14, rtol=1e-14, maxiter=500)
    return TemperatureScaling(temperature=1.0 / float(inverse_temperature), logits=logits)


def fit_platt(y_true: ArrayLike, y_prob: ArrayLike, *, logits: bool = False) -> PlattScaling:
    """Fit Platt scaling (slope and intercept on the logit) by maximum likelihood.

    Plain maximum likelihood, without the label smoothing of Platt's original paper. The
    maximum exists when neither class's scores lie entirely at or above the other's; this is
    checked first. Newton's method with step halving then converges from (1, 0) because the
    negative log-likelihood is strictly convex.
    """
    y, raw = _binary_and_probabilities(y_true, y_prob, logits=logits)
    z = raw if logits else logit(raw)
    positives, negatives = z[y == 1], z[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("Platt scaling needs both classes in y_true")
    overlap = negatives.max() > positives.min() and positives.max() > negatives.min()
    if not overlap:
        raise ValueError(
            "the scores separate the classes (a threshold puts every positive on one side), so "
            "the likelihood has no maximum"
        )
    design = np.column_stack([z, np.ones_like(z)])
    theta = np.array([1.0, 0.0])
    loss = _negative_log_likelihood(design @ theta, y)
    for _ in range(100):
        prob = expit(design @ theta)
        gradient = design.T @ (prob - y)
        if np.max(np.abs(gradient)) <= 1e-10 * len(y):
            break
        hessian = design.T @ (design * (prob * (1.0 - prob))[:, np.newaxis])
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while True:
            candidate = theta - scale * step
            candidate_loss = _negative_log_likelihood(design @ candidate, y)
            if candidate_loss <= loss or scale < 1e-10:
                break
            scale /= 2.0
        theta, loss = candidate, candidate_loss
    else:
        raise RuntimeError("Platt scaling did not converge in 100 Newton steps")
    return PlattScaling(slope=float(theta[0]), intercept=float(theta[1]), logits=logits)
