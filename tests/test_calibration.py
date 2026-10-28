"""Calibration measurement and recalibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from sklearn.calibration import calibration_curve

from evalstats import calibration as cal
from evalstats._validation import FloatArray
from evalstats.metrics import log_loss, roc_auc


def _overconfident_logits(
    rng: np.random.Generator, n: int, temperature: float, bias: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Logits z and labels with P(y = 1 | z) = sigmoid(z / temperature + bias)."""
    z = rng.normal(0.0, 3.0, n)
    y = (rng.random(n) < expit(z / temperature + bias)).astype(np.float64)
    return y, z


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_reliability_curve_matches_sklearn(strategy: cal.BinStrategy) -> None:
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(20, 500))
        prob = np.round(rng.random(n), 2)  # many values fall exactly on uniform edges
        y = (rng.random(n) < prob).astype(np.float64)
        n_bins = int(rng.integers(2, 15))
        curve = cal.reliability_curve(y, prob, n_bins=n_bins, strategy=strategy)
        expected_true, expected_pred = calibration_curve(y, prob, n_bins=n_bins, strategy=strategy)
        np.testing.assert_allclose(curve.observed_frequency, expected_true, rtol=1e-12)
        np.testing.assert_allclose(curve.mean_predicted, expected_pred, rtol=1e-12)
        assert curve.counts.sum() == n


def test_ece_hand_computed() -> None:
    # Two uniform bins: [0, 0.5] holds 0.2, 0.4 (mean 0.3) with one positive of two (0.5);
    # (0.5, 1] holds 0.6, 0.8, 1.0 (mean 0.8) with three positives of three (1.0).
    y = [0, 1, 1, 1, 1]
    prob = [0.2, 0.4, 0.6, 0.8, 1.0]
    expected = (2 / 5) * abs(0.3 - 0.5) + (3 / 5) * abs(0.8 - 1.0)
    assert cal.expected_calibration_error(y, prob, n_bins=2) == pytest.approx(expected)


def test_quantile_bins_hold_equal_counts() -> None:
    rng = np.random.default_rng(1)
    prob = rng.random(1000) ** 3  # skewed: uniform bins would be very unequal
    y = (rng.random(1000) < prob).astype(np.float64)
    curve = cal.reliability_curve(y, prob, n_bins=10, strategy="quantile")
    np.testing.assert_array_equal(curve.counts, 100)


def test_ece_is_small_for_calibrated_and_large_for_overconfident() -> None:
    rng = np.random.default_rng(2)
    prob = rng.random(20_000)
    y = (rng.random(20_000) < prob).astype(np.float64)
    assert cal.expected_calibration_error(y, prob) < 0.02
    overconfident = expit(3.0 * cal.logit(prob))
    assert cal.expected_calibration_error(y, overconfident) > 0.1


def test_temperature_recovers_the_true_value() -> None:
    # The MLE of u = 1/T has standard error 1/sqrt(I), I = sum sigma(1 - sigma) z^2 (Fisher
    # information), so the fitted T must lie within 4 standard errors of the truth.
    rng = np.random.default_rng(3)
    y, z = _overconfident_logits(rng, 20_000, temperature=2.5)
    fitted = cal.fit_temperature(y, z, logits=True)
    u = 1.0 / 2.5
    s = expit(u * z)
    se_u = 1.0 / np.sqrt(np.sum(s * (1 - s) * z**2))
    se_t = se_u / u**2
    assert abs(fitted.temperature - 2.5) < 4 * se_t
    # First-order condition at the returned value.
    u_hat = 1.0 / fitted.temperature
    assert abs(np.sum((expit(u_hat * z) - y) * z)) < 1e-6


def test_platt_recovers_the_true_values_and_satisfies_first_order_conditions() -> None:
    rng = np.random.default_rng(4)
    y, z = _overconfident_logits(rng, 20_000, temperature=2.0, bias=-1.0)
    fitted = cal.fit_platt(y, z, logits=True)
    # Standard errors from the inverse Fisher information at the truth.
    design = np.column_stack([z, np.ones_like(z)])
    s = expit(design @ np.array([0.5, -1.0]))
    cov = np.linalg.inv(design.T @ (design * (s * (1 - s))[:, None]))
    assert abs(fitted.slope - 0.5) < 4 * np.sqrt(cov[0, 0])
    assert abs(fitted.intercept + 1.0) < 4 * np.sqrt(cov[1, 1])
    gradient = design.T @ (expit(fitted.slope * z + fitted.intercept) - y)
    assert np.max(np.abs(gradient)) < 1e-6 * len(y)


def test_recalibration_fitted_on_validation_improves_test_calibration() -> None:
    rng = np.random.default_rng(5)
    y, z = _overconfident_logits(rng, 8000, temperature=3.0, bias=0.4)
    prob = np.asarray(expit(z), dtype=np.float64)  # sigmoid(z): overconfident and biased
    val, test = slice(0, 4000), slice(4000, None)
    temperature = cal.fit_temperature(y[val], prob[val])
    platt = cal.fit_platt(y[val], prob[val])
    before = cal.expected_calibration_error(y[test], prob[test])
    after_t = cal.expected_calibration_error(y[test], temperature.transform(prob[test]))
    after_p = cal.expected_calibration_error(y[test], platt.transform(prob[test]))
    assert after_t < 0.5 * before
    assert after_p < after_t  # only Platt can correct the bias term
    assert log_loss(y[test], platt.transform(prob[test])) < log_loss(y[test], prob[test])
    # Both maps are increasing, so the ranking (and the AUC) is unchanged.
    assert roc_auc(y[test], temperature.transform(prob[test])) == pytest.approx(
        roc_auc(y[test], prob[test]), abs=1e-12
    )


def test_fits_refuse_when_no_maximum_exists() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    separated = np.array([0.1, 0.3, 0.7, 0.9])
    with pytest.raises(ValueError, match="no maximum"):
        cal.fit_temperature(y, separated)
    with pytest.raises(ValueError, match="separate the classes"):
        cal.fit_platt(y, separated)
    with pytest.raises(ValueError, match="separate the classes"):
        cal.fit_platt(y, np.array([0.1, 0.5, 0.5, 0.9]))  # tie at the boundary
    with pytest.raises(ValueError, match="not positively associated"):
        cal.fit_temperature(y, np.array([0.9, 0.6, 0.4, 0.45]))
    with pytest.raises(ValueError, match="both classes"):
        cal.fit_platt([1, 1], [0.2, 0.7])


def test_input_checks() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        cal.expected_calibration_error([0, 1], [0.2, 1.2])
    with pytest.raises(ValueError, match="strategy"):
        cal.reliability_curve([0, 1], [0.2, 0.8], strategy="kmeans")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_bins"):
        cal.reliability_curve([0, 1], [0.2, 0.8], n_bins=0)
