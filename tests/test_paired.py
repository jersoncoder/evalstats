"""Paired comparisons: DeLong, McNemar, paired bootstrap and permutation test."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from evalstats import paired
from evalstats._validation import FloatArray
from evalstats.bootstrap import CIMethod, bootstrap_ci
from evalstats.metrics import brier_score, roc_auc


def _correlated_scores(
    rng: np.random.Generator, n: int, shift_b: float = 0.0, decimals: int | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Labels and two score vectors that share a common component (correlated models)."""
    y = (rng.random(n) < 0.4).astype(np.float64)
    shared = rng.normal(size=n)
    a = y + shared + 0.7 * rng.normal(size=n)
    b = (1.0 - shift_b) * y + shared + 0.7 * rng.normal(size=n)
    if decimals is not None:
        a, b = np.round(a, decimals), np.round(b, decimals)
    return y, a, b


# ---------------------------------------------------------------------------------- DeLong


def _brute_force_delong(y: FloatArray, scores: FloatArray) -> tuple[FloatArray, FloatArray]:
    """DeLong's covariance straight from its definition, with the full pair kernel."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n0 = pos.shape[1], neg.shape[1]
    psi = (pos[:, :, None] > neg[:, None, :]) + 0.5 * (pos[:, :, None] == neg[:, None, :])
    v10 = psi.mean(axis=2)  # (k, m): each positive against all negatives
    v01 = psi.mean(axis=1)  # (k, n0): each negative against all positives
    aucs = psi.mean(axis=(1, 2))
    s10 = (v10 - v10.mean(axis=1, keepdims=True)) @ (v10 - v10.mean(axis=1, keepdims=True)).T
    s01 = (v01 - v01.mean(axis=1, keepdims=True)) @ (v01 - v01.mean(axis=1, keepdims=True)).T
    return aucs, s10 / (m - 1) / m + s01 / (n0 - 1) / n0


@pytest.mark.parametrize("decimals", [None, 1])
def test_delong_covariance_matches_brute_force(decimals: int | None) -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        y, a, b = _correlated_scores(rng, int(rng.integers(10, 150)), 0.3, decimals)
        y[:2] = [0.0, 1.0]
        y[2:4] = [0.0, 1.0]
        scores = np.vstack([a, b, -a])
        aucs, cov = paired.delong_covariance(y, scores)
        expected_aucs, expected_cov = _brute_force_delong(y, scores)
        np.testing.assert_allclose(aucs, expected_aucs, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cov, expected_cov, rtol=1e-12, atol=1e-14)
        assert aucs[0] == pytest.approx(roc_auc(y, a), abs=1e-12)


def test_delong_standard_error_agrees_with_paired_bootstrap() -> None:
    # Both estimate the same asymptotic variance. With 4000 resamples the bootstrap standard
    # error has a relative Monte Carlo error near 1/sqrt(2 * 4000) = 1.1%; the two estimators
    # also differ by O(1/n) terms. 5% is a margin of several standard errors.
    rng = np.random.default_rng(1)
    y, a, b = _correlated_scores(rng, 3000, shift_b=0.15)
    delong = paired.delong_test(y, a, b)
    boot = paired.paired_bootstrap(y, a, b, "auc", n_resamples=4000, seed=2)
    delong_se = np.sqrt(float(delong.details["variance"]))
    assert float(boot.details["standard_error"]) == pytest.approx(delong_se, rel=0.05)
    assert boot.difference == pytest.approx(delong.difference, abs=1e-12)
    assert boot.ci_low is not None and delong.ci_low is not None
    assert boot.ci_low == pytest.approx(delong.ci_low, abs=0.15 * delong_se)


def test_delong_symmetry_and_identical_models() -> None:
    rng = np.random.default_rng(3)
    y, a, b = _correlated_scores(rng, 200, 0.4)
    ab, ba = paired.delong_test(y, a, b), paired.delong_test(y, b, a)
    assert ab.difference == pytest.approx(-ba.difference)
    assert ab.p_value == pytest.approx(ba.p_value)
    same = paired.delong_test(y, a, 2 * a + 1)  # same ranking
    assert same.difference == 0.0
    assert same.p_value == 1.0


def test_delong_needs_two_of_each_class() -> None:
    with pytest.raises(ValueError, match="at least two positives"):
        paired.delong_test([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3])


# --------------------------------------------------------------------------------- McNemar


def _labels_with_counts(n10: int, n01: int, n11: int, n00: int) -> tuple[FloatArray, ...]:
    """y, pred_a, pred_b with the given cells (A right/B wrong, A wrong/B right, ...)."""
    y = np.ones(n10 + n01 + n11 + n00)
    a = np.concatenate([np.ones(n10), np.zeros(n01), np.ones(n11), np.zeros(n00)])
    b = np.concatenate([np.zeros(n10), np.ones(n01), np.ones(n11), np.zeros(n00)])
    return y, a, b


def test_mcnemar_exact_matches_scipy_binomtest() -> None:
    for n10, n01 in [(8, 2), (0, 5), (13, 13), (40, 25), (1, 0)]:
        y, a, b = _labels_with_counts(n10, n01, 30, 7)
        result = paired.mcnemar(y, a, b, variant="exact")
        expected = stats.binomtest(n10, n10 + n01, 0.5).pvalue
        assert result.p_value == pytest.approx(expected, rel=1e-12)
        assert result.difference == pytest.approx((n10 - n01) / len(y))


def test_mcnemar_mid_p_hand_computed() -> None:
    # n10 = 8, n01 = 2, ten discordant pairs: P(X <= 2) = (1 + 10 + 45) / 1024.
    y, a, b = _labels_with_counts(8, 2, 20, 5)
    exact = paired.mcnemar(y, a, b, variant="exact")
    mid = paired.mcnemar(y, a, b, variant="mid-p")
    assert exact.p_value == pytest.approx(2 * 56 / 1024)
    assert mid.p_value == pytest.approx(2 * 56 / 1024 - 45 / 1024)
    assert mid.details == {"a_right_b_wrong": 8, "a_wrong_b_right": 2}


def test_mcnemar_agresti_min_interval_by_hand() -> None:
    y, a, b = _labels_with_counts(8, 2, 20, 5)
    result = paired.mcnemar(y, a, b, confidence=0.95)
    n10, n01, n = 8.5, 2.5, 37.0
    centre = (n10 - n01) / n
    se = np.sqrt((n10 + n01) - (n10 - n01) ** 2 / n) / n
    assert result.ci_low == pytest.approx(centre - 1.959963984540054 * se)
    assert result.ci_high == pytest.approx(centre + 1.959963984540054 * se)


def test_mcnemar_without_discordant_pairs() -> None:
    y, a, b = _labels_with_counts(0, 0, 10, 3)
    result = paired.mcnemar(y, a, b)
    assert result.p_value == 1.0
    assert result.difference == 0.0


# ------------------------------------------------------------------------ paired bootstrap


def test_paired_bootstrap_identical_models() -> None:
    rng = np.random.default_rng(4)
    y, a, _ = _correlated_scores(rng, 300)
    result = paired.paired_bootstrap(y, a, a, "auc", n_resamples=500, seed=0)
    assert (result.ci_low, result.difference, result.ci_high) == (0.0, 0.0, 0.0)
    assert result.p_value == 1.0


def test_pairing_narrows_the_interval() -> None:
    # The models share most of their error, so their AUCs are positively correlated (about 0.6
    # here) and the paired difference varies less than two independent estimates would. An
    # unpaired interval for the difference would have width close to hypot(width_a, width_b).
    rng = np.random.default_rng(5)
    y, a, b = _correlated_scores(rng, 800, 0.1)
    diff = paired.paired_bootstrap(y, a, b, "auc", n_resamples=2000, seed=0)
    single_a = bootstrap_ci(y, a, "auc", n_resamples=2000, seed=0)
    single_b = bootstrap_ci(y, b, "auc", n_resamples=2000, seed=0)
    assert diff.ci_low is not None and diff.ci_high is not None
    unpaired_width = np.hypot(
        single_a.ci_high - single_a.ci_low, single_b.ci_high - single_b.ci_low
    )
    assert diff.ci_high - diff.ci_low < 0.75 * unpaired_width


def test_paired_bootstrap_detects_a_large_difference() -> None:
    rng = np.random.default_rng(6)
    y, a, b = _correlated_scores(rng, 1000, 0.6)
    pct = paired.paired_bootstrap(y, a, b, "auc", n_resamples=999, method="percentile", seed=0)
    assert pct.ci_low is not None and pct.ci_low > 0
    assert pct.p_value == pytest.approx(2 / 1000)  # every replicate above zero: p = 2/(B+1)
    bca = paired.paired_bootstrap(y, a, b, "auc", n_resamples=999, seed=0)
    assert bca.ci_low is not None and bca.ci_low > 0
    assert 2 / 1000 <= bca.p_value < 0.01


def test_bootstrap_p_value_is_consistent_with_the_interval() -> None:
    # The p-value inverts the reported interval: it is below 1 - confidence exactly when the
    # interval excludes zero. Check across many small, noisy comparisons and both methods.
    rng = np.random.default_rng(11)
    for trial in range(30):
        y, a, b = _correlated_scores(rng, 120, shift_b=0.3)
        method: CIMethod = "bca" if trial % 2 else "percentile"
        result = paired.paired_bootstrap(y, a, b, "auc", n_resamples=999, method=method, seed=0)
        assert result.ci_low is not None and result.ci_high is not None
        excludes_zero = result.ci_low > 0 or result.ci_high < 0
        assert excludes_zero == (result.p_value < 0.05), (trial, result)


# ----------------------------------------------------------------------- permutation test


def test_permutation_matches_exact_enumeration() -> None:
    # With 12 examples all 4096 swap patterns can be enumerated; the Monte Carlo p-value with
    # 40 000 permutations has standard error below 0.0025, so 0.01 is four standard errors.
    rng = np.random.default_rng(7)
    y = (rng.random(12) < 0.5).astype(np.float64)
    a = np.clip(0.5 * y + 0.5 * rng.random(12), 0, 1)
    b = np.clip(0.2 * y + 0.6 * rng.random(12), 0, 1)
    observed = brier_score(y, a) - brier_score(y, b)
    extreme = 0
    for swaps in itertools.product([False, True], repeat=12):
        mask = np.array(swaps)
        pa, pb = np.where(mask, b, a), np.where(mask, a, b)
        extreme += abs(brier_score(y, pa) - brier_score(y, pb)) >= abs(observed) - 1e-12
    exact_p = extreme / 4096
    result = paired.permutation_test(y, a, b, "brier", n_permutations=40_000, seed=8)
    assert result.p_value == pytest.approx(exact_p, abs=0.01)
    assert result.difference == pytest.approx(observed)


def test_permutation_identical_and_extreme() -> None:
    rng = np.random.default_rng(9)
    y, a, b = _correlated_scores(rng, 400, 0.8)
    same = paired.permutation_test(y, a, a, "auc", n_permutations=200, seed=0)
    assert same.p_value == 1.0
    far = paired.permutation_test(y, a, b, "auc", n_permutations=199, seed=0)
    assert far.p_value == pytest.approx(1 / 200)


def test_permutation_with_weights_and_labels() -> None:
    rng = np.random.default_rng(10)
    y = (rng.random(300) < 0.5).astype(np.float64)
    a = np.where(rng.random(300) < 0.8, y, 1 - y)
    b = np.where(rng.random(300) < 0.7, y, 1 - y)
    w = rng.random(300) + 0.5
    result = paired.permutation_test(y, a, b, "f1", sample_weight=w, n_permutations=500, seed=0)
    assert 0 < result.p_value <= 1
    assert result.ci_low is None
