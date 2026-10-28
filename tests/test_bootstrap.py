"""Bootstrap engine: resampling, BCa formulas and the public interface."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from scipy import stats

from evalstats import bootstrap as bs
from evalstats._validation import FloatArray


def test_counts_rows_sum_to_n_and_have_mean_one() -> None:
    rng = np.random.default_rng(0)
    counts = bs.resample_counts(2000, 50, None, rng)
    assert counts.shape == (2000, 50)
    np.testing.assert_array_equal(counts.sum(axis=1), 50)
    # Each count is Binomial(50, 1/50): mean 1, variance 0.98; 2000 rows give SE 0.022.
    assert np.all(np.abs(counts.mean(axis=0) - 1.0) < 0.12)


def test_stratified_counts_keep_stratum_sizes() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(37) < 0.2).astype(np.float64)
    strata = bs.class_strata(y)
    counts = bs.resample_counts(500, 37, strata, rng)
    np.testing.assert_array_equal(counts[:, y == 1].sum(axis=1), np.sum(y == 1))
    np.testing.assert_array_equal(counts[:, y == 0].sum(axis=1), np.sum(y == 0))


def test_replicates_cross_block_boundaries() -> None:
    # n large enough that the resamples are split into several blocks.
    n = 300_000
    rng = np.random.default_rng(2)
    values = bs.bootstrap_replicates(lambda c: c.sum(axis=1), n, 9, rng)
    np.testing.assert_array_equal(values, n)


def test_jackknife_acceleration_matches_closed_form_for_the_mean() -> None:
    # Leaving out x_i moves the mean by (mean - x_i) / (n - 1), so the acceleration is
    # sum(d^3) / (6 sum(d^2)^1.5) with d = x - mean: the scale factor cancels.
    rng = np.random.default_rng(3)
    x = rng.exponential(size=400)

    def mean_statistic(counts: FloatArray) -> FloatArray:
        return np.asarray((counts @ x) / counts.sum(axis=1), dtype=np.float64)

    jack = bs._jackknife(mean_statistic, len(x), rng)
    d = x - x.mean()
    expected = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)
    assert bs._acceleration(jack) == pytest.approx(expected, rel=1e-9)


def test_grouped_jackknife_acceleration_targets_leave_one_out() -> None:
    rng = np.random.default_rng(4)
    x = rng.exponential(size=3000)

    def mean_statistic(counts: FloatArray) -> FloatArray:
        return np.asarray((counts @ x) / counts.sum(axis=1), dtype=np.float64)

    d = x - x.mean()
    exact = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)
    ratios = [
        bs._acceleration(bs._jackknife(mean_statistic, len(x), rng)) / exact for _ in range(20)
    ]
    # One random grouping (1000 groups of 3) gives a ratio with standard deviation about 0.10 on
    # this data (measured over 200 groupings), so the mean of 20 has SD about 0.022. A tolerance
    # of 0.08 is more than 3.5 SD and still catches a grouped estimate off by a scale factor.
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize(
    ("method", "scipy_method"), [("bca", "BCa"), ("percentile", "percentile")]
)
def test_interval_agrees_with_scipy_bootstrap(
    method: bs.CIMethod, scipy_method: Literal["BCa", "percentile"]
) -> None:
    # The mean absolute error of zero predictions is the mean of |y|, so both libraries
    # bootstrap the same statistic. With 40 000 resamples the Monte Carlo error of each endpoint
    # is well under 1% of the interval width; 3% leaves room without hiding a formula error
    # (a wrong bias correction or acceleration moves the endpoints by 10% or more here).
    rng = np.random.default_rng(5)
    y = rng.exponential(size=80)
    ours = bs.bootstrap_ci(y, np.zeros_like(y), "mae", n_resamples=40_000, method=method, seed=6)
    theirs = stats.bootstrap(
        (y,), np.mean, n_resamples=40_000, method=scipy_method, rng=np.random.default_rng(7)
    ).confidence_interval
    width = theirs.high - theirs.low
    assert abs(ours.ci_low - theirs.low) < 0.03 * width
    assert abs(ours.ci_high - theirs.high) < 0.03 * width


def test_bca_differs_from_percentile_on_skewed_data() -> None:
    rng = np.random.default_rng(8)
    y = rng.exponential(size=60) ** 2
    pct = bs.bootstrap_ci(y, np.zeros_like(y), "mae", method="percentile", seed=1)
    bca = bs.bootstrap_ci(y, np.zeros_like(y), "mae", method="bca", seed=1)
    # Right skew: BCa shifts both endpoints up.
    assert bca.ci_low > pct.ci_low
    assert bca.ci_high > pct.ci_high


def test_same_seed_same_result_and_generator_accepted() -> None:
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 200)
    s = rng.random(200) + 0.3 * y
    first = bs.bootstrap_ci(y, s, "auc", n_resamples=500, seed=42)
    second = bs.bootstrap_ci(y, s, "auc", n_resamples=500, seed=np.random.default_rng(42))
    third = bs.bootstrap_ci(y, s, "auc", n_resamples=500, seed=43)
    assert first == second
    assert first != third
    assert first.ci_low < first.estimate < first.ci_high


def test_constant_metric_gives_degenerate_interval() -> None:
    y = np.array([0, 1, 1, 0, 1] * 10)
    result = bs.bootstrap_ci(y, y, "accuracy", n_resamples=200, seed=0)
    assert (result.ci_low, result.estimate, result.ci_high) == (1.0, 1.0, 1.0)
    assert result.standard_error == 0.0


def test_rare_class_warns_unless_stratified() -> None:
    rng = np.random.default_rng(10)
    y = np.zeros(60)
    y[:2] = 1
    s = rng.random(60) + y
    with pytest.warns(RuntimeWarning, match="undefined auc"):
        plain = bs.bootstrap_ci(y, s, "auc", n_resamples=1000, seed=0, method="percentile")
    assert plain.n_invalid > 0
    stratified = bs.bootstrap_ci(y, s, "auc", n_resamples=1000, seed=0, stratify=True)
    assert stratified.n_invalid == 0
    assert stratified.stratified


def test_perfect_separation_gives_degenerate_auc_interval() -> None:
    y = np.array([0.0] * 20 + [1.0] * 20)
    s = np.concatenate([np.linspace(0, 0.4, 20), np.linspace(0.6, 1, 20)])
    result = bs.bootstrap_ci(y, s, "auc", n_resamples=200, seed=0, stratify=True)
    assert result.ci_low == result.ci_high == 1.0


def test_bca_falls_back_to_percentile_outside_the_distribution() -> None:
    # The number of examples a resample leaves out is 0 on the full sample and positive on
    # (almost) every resample, so the estimate lies below the whole bootstrap distribution.
    def left_out(counts: FloatArray) -> FloatArray:
        return np.asarray((counts == 0).sum(axis=1), dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="outside its bootstrap distribution"):
        summary = bs.summarize(
            left_out,
            30,
            n_resamples=200,
            confidence=0.9,
            method="bca",
            rng=np.random.default_rng(0),
        )
    assert summary.method == "percentile"
    assert summary.estimate == 0.0
    assert summary.ci_low > 0.0


def test_argument_checks() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="at least 100"):
        bs.bootstrap_ci(y, s, "auc", n_resamples=10)
    with pytest.raises(ValueError, match="confidence"):
        bs.bootstrap_ci(y, s, "auc", confidence=1.5)
    with pytest.raises(ValueError, match="method"):
        bs.bootstrap_ci(y, s, "auc", method="basic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="classification metric"):
        bs.bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.5, 2.0], "mae", stratify=True)
    with pytest.raises(ValueError, match="single class"):
        bs.bootstrap_ci([1, 1, 1], [0.1, 0.2, 0.3], "auc")
