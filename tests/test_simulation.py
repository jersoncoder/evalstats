"""Simulation checks of the statistical guarantees, with fixed seeds.

Each test draws many independent datasets from a model where the truth is known and counts how
often a procedure does the right thing. A count over R repetitions is binomial, so a rate whose
true value is q has standard error sqrt(q (1 - q) / R); the tolerances below are stated in those
units. The seeds are fixed, so the tests are deterministic. A correct implementation passes a
3-SE check with probability about 99.7% for an arbitrary seed, and an implementation with a real
defect (an interval 20% too short, a test with size 0.08 instead of 0.05) fails it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import binom, norm

from evalstats import paired
from evalstats._validation import FloatArray
from evalstats.bootstrap import CIMethod, bootstrap_ci
from evalstats.multiple import benjamini_hochberg, holm


def _binomial_se(q: float, repetitions: int) -> float:
    return float(np.sqrt(q * (1.0 - q) / repetitions))


# ------------------------------------------------------------------ bootstrap CI coverage

Generator = Callable[[np.random.Generator], tuple[FloatArray, FloatArray]]
N = 150
SHIFT = 1.0


def _binormal_scores(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    # Negatives ~ N(0, 1), positives ~ N(SHIFT, 1): AUC = Phi(SHIFT / sqrt(2)).
    y = (np.arange(N) % 2).astype(np.float64)
    return y, rng.normal(size=N) + SHIFT * y


def _labels_80_percent_correct(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    y = rng.integers(0, 2, N).astype(np.float64)
    return y, np.where(rng.random(N) < 0.8, y, 1.0 - y)


def _gaussian_errors(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    # Errors ~ N(0, 1): mean absolute error = sqrt(2 / pi).
    y = rng.normal(size=N)
    return y, y + rng.normal(size=N)


COVERAGE_CASES = [
    ("auc", "bca", True, _binormal_scores, float(ndtr(SHIFT / np.sqrt(2.0)))),
    ("auc", "percentile", True, _binormal_scores, float(ndtr(SHIFT / np.sqrt(2.0)))),
    ("accuracy", "percentile", False, _labels_80_percent_correct, 0.8),
    ("mae", "bca", False, _gaussian_errors, float(np.sqrt(2.0 / np.pi))),
]


@pytest.mark.parametrize(
    ("metric", "method", "stratify", "generate", "truth"),
    COVERAGE_CASES,
    ids=[f"{case[0]}-{case[1]}" for case in COVERAGE_CASES],
)
def test_bootstrap_interval_coverage(
    metric: str, method: CIMethod, stratify: bool, generate: Generator, truth: float
) -> None:
    # 400 repetitions: SE of the coverage at 0.95 is 0.011, and the tolerance 0.035 is 3.2 SE.
    # An interval 20% too short would cover about 88% and fail. Bootstrap intervals at n = 150
    # undercover by about one point in larger runs (scripts/simulation_study.py), which the
    # tolerance absorbs.
    repetitions = 400
    rng = np.random.default_rng(2026)
    covered = 0
    for _ in range(repetitions):
        y, pred = generate(rng)
        result = bootstrap_ci(
            y, pred, metric, n_resamples=1000, method=method, stratify=stratify, seed=rng
        )
        covered += result.ci_low <= truth <= result.ci_high
    coverage = covered / repetitions
    assert abs(coverage - 0.95) <= 0.035, coverage


# ----------------------------------------------------------------- type-I error of tests


def _mcnemar_null_size(n: int, discordance: float, variant: str, alpha: float) -> float:
    """Exact probability that McNemar rejects at ``alpha`` under the null, by enumeration."""
    size = 0.0
    for discordant in range(1, n + 1):
        k = np.arange(discordant + 1)
        smaller = np.minimum(k, discordant - k)
        tail = binom.cdf(smaller, discordant, 0.5)
        if variant == "mid-p":
            tail = tail - 0.5 * binom.pmf(smaller, discordant, 0.5)
        rejects = np.minimum(1.0, 2.0 * tail) <= alpha
        size += binom.pmf(discordant, n, discordance) * np.sum(
            binom.pmf(k, discordant, 0.5) * rejects
        )
    return float(size)


def test_mcnemar_holds_its_size_under_the_null() -> None:
    # Each example is discordant with probability 0.2, either way with probability 0.1, so
    # the two models have the same accuracy. The exact size of each variant is computed by
    # enumeration; the simulated rejection rate must match it within 3 SE and stay <= alpha.
    n, repetitions, alpha = 50, 4000, 0.05
    rng = np.random.default_rng(1)
    rejections = {"exact": 0, "mid-p": 0}
    y = np.ones(n)
    for _ in range(repetitions):
        u = rng.random(n)
        a = np.where(u < 0.1, 1.0, np.where(u < 0.2, 0.0, 1.0))
        b = np.where(u < 0.1, 0.0, np.where(u < 0.2, 1.0, 1.0))
        for variant in rejections:
            result = paired.mcnemar(y, a, b, variant=variant)  # type: ignore[arg-type]
            rejections[variant] += result.p_value <= alpha
    for variant, count in rejections.items():
        size = _mcnemar_null_size(n, 0.2, variant, alpha)
        assert size <= alpha
        rate = count / repetitions
        assert abs(rate - size) <= 3 * _binomial_se(size, repetitions), (variant, rate, size)
    # Mid-p is closer to the nominal level than the conservative exact test.
    assert rejections["exact"] < rejections["mid-p"]


def test_mcnemar_interval_coverage() -> None:
    # True accuracy difference 0.12 - 0.06 = 0.06. 2000 repetitions: SE 0.0049. The
    # Agresti-Min interval is discrete, so its coverage oscillates around the nominal level by
    # about a point as n and the cell probabilities change; tolerance 0.025 = 0.01 + 3 SE.
    n, repetitions = 100, 2000
    rng = np.random.default_rng(2)
    covered = 0
    y = np.ones(n)
    for _ in range(repetitions):
        u = rng.random(n)
        a = np.where(u < 0.12, 1.0, np.where(u < 0.18, 0.0, 1.0))
        b = np.where(u < 0.12, 0.0, np.where(u < 0.18, 1.0, 1.0))
        result = paired.mcnemar(y, a, b)
        assert result.ci_low is not None and result.ci_high is not None
        covered += result.ci_low <= 0.06 <= result.ci_high
    assert abs(covered / repetitions - 0.95) <= 0.025


@pytest.mark.parametrize("metric", ["brier", "accuracy"])
def test_permutation_test_holds_its_size_under_the_null(metric: str) -> None:
    # The two models' predictions for each example are drawn from the same distribution, so
    # they are exchangeable. With B = 199 permutations and a continuous statistic the size at
    # 0.05 is exactly 10/200 = 0.05; a discrete statistic (accuracy) ties more and rejects
    # less. 2000 repetitions: SE 0.0049, 3 SE = 0.015.
    n, repetitions, alpha = 40, 2000, 0.05
    rng = np.random.default_rng(3)
    rejections = 0
    for _ in range(repetitions):
        y = (rng.random(n) < 0.5).astype(np.float64)
        signal = 0.3 * y + 0.35
        a = np.clip(signal + 0.25 * rng.normal(size=n), 0.0, 1.0)
        b = np.clip(signal + 0.25 * rng.normal(size=n), 0.0, 1.0)
        if metric == "accuracy":
            a, b = (a >= 0.5).astype(np.float64), (b >= 0.5).astype(np.float64)
        result = paired.permutation_test(y, a, b, metric, n_permutations=199, seed=rng)
        rejections += result.p_value <= alpha
    rate = rejections / repetitions
    tolerance = 3 * _binomial_se(alpha, repetitions)
    assert rate <= alpha + tolerance, rate
    if metric == "brier":
        assert rate >= alpha - tolerance, rate


def test_delong_holds_its_size_under_the_null() -> None:
    # Two models with the same score distribution and correlated noise. DeLong's test is
    # asymptotic; at 100 positives and 100 negatives its size is close to nominal.
    # 1500 repetitions: SE 0.0056, 3 SE = 0.017.
    n, repetitions, alpha = 200, 1500, 0.05
    rng = np.random.default_rng(4)
    rejections = 0
    y = (np.arange(n) % 2).astype(np.float64)
    for _ in range(repetitions):
        shared = rng.normal(size=n)
        a = y + shared + rng.normal(size=n)
        b = y + shared + rng.normal(size=n)
        rejections += paired.delong_test(y, a, b).p_value <= alpha
    rate = rejections / repetitions
    assert abs(rate - alpha) <= 3 * _binomial_se(alpha, repetitions), rate


# ------------------------------------------------------------ multiple-comparison control


def test_benjamini_hochberg_controls_fdr_and_holm_controls_fwer() -> None:
    # 20 independent tests, 16 true nulls with uniform p-values and 4 alternatives with
    # z ~ N(3, 1). For independent tests BH's false discovery rate is exactly
    # q * m0 / m = 0.05 * 16 / 20 = 0.04 (Benjamini and Hochberg, 1995). The false discovery
    # proportion is bounded, so its mean over 4000 repetitions has a standard error we can
    # estimate from the sample; we require the mean within 3 SE of 0.04.
    m, m0, q, repetitions = 20, 16, 0.05, 4000
    rng = np.random.default_rng(5)
    is_null = np.arange(m) < m0
    false_discovery_proportions = np.empty(repetitions)
    any_false_rejection_holm = 0
    for r in range(repetitions):
        p = np.where(is_null, rng.random(m), norm.sf(rng.normal(3.0, 1.0, m)))
        rejected = benjamini_hochberg(p) <= q
        n_rejected = int(np.sum(rejected))
        false_discovery_proportions[r] = int(np.sum(rejected & is_null)) / max(1, n_rejected)
        any_false_rejection_holm += int(np.any((holm(p) <= q) & is_null))
    fdr = false_discovery_proportions.mean()
    se = false_discovery_proportions.std(ddof=1) / np.sqrt(repetitions)
    assert abs(fdr - q * m0 / m) <= 3 * se, (fdr, se)
    fwer = any_false_rejection_holm / repetitions
    assert fwer <= q + 3 * _binomial_se(q, repetitions), fwer
