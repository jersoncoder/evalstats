"""Holm and Benjamini-Hochberg adjustments."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from evalstats.multiple import adjust_p_values, benjamini_hochberg, holm


def test_holm_hand_computed() -> None:
    # Sorted: 0.005 * 4 = 0.02, 0.01 * 3 = 0.03, 0.03 * 2 = 0.06, 0.04 * 1 -> max(0.06, 0.04).
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_bh_hand_computed() -> None:
    # Sorted: 0.005 * 4/1 = 0.02, 0.01 * 4/2 = 0.02, 0.03 * 4/3 = 0.04, 0.04 * 4/4 = 0.04.
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.02])


def test_bh_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(int(rng.integers(1, 60))) ** 3
        expected = stats.false_discovery_control(p, method="bh")
        np.testing.assert_allclose(benjamini_hochberg(p), expected, rtol=1e-12)


def test_adjusted_values_reproduce_the_rejection_rules() -> None:
    rng = np.random.default_rng(1)
    alpha = 0.05
    for _ in range(200):
        m = int(rng.integers(1, 30))
        p = np.round(rng.random(m) ** 4, 3)  # rounding creates ties
        order = np.sort(p)
        # Holm: reject the smallest k with p(i) <= alpha / (m - i + 1) for every i <= k.
        holm_k = 0
        while holm_k < m and order[holm_k] <= alpha / (m - holm_k):
            holm_k += 1
        holm_threshold = order[holm_k - 1] if holm_k else -1.0
        np.testing.assert_array_equal(holm(p) <= alpha, p <= holm_threshold)
        # BH: reject the smallest k, where k is the largest i with p(i) <= i alpha / m.
        passing = np.nonzero(order <= np.arange(1, m + 1) * alpha / m)[0]
        bh_threshold = order[passing[-1]] if len(passing) else -1.0
        np.testing.assert_array_equal(benjamini_hochberg(p) <= alpha, p <= bh_threshold)


def test_properties() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(25)
    for adjusted in (holm(p), benjamini_hochberg(p)):
        assert np.all(adjusted >= p)
        assert np.all(adjusted <= 1.0)
        order = np.argsort(p)
        assert np.all(np.diff(adjusted[order]) >= 0)  # monotone in the raw p-values
    assert np.all(benjamini_hochberg(p) <= holm(p))
    np.testing.assert_array_equal(adjust_p_values(p, "bh"), benjamini_hochberg(p))
    np.testing.assert_array_equal(holm([0.2]), [0.2])
    assert holm([]).shape == (0,)


def test_validation() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm([0.1, 1.2])
    with pytest.raises(ValueError, match="NaN"):
        benjamini_hochberg([0.1, np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        holm([[0.1, 0.2]])
    with pytest.raises(ValueError, match="'holm' or 'bh'"):
        adjust_p_values([0.1], "bonferroni")  # type: ignore[arg-type]
