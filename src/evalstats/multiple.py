"""Adjusted p-values for multiple comparisons.

* Holm (1979) controls the family-wise error rate (the chance of any false rejection) under any
  dependence between the tests. It rejects every hypothesis Bonferroni rejects, and often more.
* Benjamini-Hochberg (1995) controls the false discovery rate (the expected share of false
  rejections among all rejections) for independent or positively dependent tests.

Rejecting every hypothesis whose adjusted p-value is at most ``alpha`` is the same as running
the corresponding step-down (Holm) or step-up (BH) procedure at level ``alpha``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from evalstats._validation import FloatArray

Adjustment = Literal["holm", "bh"]


def _as_p_values(p_values: ArrayLike) -> FloatArray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {p.shape}")
    if np.any(np.isnan(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_values must lie in [0, 1] and contain no NaN")
    return p


def holm(p_values: ArrayLike) -> FloatArray:
    """Holm step-down adjusted p-values, in the input order.

    With sorted p-values p(1) <= ... <= p(m), the adjusted value of p(i) is
    max over j <= i of min(1, (m - j + 1) p(j)).
    """
    p = _as_p_values(p_values)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    scaled = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(scaled)
    return adjusted


def benjamini_hochberg(p_values: ArrayLike) -> FloatArray:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), in the input order.

    With sorted p-values p(1) <= ... <= p(m), the adjusted value of p(i) is
    min over j >= i of min(1, m p(j) / j).
    """
    p = _as_p_values(p_values)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    # Multiply by m / rank (not m * p / rank) so the largest p-value is kept exactly.
    scaled = np.minimum(1.0, p[order] * (m / np.arange(1, m + 1)))
    adjusted = np.empty(m)
    adjusted[order] = np.minimum.accumulate(scaled[::-1])[::-1]
    return adjusted


def adjust_p_values(p_values: ArrayLike, method: Adjustment = "holm") -> FloatArray:
    """Adjusted p-values by ``"holm"`` or ``"bh"`` (Benjamini-Hochberg)."""
    if method == "holm":
        return holm(p_values)
    if method == "bh":
        return benjamini_hochberg(p_values)
    raise ValueError(f"method must be 'holm' or 'bh', got {method!r}")
