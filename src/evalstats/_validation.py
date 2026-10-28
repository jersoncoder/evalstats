"""Input checks shared by the public functions.

Every public entry point converts its inputs here, so the numerical code can assume
1-D float64 arrays of equal length with no NaN or infinite values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]

SeedLike = int | np.random.Generator | None


def as_float_vector(values: ArrayLike, name: str) -> FloatArray:
    """Return ``values`` as a finite 1-D float64 array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if array.dtype == np.bool_:
        array = array.astype(np.float64)
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def check_binary(values: FloatArray, name: str) -> None:
    """Raise unless every value is 0 or 1."""
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError(f"{name} must contain only 0 and 1")


def check_probabilities(values: FloatArray, name: str) -> None:
    """Raise unless every value lies in [0, 1]."""
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")


def check_same_length(**arrays: FloatArray) -> int:
    """Return the common length of the arrays, or raise naming the mismatch."""
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"inputs must have the same length, got {detail}")
    return next(iter(lengths.values()))


def as_weights(sample_weight: ArrayLike | None, n: int) -> FloatArray:
    """Return validated non-negative weights, or ones when ``sample_weight`` is None."""
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    weights = as_float_vector(sample_weight, "sample_weight")
    if len(weights) != n:
        raise ValueError(f"sample_weight has length {len(weights)}, expected {n}")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must have at least one positive entry")
    return weights


def check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def check_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def as_generator(seed: SeedLike) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from an int seed, a Generator or None."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
