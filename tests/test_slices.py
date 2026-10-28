"""Slice analysis."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from evalstats import slices
from evalstats._validation import FloatArray
from evalstats.metrics import accuracy, roc_auc
from evalstats.multiple import Adjustment, benjamini_hochberg, holm


def _sliced_predictions(
    rng: np.random.Generator, accuracies: dict[str, float], per_slice: int
) -> tuple[FloatArray, FloatArray, list[str]]:
    y_parts, pred_parts, groups = [], [], []
    for name, acc in accuracies.items():
        y = (rng.random(per_slice) < 0.5).astype(np.float64)
        pred = np.where(rng.random(per_slice) < acc, y, 1.0 - y)
        y_parts.append(y)
        pred_parts.append(pred)
        groups += [name] * per_slice
    return np.concatenate(y_parts), np.concatenate(pred_parts), groups


def test_flags_the_weak_slice() -> None:
    # Ten slices; "g3" is much worse (0.60 against 0.85). Its rest is the other nine slices.
    # The weak slice also lowers the rest of every other slice by about 3 points, so a strong
    # slice can be flagged too; this is inherent to slice-versus-rest comparisons and is why
    # the test only requires the weak slice to be flagged with the smallest p-value.
    rng = np.random.default_rng(0)
    accuracies = {f"g{i}": 0.85 for i in range(10)}
    accuracies["g3"] = 0.6
    y, pred, groups = _sliced_predictions(rng, accuracies, 300)
    report = slices.slice_analysis(y, pred, groups, "accuracy", n_resamples=2000, seed=1)
    weak = next(s for s in report.slices if s.name == "g3")
    assert weak in report.flagged()
    assert weak.p_adjusted == min(s.p_adjusted for s in report.slices)
    assert weak.difference < 0
    assert weak.difference_ci_high < 0
    assert weak.ci_low < 0.6 < weak.ci_high


def test_estimates_equal_metric_on_subsets_and_adjustment_is_applied() -> None:
    rng = np.random.default_rng(2)
    y, pred, groups = _sliced_predictions(rng, {"a": 0.9, "b": 0.8, "c": 0.7}, 200)
    cases: list[tuple[Adjustment, Callable[[list[float]], FloatArray]]] = [
        ("holm", holm),
        ("bh", benjamini_hochberg),
    ]
    for adjust, fn in cases:
        report = slices.slice_analysis(
            y, pred, groups, "accuracy", n_resamples=500, adjust=adjust, seed=0
        )
        labels = np.array(groups)
        for result in report.slices:
            inside = labels == result.name
            assert result.estimate == pytest.approx(accuracy(y[inside], pred[inside]))
            assert result.rest_estimate == pytest.approx(accuracy(y[~inside], pred[~inside]))
            assert result.difference == pytest.approx(result.estimate - result.rest_estimate)
        raw = [r.p_value for r in report.slices]
        np.testing.assert_allclose([r.p_adjusted for r in report.slices], fn(raw))
        assert report.overall == pytest.approx(accuracy(y, pred))


def test_small_and_undefined_slices_are_skipped_with_reasons() -> None:
    rng = np.random.default_rng(3)
    n = 400
    y = (rng.random(n) < 0.5).astype(np.float64)
    scores = rng.random(n) + 0.5 * y
    groups = np.array(["big"] * 200 + ["other"] * 180 + ["tiny"] * 20, dtype=object)
    y[200:260] = 1.0  # "one" has a single class: AUC undefined there
    groups[200:260] = "one"
    report = slices.slice_analysis(y, scores, groups, "auc", min_support=30, n_resamples=300)
    skipped = {s.name: s.reason for s in report.skipped}
    assert skipped["tiny"] == "fewer than 30 examples"
    assert "undefined on the slice" in skipped["one"]
    assert [s.name for s in report.slices] == ["big", "other"]
    # Skipped slices still belong to the rest of the others.
    big = report.slices[0]
    inside = groups == "big"
    assert big.rest_estimate == pytest.approx(roc_auc(y[~inside], scores[~inside]))


def test_stratified_slices_avoid_undefined_resamples() -> None:
    rng = np.random.default_rng(4)
    y = np.zeros(600)
    y[[0, 1, 2, 200, 201, 202, 400, 401, 402]] = 1.0  # three positives per slice
    scores = rng.random(600) + y
    groups = np.repeat(["a", "b", "c"], 200)
    with pytest.warns(RuntimeWarning, match="undefined auc"):
        slices.slice_analysis(y, scores, groups, "auc", n_resamples=500, seed=0)
    report = slices.slice_analysis(
        y, scores, groups, "auc", n_resamples=500, seed=0, stratify=True
    )
    assert len(report.slices) == 3


def test_integer_and_mixed_group_labels() -> None:
    rng = np.random.default_rng(5)
    y, pred, _ = _sliced_predictions(rng, {"x": 0.8, "y": 0.8}, 100)
    report = slices.slice_analysis(y, pred, [1] * 100 + [2] * 100, "accuracy", n_resamples=200)
    assert [s.name for s in report.slices] == ["1", "2"]
    with pytest.raises(ValueError, match="length 200"):
        slices.slice_analysis(y, pred, [1, 2], "accuracy")
    with pytest.raises(ValueError, match="'holm' or 'bh'"):
        slices.slice_analysis(y, pred, [1] * 200, "accuracy", adjust="none")  # type: ignore[arg-type]


@pytest.mark.slow
def test_slice_family_wise_error_under_the_null() -> None:
    # Five identical slices: the chance that Holm flags any of them must stay near or below 5%.
    # The bootstrap p-values are approximate, so this is a check of calibration rather than a
    # guarantee. 400 repetitions: SE 0.011, tolerance 3 SE.
    repetitions, alpha = 400, 0.05
    rng = np.random.default_rng(6)
    any_flagged = 0
    for _ in range(repetitions):
        y, pred, groups = _sliced_predictions(rng, {f"s{i}": 0.8 for i in range(5)}, 200)
        report = slices.slice_analysis(y, pred, groups, "accuracy", n_resamples=1000, seed=rng)
        any_flagged += bool(report.flagged(alpha))
    rate = any_flagged / repetitions
    assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / repetitions), rate
