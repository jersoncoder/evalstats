"""Command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evalstats import __version__
from evalstats._validation import FloatArray
from evalstats.cli import main
from evalstats.metrics import accuracy, log_loss, roc_auc


def _write(path: Path, header: Sequence[str], *columns: Iterable[object]) -> Path:
    lines = [",".join(header)] + [
        ",".join(str(v) for v in row) for row in zip(*columns, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.4).astype(int)
    shared = rng.normal(size=n)
    a = 1 / (1 + np.exp(-(1.5 * y + shared + 0.5 * rng.normal(size=n) - 0.6)))
    b = 1 / (1 + np.exp(-(0.9 * y + shared + 0.5 * rng.normal(size=n) - 0.6)))
    return {
        "labels": _write(tmp_path / "y.csv", ["label"], y),
        "a": _write(tmp_path / "a.csv", ["score"], np.round(a, 6)),
        "b": _write(tmp_path / "b.csv", ["score"], np.round(b, 6)),
    }


def _args(files: dict[str, Path], metric: str, *extra: str) -> list[str]:
    return [
        "compare",
        "--labels",
        str(files["labels"]),
        "--a",
        str(files["a"]),
        "--b",
        str(files["b"]),
        "--metric",
        metric,
        "--n-resamples",
        "500",
        *extra,
    ]


def _values(path: Path) -> FloatArray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _json_report(
    files: dict[str, Path], metric: str, capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
    assert main(_args(files, metric, "--json")) == 0

    def refuse(constant: str) -> None:
        raise AssertionError(f"invalid JSON constant {constant}")

    report: dict[str, Any] = json.loads(capsys.readouterr().out, parse_constant=refuse)
    return report


def test_text_report(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args(files, "auc")) == 0
    out = capsys.readouterr().out
    y, a, b = (_values(files[k]) for k in ("labels", "a", "b"))
    assert out.startswith(f"evalstats compare: auc on 300 examples ({int(y.sum())} positive)")
    assert f"{roc_auc(y, a):.4f}" in out
    assert f"{roc_auc(y, b):.4f}" in out
    for name in ("paired bootstrap", "DeLong", "permutation"):
        assert name in out
    assert "A is better on auc" in out


def test_json_report_matches_the_library(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    y, a, b = (_values(files[k]) for k in ("labels", "a", "b"))
    report = _json_report(files, "auc", capsys)
    assert report["models"]["a"]["estimate"] == pytest.approx(roc_auc(y, a))
    assert report["difference"] == pytest.approx(roc_auc(y, a) - roc_auc(y, b))
    assert [t["method"] for t in report["tests"]] == ["paired_bootstrap", "delong", "permutation"]

    report = _json_report(files, "accuracy", capsys)
    assert report["models"]["b"]["estimate"] == pytest.approx(accuracy(y, b >= 0.5))
    assert [t["method"] for t in report["tests"]] == [
        "paired_bootstrap",
        "mcnemar_mid_p",
        "permutation",
    ]


def test_same_seed_same_report(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert _json_report(files, "brier", capsys) == _json_report(files, "brier", capsys)


def test_lower_is_better_verdict(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    y, a, b = (_values(files[k]) for k in ("labels", "a", "b"))
    assert log_loss(y, a) < log_loss(y, b)
    assert main(_args(files, "log_loss")) == 0
    out = capsys.readouterr().out
    assert "lower is better" in out
    assert "A is better on log_loss" in out


def test_rows_are_aligned_by_id(
    tmp_path: Path, files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    y, a, b = (_values(files[k]) for k in ("labels", "a", "b"))
    ids = [f"ex{i}" for i in range(len(y))]
    order = np.random.default_rng(1).permutation(len(y))
    shuffled = {
        "labels": _write(tmp_path / "yi.csv", ["id", "label"], ids, y.astype(int)),
        "a": _write(tmp_path / "ai.csv", ["score", "id"], a[order], [ids[i] for i in order]),
        "b": _write(tmp_path / "bi.csv", ["id", "score"], [ids[i] for i in order], b[order]),
    }
    by_position = _json_report(files, "auc", capsys)
    by_id = _json_report(shuffled, "auc", capsys)
    assert by_id["models"] == by_position["models"]
    assert by_id["tests"] == by_position["tests"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("label\n1\n0\n", "different numbers of rows"),
        ("x,y,z\n1,2,3\n", "expected one column"),
        ("label\n1\nyes\n", "'yes' is not a number"),
        ("label\n", "at least one data row"),
        ("label\n1\n2\n" + "0\n" * 298, "only 0 and 1"),
        ("id,label\na,1\n", "every file has an 'id' column or none"),
    ],
)
def test_input_errors(
    files: dict[str, Path], content: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    files["labels"].write_text(content, encoding="utf-8")
    assert main(_args(files, "auc")) == 2
    assert message in capsys.readouterr().err


def test_id_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = _write(tmp_path / "y.csv", ["id", "label"], ["a", "b", "c"], [0, 1, 1])
    dup = _write(tmp_path / "a.csv", ["id", "score"], ["a", "a", "c"], [0.1, 0.2, 0.3])
    other = _write(tmp_path / "b.csv", ["id", "score"], ["a", "b", "d"], [0.1, 0.2, 0.3])
    argv = ["compare", "--labels", str(labels), "--metric", "auc"]
    assert main([*argv, "--a", str(dup), "--b", str(other)]) == 2
    assert "duplicate ids" in capsys.readouterr().err
    assert main([*argv, "--a", str(other), "--b", str(other)]) == 2
    assert "ids do not match" in capsys.readouterr().err


def test_missing_file(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    files["a"] = files["a"].with_name("missing.csv")
    assert main(_args(files, "auc")) == 2
    assert "cannot read" in capsys.readouterr().err


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "evalstats", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"evalstats {__version__}"
