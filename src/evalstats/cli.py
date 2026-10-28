"""Command-line interface: ``evalstats compare``.

Input files are CSV with a header row and either a single column of values (rows matched by
position) or an ``id`` column and one value column (rows matched by id, in any order).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from evalstats import __version__
from evalstats._validation import FloatArray
from evalstats.bootstrap import BootstrapResult, bootstrap_ci
from evalstats.metrics import METRICS, Metric, get_metric
from evalstats.paired import (
    PairedResult,
    delong_test,
    mcnemar,
    paired_bootstrap,
    permutation_test,
)

_TEST_NAMES = {
    "paired_bootstrap": "paired bootstrap",
    "delong": "DeLong",
    "mcnemar_mid_p": "McNemar (mid-p)",
    "permutation": "permutation",
}


class InputError(Exception):
    """A problem with the input files, reported without a traceback."""


def read_column(path: Path) -> tuple[list[str] | None, FloatArray]:
    """Read one value column, and the ids if the file has an ``id`` column."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    except OSError as error:
        raise InputError(f"cannot read {path}: {error.strerror}") from error
    if len(rows) < 2:
        raise InputError(f"{path}: expected a header row and at least one data row")
    header = [cell.strip() for cell in rows[0]]
    if len(header) == 1:
        id_column, value_column = None, 0
    elif len(header) == 2 and "id" in header:
        id_column = header.index("id")
        value_column = 1 - id_column
    else:
        raise InputError(
            f"{path}: expected one column, or an 'id' column and one value column; found {header}"
        )
    values = np.empty(len(rows) - 1)
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise InputError(f"{path}, line {line}: expected {len(header)} fields")
        try:
            values[line - 2] = float(row[value_column])
        except ValueError:
            raise InputError(
                f"{path}, line {line}: {row[value_column]!r} is not a number"
            ) from None
    if not np.all(np.isfinite(values)):
        raise InputError(f"{path}: values must be finite numbers")
    if id_column is None:
        return None, values
    ids = [row[id_column].strip() for row in rows[1:]]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: duplicate ids")
    return ids, values


def load_aligned(labels: Path, a: Path, b: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Read the three files and return their values in a common row order."""
    columns = [read_column(path) for path in (labels, a, b)]
    with_ids = [ids is not None for ids, _ in columns]
    if all(with_ids):
        reference = columns[0][0]
        assert reference is not None
        aligned = []
        for path, (ids, values) in zip((labels, a, b), columns, strict=True):
            assert ids is not None
            if set(ids) != set(reference):
                raise InputError(f"{path}: ids do not match those of {labels}")
            position = {key: i for i, key in enumerate(ids)}
            aligned.append(values[[position[key] for key in reference]])
        return aligned[0], aligned[1], aligned[2]
    if any(with_ids):
        raise InputError("either every file has an 'id' column or none does")
    lengths = {
        str(path): len(values) for path, (_, values) in zip((labels, a, b), columns, strict=True)
    }
    if len(set(lengths.values())) != 1:
        raise InputError(f"files have different numbers of rows: {lengths}")
    return columns[0][1], columns[1][1], columns[2][1]


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    return value


def compare(
    y: FloatArray,
    a: FloatArray,
    b: FloatArray,
    spec: Metric,
    *,
    threshold: float,
    n_resamples: int,
    confidence: float,
    method: str,
    stratify: bool,
    seed: int,
) -> dict[str, Any]:
    """Run every applicable comparison and return a JSON-ready report."""
    if spec.kind == "label":
        a = (a >= threshold).astype(np.float64)
        b = (b >= threshold).astype(np.float64)
    rng = np.random.default_rng(seed)
    options: dict[str, Any] = {
        "n_resamples": n_resamples,
        "confidence": confidence,
        "method": method,
        "stratify": stratify,
    }
    models: dict[str, BootstrapResult] = {
        name: bootstrap_ci(y, pred, spec, seed=rng, **options)
        for name, pred in (("a", a), ("b", b))
    }
    tests: list[PairedResult] = [paired_bootstrap(y, a, b, spec, seed=rng, **options)]
    if spec.name == "auc":
        tests.append(delong_test(y, a, b, confidence=confidence))
    if spec.name == "accuracy":
        tests.append(mcnemar(y, a, b, variant="mid-p", confidence=confidence))
    tests.append(permutation_test(y, a, b, spec, n_permutations=n_resamples, seed=rng))
    report: dict[str, Any] = {
        "metric": spec.name,
        "higher_is_better": spec.higher_is_better,
        "n": len(y),
        "positives": int(np.sum(y == 1)) if spec.is_classification else None,
        "threshold": threshold if spec.kind == "label" else None,
        "confidence": confidence,
        "ci_method": method,
        "n_resamples": n_resamples,
        "seed": seed,
        "models": {name: asdict(result) for name, result in models.items()},
        "difference": tests[0].difference,
        "tests": [asdict(result) for result in tests],
    }
    return dict(_finite_or_none(report))


def format_report(report: dict[str, Any], files: dict[str, str]) -> str:
    """Plain-text rendering of :func:`compare`'s report."""
    metric = report["metric"]
    level = f"{round(100 * report['confidence'])}%"
    counts = f"{report['n']} examples"
    if report["positives"] is not None:
        counts += f" ({report['positives']} positive)"
    lines = [
        f"evalstats compare: {metric} on {counts}",
        f"{'higher' if report['higher_is_better'] else 'lower'} is better; intervals are "
        f"{level} {report['ci_method']} bootstrap, {report['n_resamples']} resamples, "
        f"seed {report['seed']}",
    ]
    if report["threshold"] is not None:
        lines.append(f"predictions thresholded at {report['threshold']:g}")
    lines += ["", f"{'model':<7}{metric:>10}  {level + ' CI':<20}  file"]
    for name in ("a", "b"):
        model = report["models"][name]
        interval = f"[{model['ci_low']:.4f}, {model['ci_high']:.4f}]"
        lines.append(f"{name.upper():<7}{model['estimate']:>10.4f}  {interval:<20}  {files[name]}")
    lines += [
        "",
        f"difference A - B = {report['difference']:+.4f}",
        f"{'test':<18}{level + ' CI of difference':<24}p-value",
    ]
    for test in report["tests"]:
        if test["ci_low"] is None:
            interval = "-"
        else:
            interval = f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]"
        name = _TEST_NAMES.get(test["method"], test["method"])
        lines.append(f"{name:<18}{interval:<24}{test['p_value']:.4g}")
    boot = report["tests"][0]
    if boot["ci_low"] > 0 or boot["ci_high"] < 0:
        better = "A" if (boot["ci_low"] > 0) == report["higher_is_better"] else "B"
        verdict = f"the paired bootstrap interval excludes 0: {better} is better on {metric}"
    else:
        verdict = "the paired bootstrap interval includes 0: no difference detected"
    lines += ["", verdict]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalstats", description="Compare two models evaluated on the same examples."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    cmp = commands.add_parser(
        "compare",
        help="compare two models on one metric",
        description="Bootstrap CIs for each model, then paired tests of the difference.",
    )
    cmp.add_argument("--labels", type=Path, required=True, help="CSV of true labels or targets")
    cmp.add_argument("--a", type=Path, required=True, help="CSV of model A's predictions")
    cmp.add_argument("--b", type=Path, required=True, help="CSV of model B's predictions")
    cmp.add_argument("--metric", required=True, choices=sorted(METRICS), help="metric to compare")
    cmp.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="scores at or above it count as 1 for label metrics (default 0.5)",
    )
    cmp.add_argument("--n-resamples", type=int, default=10_000, help="default 10000")
    cmp.add_argument("--confidence", type=float, default=0.95, help="default 0.95")
    cmp.add_argument("--ci-method", choices=["bca", "percentile"], default="bca")
    cmp.add_argument("--stratify", action="store_true", help="resample within each class")
    cmp.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    cmp.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``evalstats`` command. Returns the exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        y, a, b = load_aligned(args.labels, args.a, args.b)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = compare(
                y,
                a,
                b,
                get_metric(args.metric),
                threshold=args.threshold,
                n_resamples=args.n_resamples,
                confidence=args.confidence,
                method=args.ci_method,
                stratify=args.stratify,
                seed=args.seed,
            )
    except (InputError, ValueError) as error:
        print(f"evalstats: error: {error}", file=sys.stderr)
        return 2
    for warning in caught:
        print(f"evalstats: warning: {warning.message}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report, {"a": str(args.a), "b": str(args.b)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
