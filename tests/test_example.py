"""The example script runs and still shows what the README says it shows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "compare_two_models.py"


def test_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("compare_two_models", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()
    out = capsys.readouterr().out
    assert "Logistic regression is ahead by" in out
    assert "but the interval includes zero" in out
    assert "The interval excludes zero: the tree ranks tumours worse" in out
    assert "temperature (T = " in out
