"""Wall-clock time of the resampling procedures, as reported in the README.

    python scripts/timings.py

Each line is the best of three runs. Timings depend on the machine and on NumPy's build; the
script prints both so the numbers can be read in context.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import evalstats as es


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def best_of(runs: int, fn: Callable[[], object]) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def cases(n: int, resamples: int) -> list[tuple[str, Callable[[], object]]]:
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.3).astype(np.float64)
    a = rng.normal(size=n) + y
    b = rng.normal(size=n) + 0.8 * y
    prob = 1 / (1 + np.exp(-a))
    return [
        (
            "bootstrap_ci, accuracy, BCa",
            lambda: es.bootstrap_ci(y, prob >= 0.5, "accuracy", n_resamples=resamples),
        ),
        (
            "bootstrap_ci, log loss, BCa",
            lambda: es.bootstrap_ci(y, prob, "log_loss", n_resamples=resamples),
        ),
        ("bootstrap_ci, AUC, BCa", lambda: es.bootstrap_ci(y, a, "auc", n_resamples=resamples)),
        (
            "paired_bootstrap, AUC, BCa",
            lambda: es.paired_bootstrap(y, a, b, "auc", n_resamples=resamples),
        ),
        (
            "permutation_test, AUC",
            lambda: es.permutation_test(y, a, b, "auc", n_permutations=resamples),
        ),
    ]


def main() -> None:
    resamples = 10_000
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, {cpu_name()}")
    print()
    print(f"| procedure | n | seconds for {resamples} resamples |")
    print("|---|---:|---:|")
    for n in (1_000, 10_000):
        for name, fn in cases(n, resamples):
            print(f"| {name} | {n} | {best_of(3, fn):.2f} |")


if __name__ == "__main__":
    main()
