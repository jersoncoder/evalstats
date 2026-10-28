"""Simulation study behind the tables in the README.

Draws many datasets from models where the truth is known and reports, as Markdown tables:

1. coverage of 95% bootstrap intervals (percentile and BCa) for three metrics and three sizes;
2. the rejection rate of each paired test at level 0.05 when the two models are equally good;
3. the false discovery rate of Benjamini-Hochberg and the family-wise error rate of Holm.

Every cell has its own seed derived from one root seed, so the output is reproducible:

    python scripts/simulation_study.py            # several minutes
    python scripts/simulation_study.py --quick    # a tenth of the repetitions
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

import evalstats as es
from evalstats._validation import FloatArray

ROOT_SEED = 2026
Dataset = tuple[FloatArray, FloatArray]


def binormal(n: int) -> Callable[[np.random.Generator], Dataset]:
    """Half negatives ~ N(0, 1), half positives ~ N(1, 1): AUC = Phi(1 / sqrt(2))."""
    y = (np.arange(n) % 2).astype(np.float64)
    return lambda rng: (y, rng.normal(size=n) + y)


def bernoulli_labels(n: int) -> Callable[[np.random.Generator], Dataset]:
    """Balanced labels, predictions correct with probability 0.8: accuracy = 0.8."""

    def draw(rng: np.random.Generator) -> Dataset:
        y = rng.integers(0, 2, n).astype(np.float64)
        return y, np.where(rng.random(n) < 0.8, y, 1.0 - y)

    return draw


def gaussian_errors(n: int) -> Callable[[np.random.Generator], Dataset]:
    """Errors ~ N(0, 1): mean absolute error = sqrt(2 / pi)."""

    def draw(rng: np.random.Generator) -> Dataset:
        y = rng.normal(size=n)
        return y, y + rng.normal(size=n)

    return draw


def coverage_table(repetitions: int, seeds: np.random.SeedSequence) -> list[str]:
    cases = [
        ("auc", binormal, float(ndtr(1 / np.sqrt(2))), True),
        ("accuracy", bernoulli_labels, 0.8, False),
        ("mae", gaussian_errors, float(np.sqrt(2 / np.pi)), False),
    ]
    sizes = (50, 200, 1000)
    se = np.sqrt(0.95 * 0.05 / repetitions)
    lines = [
        f"Coverage of 95% intervals ({repetitions} datasets per cell, 1000 resamples each; "
        f"Monte Carlo SE {se:.3f})",
        "",
        "| metric | method | " + " | ".join(f"n = {n}" for n in sizes) + " |",
        "|---|---|" + "---:|" * len(sizes),
    ]
    for metric, generator, truth, stratify in cases:
        for method in ("percentile", "bca"):
            cells = []
            for n in sizes:
                rng = np.random.default_rng(seeds.spawn(1)[0])
                draw = generator(n)
                covered = 0
                for _ in range(repetitions):
                    y, pred = draw(rng)
                    result = es.bootstrap_ci(
                        y,
                        pred,
                        metric,
                        n_resamples=1000,
                        method=method,
                        stratify=stratify,
                        seed=rng,
                    )
                    covered += result.ci_low <= truth <= result.ci_high
                cells.append(f"{covered / repetitions:.3f}")
            lines.append(f"| {metric} | {method} | " + " | ".join(cells) + " |")
    return lines


def null_size_table(repetitions: int, seeds: np.random.SeedSequence) -> list[str]:
    """Every test runs on the same datasets, so the rates are directly comparable."""
    n, alpha = 200, 0.05
    y = (np.arange(n) % 2).astype(np.float64)
    names = [
        "McNemar, exact (accuracy at threshold 0.5)",
        "McNemar, mid-p (accuracy at threshold 0.5)",
        "permutation (Brier score)",
        "DeLong (AUC)",
        "paired bootstrap, BCa (AUC)",
        "paired bootstrap, percentile (AUC)",
    ]
    rejections = np.zeros(len(names))
    rng = np.random.default_rng(seeds.spawn(1)[0])
    for _ in range(repetitions):
        # Exchangeable models: same signal, shared noise, independent noise of equal size.
        shared = rng.normal(size=n)
        a = y + shared + rng.normal(size=n)
        b = y + shared + rng.normal(size=n)
        prob_a, prob_b = 1 / (1 + np.exp(-(a - 0.5))), 1 / (1 + np.exp(-(b - 0.5)))
        p_values = [
            es.mcnemar(y, prob_a >= 0.5, prob_b >= 0.5, variant="exact").p_value,
            es.mcnemar(y, prob_a >= 0.5, prob_b >= 0.5, variant="mid-p").p_value,
            es.permutation_test(y, prob_a, prob_b, "brier", n_permutations=999, seed=rng).p_value,
            es.delong_test(y, a, b).p_value,
            es.paired_bootstrap(y, a, b, "auc", n_resamples=1000, seed=rng).p_value,
            es.paired_bootstrap(
                y, a, b, "auc", n_resamples=1000, method="percentile", seed=rng
            ).p_value,
        ]
        rejections += np.array(p_values) <= alpha
    se = np.sqrt(alpha * (1 - alpha) / repetitions)
    lines = [
        f"Rejection rate at level 0.05 when both models are equally good (n = {n}, "
        f"{repetitions} datasets shared by all tests; Monte Carlo SE {se:.4f})",
        "",
        "| test | rejection rate |",
        "|---|---:|",
    ]
    lines += [
        f"| {name} | {count / repetitions:.4f} |"
        for name, count in zip(names, rejections, strict=True)
    ]
    return lines


def multiple_testing_table(repetitions: int, seeds: np.random.SeedSequence) -> list[str]:
    m, m0, q = 20, 16, 0.05
    rng = np.random.default_rng(seeds.spawn(1)[0])
    is_null = np.arange(m) < m0
    fdp = np.empty(repetitions)
    family_errors = np.empty(repetitions)
    for r in range(repetitions):
        p = np.where(is_null, rng.random(m), norm.sf(rng.normal(3.0, 1.0, m)))
        bh = es.benjamini_hochberg(p) <= q
        fdp[r] = np.sum(bh & is_null) / max(1, int(np.sum(bh)))
        family_errors[r] = np.any((es.holm(p) <= q) & is_null)
    fdr_se = fdp.std(ddof=1) / np.sqrt(repetitions)
    return [
        f"Multiple comparisons: {m} independent tests, {m0} true nulls, alternatives with "
        f"z ~ N(3, 1), level {q} ({repetitions} repetitions)",
        "",
        "| procedure | quantity | estimate | theory |",
        "|---|---|---:|---:|",
        f"| Benjamini-Hochberg | false discovery rate | {fdp.mean():.4f} (SE {fdr_se:.4f}) "
        f"| {q * m0 / m:.4f} |",
        f"| Holm | family-wise error rate | {family_errors.mean():.4f} | at most {q} |",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="a tenth of the repetitions")
    args = parser.parse_args()
    scale = 10 if args.quick else 1
    start = time.perf_counter()
    root = np.random.SeedSequence(ROOT_SEED)
    coverage_seeds, null_seeds, multiple_seeds = root.spawn(3)
    sections = [
        coverage_table(1000 // scale, coverage_seeds),
        null_size_table(2000 // scale, null_seeds),
        multiple_testing_table(20_000 // scale, multiple_seeds),
    ]
    print("\n\n".join("\n".join(section) for section in sections))
    print(f"\n(root seed {ROOT_SEED}; {time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
