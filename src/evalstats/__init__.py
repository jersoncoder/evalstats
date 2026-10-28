"""Statistical tools for comparing machine-learning models.

Metrics with bootstrap confidence intervals, paired tests between two models evaluated on the
same examples, multiple-comparison adjustments, calibration analysis and slice analysis.
"""

from evalstats.bootstrap import BootstrapResult, bootstrap_ci
from evalstats.calibration import (
    PlattScaling,
    ReliabilityCurve,
    TemperatureScaling,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    reliability_curve,
)
from evalstats.metrics import METRICS, Metric, get_metric
from evalstats.multiple import adjust_p_values, benjamini_hochberg, holm
from evalstats.paired import (
    PairedResult,
    delong_covariance,
    delong_test,
    mcnemar,
    paired_bootstrap,
    permutation_test,
)
from evalstats.slices import SkippedSlice, SliceReport, SliceResult, slice_analysis

__version__ = "0.1.0"

__all__ = [
    "METRICS",
    "BootstrapResult",
    "Metric",
    "PairedResult",
    "PlattScaling",
    "ReliabilityCurve",
    "SkippedSlice",
    "SliceReport",
    "SliceResult",
    "TemperatureScaling",
    "__version__",
    "adjust_p_values",
    "benjamini_hochberg",
    "bootstrap_ci",
    "delong_covariance",
    "delong_test",
    "expected_calibration_error",
    "fit_platt",
    "fit_temperature",
    "get_metric",
    "holm",
    "mcnemar",
    "paired_bootstrap",
    "permutation_test",
    "reliability_curve",
    "slice_analysis",
]
