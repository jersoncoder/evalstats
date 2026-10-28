"""Compare models on the breast cancer data bundled with scikit-learn.

Shows a difference that looks meaningful on point estimates but whose interval includes zero,
a difference that is real, and the recalibration of an overconfident model.

Run from the repository root after installing the development extras:

    python examples/compare_two_models.py
"""

from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import evalstats as es
from evalstats import PairedResult
from evalstats._validation import FloatArray
from evalstats.metrics import log_loss

SEED = 0
RESAMPLES = 10_000


def interval(low: float | None, high: float | None) -> str:
    assert low is not None and high is not None
    return f"[{low:+.4f}, {high:+.4f}]"


def row(label: str, result: es.BootstrapResult) -> str:
    return (
        f"   {label:<22}{result.estimate:.4f}  95% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
    )


def bootstrap_p(result: PairedResult) -> str:
    # B resamples cannot resolve p-values below 2 / (B + 1); report that bound as such.
    floor = 2 / (RESAMPLES + 1)
    return f"p <= {floor:.1e}" if result.p_value <= floor else f"p = {result.p_value:.3f}"


def verdict(result: PairedResult) -> str:
    assert result.ci_low is not None and result.ci_high is not None
    if result.ci_low <= 0.0 <= result.ci_high:
        return "includes zero"
    return "excludes zero"


def main() -> None:
    data = load_breast_cancer()
    x = data.data
    y = (data.target == 0).astype(float)  # in scikit-learn 0 is malignant; make it the positive
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, stratify=y, random_state=SEED
    )
    print(
        f"Breast cancer data: {len(y)} tumours, {x.shape[1]} features, "
        f"{int(y.sum())} malignant (the positive class)."
    )
    print(
        f"Stratified 50/50 split (seed {SEED}): {len(y_train)} train, {len(y_test)} test "
        f"({int(y_test.sum())} malignant). Intervals: BCa bootstrap, {RESAMPLES} resamples.\n"
    )

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    forest = RandomForestClassifier(n_estimators=300, random_state=SEED)
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    scores: dict[str, FloatArray] = {}
    for name, model in (("logistic", logistic), ("forest", forest), ("tree", tree)):
        scores[name] = model.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    labels = {name: (s >= 0.5).astype(float) for name, s in scores.items()}

    # 1. A difference that is not resolved by 285 test examples.
    lr_acc = es.bootstrap_ci(y_test, labels["logistic"], "accuracy", seed=SEED)
    rf_acc = es.bootstrap_ci(y_test, labels["forest"], "accuracy", seed=SEED)
    boot = es.paired_bootstrap(y_test, labels["logistic"], labels["forest"], "accuracy", seed=SEED)
    mcn = es.mcnemar(y_test, labels["logistic"], labels["forest"])
    print("1. Logistic regression vs random forest, accuracy")
    print(row("logistic regression", lr_acc))
    print(row("random forest", rf_acc))
    print(
        f"   {'difference':<22}{boot.difference:+.4f}  paired bootstrap 95% CI "
        f"{interval(boot.ci_low, boot.ci_high)}, {bootstrap_p(boot)}"
    )
    print(
        f"   {'McNemar mid-p':<22}p = {mcn.p_value:.3f} (only logistic right: "
        f"{mcn.details['a_right_b_wrong']}, only forest right: {mcn.details['a_wrong_b_right']})"
    )
    print(
        f"   Logistic regression is ahead by {100 * boot.difference:.1f} points, but the "
        f"interval {verdict(boot)}.\n"
    )

    # 2. A real difference.
    lr_auc = es.bootstrap_ci(y_test, scores["logistic"], "auc", seed=SEED)
    tree_auc = es.bootstrap_ci(y_test, scores["tree"], "auc", seed=SEED)
    delong = es.delong_test(y_test, scores["logistic"], scores["tree"])
    boot_auc = es.paired_bootstrap(y_test, scores["logistic"], scores["tree"], "auc", seed=SEED)
    print("2. Logistic regression vs a depth-3 decision tree, ROC AUC")
    print(row("logistic regression", lr_auc))
    print(row("decision tree", tree_auc))
    print(
        f"   {'difference':<22}{delong.difference:+.4f}  DeLong 95% CI "
        f"{interval(delong.ci_low, delong.ci_high)}, p = {delong.p_value:.1e}"
    )
    print(
        f"   {'':<22}{'':>7}  paired bootstrap 95% CI "
        f"{interval(boot_auc.ci_low, boot_auc.ci_high)}, {bootstrap_p(boot_auc)}"
    )
    print(f"   The interval {verdict(delong)}: the tree ranks tumours worse.\n")

    # 3. Recalibration on a validation split held out from the training half.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.3, stratify=y_train, random_state=SEED
    )
    bayes = GaussianNB().fit(x_fit, y_fit)
    val_prob = bayes.predict_proba(x_val)[:, 1]
    test_prob = bayes.predict_proba(x_test)[:, 1]
    temperature = es.fit_temperature(y_val, val_prob)
    platt = es.fit_platt(y_val, val_prob)
    print("3. Calibration of Gaussian naive Bayes")
    print(
        f"   Trained on {len(y_fit)} examples, recalibrated on {len(y_val)} held-out training "
        "examples, measured on the test split."
    )
    print(f"   {'':<30}{'log loss':>9}  {'ECE (10 equal-mass bins)':>25}")
    calibrated = {
        "as trained": test_prob,
        f"temperature (T = {temperature.temperature:.2f})": temperature.transform(test_prob),
        f"Platt (a = {platt.slope:.3f}, b = {platt.intercept:+.2f})": platt.transform(test_prob),
    }
    for name, prob in calibrated.items():
        ece = es.expected_calibration_error(y_test, prob, n_bins=10, strategy="quantile")
        print(f"   {name:<30}{log_loss(y_test, prob):>9.4f}  {ece:>25.4f}")
    gain = es.paired_bootstrap(
        y_test, test_prob, temperature.transform(test_prob), "log_loss", seed=SEED
    )
    print(
        f"   Log loss reduction from temperature scaling: {gain.difference:.4f}, paired "
        f"bootstrap 95% CI {interval(gain.ci_low, gain.ci_high)}, {bootstrap_p(gain)}."
    )


if __name__ == "__main__":
    main()
