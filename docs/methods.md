# Methods

The formulas evalstats implements, the conventions it chooses where the literature offers
several, and the sources. Notation: `n` examples, labels `y_i` in {0, 1}, predictions or scores
`s_i`, weights `w_i` (all ones without sample weights).

## Metrics

| metric | definition | undefined when |
|---|---|---|
| accuracy | sum w 1{pred = y} / sum w | never (weights must have a positive entry) |
| precision | TP / (TP + FP) | no positive predictions |
| recall | TP / (TP + FN) | no positive labels |
| F1 | 2 TP / (2 TP + FP + FN) | no positive labels and no positive predictions |
| ROC AUC | P(s_pos > s_neg) + 1/2 P(s_pos = s_neg), weighted over pairs | one class only |
| average precision | sum over distinct thresholds of (recall increase) x (precision) | no positive labels |
| log loss | mean of -[y log p + (1 - y) log(1 - p)], p clipped to [eps, 1 - eps] | never |
| Brier score | mean of (y - p)^2 | never |
| MAE, RMSE | mean of abs(y - s); square root of the mean of (y - s)^2 | never |
| R2 | 1 - SS_res / SS_tot, weighted | constant y |

The AUC definition with one half for ties equals the trapezoidal area under the ROC curve.
Average precision is the step-wise sum, without interpolation, as in scikit-learn. Log loss uses
`eps` = 2.2e-16 (float64 machine epsilon), scikit-learn's convention. Every value matches
scikit-learn to 1e-12 (`tests/test_metrics.py`).

### One implementation, three uses

Each metric is a batch function `batch(y, pred, weights)` where `pred` and `weights` have shape
`(n,)` or `(rows, n)` and row `r` of the result uses `pred[r]` and `weights[r]`:

- point estimate: one row, the sample weights;
- bootstrap: `weights = counts * sample_weight`, where `counts[r, i]` is how many times example
  `i` was drawn in resample `r` (a resample is never copied);
- permutation test: `pred` is a matrix of swapped predictions.

For the rank metrics the scores are sorted once per row. In a bootstrap the scores are shared,
so they are sorted once and only the count matrix is permuted. Ties are handled by computing, for
each sorted position, the first and last index of its tie group with two cumulative scans.

## Bootstrap intervals

Resample counts are drawn in blocks of about 2^20 matrix entries (with `bincount` on uniform
draws), so memory stays bounded for any `n`. With `stratify=True` each class is resampled on its
own and the class counts stay fixed, which avoids resamples with one class when a class is rare.

**Percentile:** the `alpha/2` and `1 - alpha/2` quantiles of the replicates.

**BCa** (Efron, 1987): with `G` the empirical CDF of the replicates and `theta` the estimate,

- bias correction `z0 = Phi^-1(G(theta))`, where `G` counts ties with the estimate as one half;
- acceleration `a = sum d_j^3 / (6 (sum d_j^2)^(3/2))`, `d_j = mean(theta_(.)) - theta_(j)`,
  from the jackknife values `theta_(j)`;
- interval endpoints at the quantiles `Phi(z0 + (z0 + z) / (1 - a (z0 + z)))` for
  `z = Phi^-1(alpha/2)` and `Phi^-1(1 - alpha/2)`.

The leave-one-out jackknife costs O(n^2). Above 1000 examples the data are split at random into
1000 groups and one group is deleted at a time. Summing the influence of the members of a group
multiplies the numerator and the denominator of `a` by the same factor, so the grouped estimate
targets the same quantity; `tests/test_bootstrap.py` checks that it averages to the
leave-one-out value. When the estimate lies outside the replicate distribution (`G(theta)` is 0
or 1), `z0` is infinite, so evalstats warns and reports the percentile interval instead.

**p-values for a difference** (paired bootstrap, slices): the smallest `alpha` at which the
reported interval excludes zero. For the percentile interval this is
`2 min(G(0), 1 - G(0))`; for BCa the same mapping that moves the endpoints is inverted at zero.
`G(0)` is kept inside `[1/(B+1), B/(B+1)]` and the p-value is never reported below `2/(B+1)`,
the resolution of `B` resamples. These p-values are approximate; the permutation test has exact
size.

## Paired comparisons

**Paired bootstrap:** the same count matrix weights both models, so a resample keeps the
correlation between their errors. The difference `metric(A) - metric(B)` gets a percentile or
BCa interval and the p-value above.

**Permutation test:** under the null hypothesis the two predictions of each example are
exchangeable. Each pair is swapped with probability one half; the p-value is
`(1 + #{|T*| >= |T|}) / (B + 1)`. Counting the observed arrangement makes the Monte Carlo test
exact in size for any `B` (Phipson and Smyth, 2010).

**McNemar:** with `n10` examples where only A is right and `n01` where only B is right, under
the null `n10 ~ Binomial(n10 + n01, 1/2)`. The exact test doubles the smaller tail,
`min(1, 2 P(X <= min(n10, n01)))`, and is conservative; the mid-p version subtracts the
probability of the observed value, `2 P(X <= k) - P(X = k)`, and keeps the size closer to the
nominal level (Fagerland, Lydersen and Laake, 2013). The interval for the accuracy difference is
Agresti and Min's (2005): add 1/2 to each of the four cells, then use the Wald interval
`(n10' - n01') / n' +- z sqrt((n10' + n01') - (n10' - n01')^2 / n') / n'`.

**DeLong:** for model `r`, the placement of a positive `i` is
`V10_r(i) = mean_j psi(s_r(i), s_r(j))` over negatives `j`, and the placement of a negative `j`
is `V01_r(j) = mean_i psi(s_r(i), s_r(j))` over positives `i`, with
`psi(x, y) = 1{x > y} + 1/2 1{x = y}`. The AUCs are the means of the placements and their
covariance is estimated by `S10 / m + S01 / n0`, the sample covariance matrices of the
placements of the `m` positives and the `n0` negatives (DeLong, DeLong and Clarke-Pearson,
1988). Placements come from midranks in O(n log n) (Sun and Xu, 2014):
`V10(i) = (rank of s(i) among all - rank among positives) / n0`. The test statistic is the AUC
difference over `sqrt(Var A + Var B - 2 Cov(A, B))`.

## Multiple comparisons

With sorted p-values `p(1) <= ... <= p(m)`:

- Holm (1979): `adj(i) = max over j <= i of min(1, (m - j + 1) p(j))`; controls the family-wise
  error rate under any dependence.
- Benjamini and Hochberg (1995): `adj(i) = min over j >= i of min(1, m p(j) / j)`; controls the
  false discovery rate at `q m0 / m` for independent or positively dependent tests.

## Calibration

Bins are right-closed, `(e_k, e_k+1]`, with the first bin also holding `e_0`: the convention of
`sklearn.calibration.calibration_curve`, which `tests/test_calibration.py` checks against.
Equal-mass bins put their edges at the percentiles of the predictions. ECE is
`sum_b (n_b / n) |mean prediction_b - frequency_b|` over non-empty bins. It is biased upward when
bins hold few examples.

**Temperature scaling** (Guo et al., 2017) fits `p' = sigmoid(z / T)` on the logits
`z = logit(p)` (probabilities clipped to [1e-12, 1 - 1e-12]). With `u = 1/T` the negative
log-likelihood is convex and its derivative `sum (sigmoid(u z) - y) z` increases with `u`, so
the estimate is the root of the derivative, found by bracketing (Brent's method). A positive root
exists exactly when the derivative at zero is negative (scores are positively associated with
the labels) and some example lies on the wrong side of zero; otherwise evalstats raises.

**Platt scaling** (Platt, 1999) fits `p' = sigmoid(a z + b)` by plain maximum likelihood (no
label smoothing), with Newton's method and step halving from `(1, 0)`. The maximum exists when
the scores of the two classes overlap: the largest negative score exceeds the smallest positive
score and the largest positive score exceeds the smallest negative score. Without overlap the
classes are separated and the slope would grow without bound, so evalstats raises.

## Slices

Examples are resampled within each slice (and within each class of each slice with
`stratify=True`), one bootstrap for all slices. For slice `s`, the replicates of
`metric(slice)` give its percentile interval and the replicates of
`metric(slice) - metric(rest)` give the interval and p-value of the difference. The p-values of
the eligible slices are then adjusted together. A slice is skipped when it, or its rest, has
fewer than `min_support` examples, or when the metric is undefined on it.

## References

- Agresti, A. and Min, Y. (2005). Simple improved confidence intervals for comparing matched
  proportions. *Statistics in Medicine* 24(5), 729-740.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *Journal of the Royal Statistical Society B* 57(1),
  289-300.
- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing the areas under two
  or more correlated receiver operating characteristic curves: a nonparametric approach.
  *Biometrics* 44(3), 837-845.
- Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical
  Association* 82(397), 171-185.
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Fagerland, M. W., Lydersen, S. and Laake, P. (2013). The McNemar test for binary matched-pairs
  data: mid-p and asymptotic are better than exact conditional. *BMC Medical Research
  Methodology* 13, 91.
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural
  networks. *Proceedings of ICML*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
  Journal of Statistics* 6(2), 65-70.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated
  proportions or percentages. *Psychometrika* 12(2), 153-157.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero: calculating
  exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and
  Molecular Biology* 9(1).
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press.
- Sun, X. and Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas
  under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*
  21(11), 1389-1393.
