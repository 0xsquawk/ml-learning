# Statistical Tests for Machine Learning — Cheat Sheet

Quick-reference for the tests you'll actually reach for during EDA, feature selection, and model comparison. All snippets use `scipy.stats`, `statsmodels`, or `sklearn`.

```python
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
```

## 1. Normality Tests
| Test | Use when |
|---|---|
| Shapiro-Wilk | n < 5000, most powerful for small samples |
| D'Agostino K² | Larger samples, checks skew + kurtosis |
| Anderson-Darling | More weight on distribution tails |

```python
stat, p = stats.shapiro(x)
stat, p = stats.normaltest(x)                 # D'Agostino
result = stats.anderson(x, dist='norm')
```

## 2. Comparing Two Groups
| Test | Use when |
|---|---|
| Independent t-test | 2 independent groups, normal data |
| Welch's t-test | Same, unequal variances |
| Paired t-test | Same subjects, before/after |
| Mann-Whitney U | Non-normal, independent groups |
| Wilcoxon signed-rank | Non-normal, paired samples |

```python
stats.ttest_ind(a, b, equal_var=True)
stats.ttest_ind(a, b, equal_var=False)        # Welch
stats.ttest_rel(a, b)                         # paired
stats.mannwhitneyu(a, b, alternative='two-sided')
stats.wilcoxon(a, b)
```

## 3. Comparing 3+ Groups
| Test | Use when |
|---|---|
| One-way ANOVA | Normal, equal variance, independent |
| Kruskal-Wallis | Non-normal / ordinal, independent |
| Friedman | Non-normal, repeated measures |

```python
stats.f_oneway(g1, g2, g3)
stats.kruskal(g1, g2, g3)
stats.friedmanchisquare(g1, g2, g3)
```

## 4. Correlation
| Test | Use when |
|---|---|
| Pearson r | Linear relationship, continuous, normal |
| Spearman ρ | Monotonic, ordinal or non-normal |
| Kendall τ | Small n, many tied ranks |

```python
stats.pearsonr(x, y)
stats.spearmanr(x, y)
stats.kendalltau(x, y)
```

## 5. Categorical Association
| Test | Use when |
|---|---|
| Chi-square independence | Two categorical vars, expected counts ≥ 5 |
| Fisher's exact | Small samples / low expected counts (2x2) |
| McNemar's | Paired categorical (e.g. before/after classifier) |

```python
chi2, p, dof, expected = stats.chi2_contingency(table)
stats.fisher_exact(table_2x2)
mcnemar(table_2x2, exact=True)
```

## 6. Variance Equality
| Test | Use when |
|---|---|
| Levene's | Robust to non-normality |
| Bartlett's | Assumes normality, more sensitive |
| F-test | Two samples, normal data |

```python
stats.levene(a, b, c)
stats.bartlett(a, b, c)
f_stat = np.var(a, ddof=1) / np.var(b, ddof=1)
```

## 7. Goodness-of-Fit / Distribution Comparison
| Test | Use when |
|---|---|
| Kolmogorov-Smirnov (1-sample) | Sample vs a theoretical distribution |
| KS (2-sample) | Compare two empirical distributions (e.g. train vs test drift) |
| Anderson-Darling (k-sample) | Compare multiple samples, tail-sensitive |

```python
stats.kstest(x, 'norm')
stats.ks_2samp(sample1, sample2)
stats.anderson_ksamp([sample1, sample2])
```

## 8. Model / Classifier Comparison
| Test | Use when |
|---|---|
| Paired t-test on CV folds | Compare 2 models' scores across same folds |
| Wilcoxon signed-rank on CV folds | Same, non-normal score distribution |
| McNemar's test | Compare 2 classifiers' predictions on same test set |
| 5x2cv paired t-test | More robust than plain CV t-test (Dietterich) |
| DeLong's test | Compare two models' ROC-AUC |

```python
# CV fold comparison
stats.ttest_rel(cv_scores_model_a, cv_scores_model_b)
stats.wilcoxon(cv_scores_model_a, cv_scores_model_b)

# McNemar's for classifier predictions
b = np.sum((pred_a == y_true) & (pred_b != y_true))
c = np.sum((pred_a != y_true) & (pred_b == y_true))
table = [[0, b], [c, 0]]
mcnemar(table, exact=True)

# 5x2cv (mlxtend)
from mlxtend.evaluate import paired_ttest_5x2cv
t, p = paired_ttest_5x2cv(estimator1=model_a, estimator2=model_b, X=X, y=y)
```

## 9. Feature Selection
| Test | Use when |
|---|---|
| ANOVA F-test | Continuous feature vs categorical target |
| Chi-square | Categorical feature vs categorical target (non-negative) |
| Mutual information | Any relationship type, no distribution assumption |

```python
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
f_scores, p_values = f_classif(X, y)
chi2_scores, p_values = chi2(X, y)
mi_scores = mutual_info_classif(X, y)
```

## 10. A/B Testing & Proportions
| Test | Use when |
|---|---|
| Two-proportion z-test | Compare conversion rates between 2 groups |
| Chi-square for proportions | Same, or 2x2/RxC contingency form |

```python
from statsmodels.stats.proportion import proportions_ztest
count = np.array([successes_a, successes_b])
nobs = np.array([n_a, n_b])
z_stat, p_value = proportions_ztest(count, nobs)
```

## 11. Multiple Testing Correction
| Method | Use when |
|---|---|
| Bonferroni | Strict family-wise error control, few tests |
| Benjamini-Hochberg (FDR) | Many tests (e.g. per-feature p-values), less conservative |

```python
from statsmodels.stats.multitest import multipletests
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

## 12. Resampling-Based Tests
| Test | Use when |
|---|---|
| Permutation test | No distributional assumptions, exact under null |
| Bootstrap CI | Estimate uncertainty of any statistic/metric |

```python
res = stats.permutation_test((a, b), lambda x, y: np.mean(x) - np.mean(y),
                              n_resamples=10000, alternative='two-sided')

res = stats.bootstrap((data,), np.mean, n_resamples=10000, confidence_level=0.95)
```

---

### Quick decision guide
- **Normal + 2 groups** → t-test · **Non-normal + 2 groups** → Mann-Whitney/Wilcoxon
- **Normal + 3+ groups** → ANOVA · **Non-normal + 3+ groups** → Kruskal-Wallis
- **Two categorical vars** → Chi-square (or Fisher's if counts are small)
- **Comparing models on the same data** → McNemar's (classification) or paired t-test/Wilcoxon on CV folds
- **Many simultaneous tests** → correct with Benjamini-Hochberg, not raw p-values
