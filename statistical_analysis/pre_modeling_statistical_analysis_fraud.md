# Pre-Modeling Statistical Analysis — Credit Card Fraud Detection

Before touching a model, the goal is to characterize the data well enough that every downstream choice (scaling, resampling, feature set, algorithm family, evaluation metric) is justified by evidence, not habit. Below is the full battery, grouped by what question each test answers.

---

## 1. Data Quality & Structural Checks

These aren't "statistical tests" in the inferential sense, but they gate everything else — a test run on unclean data is meaningless.

**1.1 Missing value audit**
- Compute missingness per column: `df.isnull().mean().sort_values(ascending=False)`
- Check if missingness is random (MCAR) or correlated with the target — cross-tab missing indicator vs `is_fraud` using a Chi-square test. If fraud rows have systematically different missingness, that's leakage risk or a real signal, not noise.

**1.2 Duplicate detection**
- Full-row duplicates: `df.duplicated().sum()`
- Near-duplicate/fuzzy duplicates on transaction amount + timestamp + card ID — relevant in fraud data because repeated attempts (card testing) are a fraud signature themselves, not just noise to drop.

**1.3 Data type and range sanity**
- Verify each column's dtype matches its semantic type (e.g., `trans_date_trans_time` as datetime, not object).
- Range checks: negative amounts, impossible lat/long, future-dated transactions, zip code format.

**1.4 Cardinality check (categorical features)**
- `df[col].nunique()` for merchant, category, job, city, state. High-cardinality categoricals (e.g., merchant, job) need a different encoding strategy (target encoding, hashing) than low-cardinality ones (one-hot).

**1.5 Data leakage scan**
- Check for post-event features accidentally included (e.g., a "chargeback flag" or "investigation status" column that wouldn't exist at prediction time).
- Check for ID leakage — verify that `cc_num` or `trans_num` don't act as a proxy that memorizes fraud rows (e.g., a card number that appears in train and test with identical fraud label).

---

## 2. Class Imbalance Analysis

This is the single most important characteristic of fraud data and drives your metric and resampling choices.

**2.1 Base rate**
- `df['is_fraud'].value_counts(normalize=True)` — typically fraud is 0.1%–0.5% of transactions.

**2.2 Imbalance ratio**
- Majority:minority ratio. Document it explicitly (e.g., 578:1) — this number determines whether you need SMOTE/undersampling/class-weighting and rules out plain accuracy as a metric.

```python
import pandas as pd

fraud_rate = df['is_fraud'].value_counts(normalize=True)
imbalance_ratio = fraud_rate[0] / fraud_rate[1]
print(f"Fraud rate: {fraud_rate[1]:.4%}")
print(f"Imbalance ratio: {imbalance_ratio:.1f} : 1")
```

---

## 3. Univariate Distribution Analysis

For every numeric feature (transaction amount, city population, lat/long deltas, age, etc.):

**3.1 Descriptive statistics**
- Mean, median, std, skewness, kurtosis: `df[col].describe()`, `df[col].skew()`, `df[col].kurt()`

**3.2 Normality tests**
- **Shapiro-Wilk test** — best for smaller samples (<5000); tests H₀: data is normally distributed.
- **Kolmogorov-Smirnov (KS) test** (`scipy.stats.kstest`) — better for large samples like this dataset; compares empirical CDF against a theoretical normal.
- **D'Agostino-Pearson test** (`scipy.stats.normaltest`) — combines skew + kurtosis into one omnibus test, works well at scale.
- Practical note: with ~1.85M rows, these tests will almost always reject normality (p → 0) because of statistical power, not necessarily practical non-normality. Pair with a Q-Q plot or histogram for visual judgment — don't rely on the p-value alone at this sample size.

```python
from scipy import stats

stat, p = stats.normaltest(df['amt'].dropna())
print(f"D'Agostino-Pearson: stat={stat:.2f}, p={p:.4f}")
```

**3.3 Why this matters**
- Determines whether to use parametric (t-test, Pearson correlation) or non-parametric (Mann-Whitney U, Spearman) tests downstream.
- Determines scaling choice: heavily skewed features (transaction amount almost always is) benefit from log-transform or `RobustScaler`/`PowerTransformer` (Yeo-Johnson/Box-Cox) rather than `StandardScaler`.

---

## 4. Bivariate Analysis — Feature vs Target (`is_fraud`)

This is the core of "which features actually separate fraud from non-fraud."

**4.1 Numeric feature vs binary target**
- **Mann-Whitney U test** (`scipy.stats.mannwhitneyu`) — preferred default here since amount/distance features are skewed and non-normal (per section 3). Tests whether the distribution of a feature differs between fraud=0 and fraud=1 groups without assuming normality.
- **Independent t-test / Welch's t-test** — only if normality roughly holds, or as a secondary confirmatory check. Welch's variant (`equal_var=False`) is safer since fraud/non-fraud group variances are rarely equal.
- **Point-biserial correlation** (`scipy.stats.pointbiserialr`) — correlates a continuous variable directly with the binary target; gives both direction and magnitude.

```python
from scipy.stats import mannwhitneyu, pointbiserialr

fraud_amt = df.loc[df.is_fraud == 1, 'amt']
legit_amt = df.loc[df.is_fraud == 0, 'amt']

u_stat, p_val = mannwhitneyu(fraud_amt, legit_amt, alternative='two-sided')
corr, corr_p = pointbiserialr(df['is_fraud'], df['amt'])
print(f"Mann-Whitney U p={p_val:.4g} | point-biserial r={corr:.3f} (p={corr_p:.4g})")
```

**4.2 Categorical feature vs binary target**
- **Chi-square test of independence** (`scipy.stats.chi2_contingency`) — for merchant category, state, job, gender vs `is_fraud`. Tests whether fraud rate differs significantly across categories.
- **Cramér's V** — effect size to accompany Chi-square (Chi-square p-value alone doesn't tell you how strong the association is, especially with 1.85M rows where trivial associations become "significant").

```python
import numpy as np
from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['category'], df['is_fraud'])
chi2, p, dof, expected = chi2_contingency(contingency)

n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
print(f"Chi-square p={p:.4g} | Cramér's V={cramers_v:.3f}")
```

**4.3 Multi-group numeric comparison (if >2 groups matter)**
- **Kruskal-Wallis H test** — non-parametric ANOVA equivalent, e.g. comparing transaction amount distributions across multiple merchant categories, not just fraud/non-fraud.
- **One-way ANOVA** — parametric alternative if normality holds within groups.

---

## 5. Correlation & Multicollinearity Analysis

**5.1 Pairwise correlation matrix**
- **Pearson** for linear relationships between normally-behaved numeric features.
- **Spearman rank correlation** — default choice here given skewed features; robust to monotonic non-linear relationships and outliers.
- Visualize as a heatmap; flag pairs with |r| > 0.8 as multicollinearity candidates.

**5.2 Variance Inflation Factor (VIF)**
- For any linear/logistic regression baseline, compute VIF per feature (`statsmodels.stats.outliers_influence.variance_inflation_factor`). VIF > 5–10 signals redundant features that will destabilize coefficient estimates.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[numeric_features].dropna()
vif_data = pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
```

**5.3 Mutual Information**
- **Mutual information (MI)** (`sklearn.feature_selection.mutual_info_classif`) — captures non-linear dependence between each feature and `is_fraud` that correlation coefficients miss entirely. Especially relevant for engineered features like time-since-last-transaction or distance-from-home.

---

## 6. Outlier Detection

Fraud is itself often an outlier phenomenon, so this step needs care — you're distinguishing "noisy outlier to clean" from "outlier that IS the signal."

**6.1 Univariate methods**
- **IQR method**: flag points outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`.
- **Z-score / Modified Z-score (using median absolute deviation)** — modified version is more robust and preferred given the skew.

**6.2 Multivariate methods**
- **Isolation Forest** — unsupervised, tree-based; run before modeling as a diagnostic to see if "outlier score" aligns with fraud label (informative, not for cleaning).
- **Local Outlier Factor (LOF)** — density-based, useful for catching fraud that's only anomalous relative to a local neighborhood (e.g., a normally low-spend card suddenly making a large purchase).
- **Mahalanobis distance** — accounts for correlation structure between features when flagging multivariate outliers.

**6.3 Decision rule**
- Do NOT blanket-remove outliers before checking their fraud rate — cross-tab outlier flag vs `is_fraud`. If outliers are disproportionately fraud, they're signal, not noise, and must stay in.

---

## 7. Temporal / Sequential Analysis

Fraud data has a time dimension that flat statistical tests can miss.

**7.1 Stationarity check**
- **Augmented Dickey-Fuller (ADF) test** (`statsmodels.tsa.stattools.adfuller`) on the daily/hourly fraud rate time series — tests whether fraud rate is stable over time or trending/seasonal, which affects whether a time-based train/test split is needed.

**7.2 Autocorrelation**
- ACF/PACF plots on transaction volume and fraud rate by hour/day-of-week to detect cyclical patterns (fraud often spikes at odd hours or specific days).

**7.3 Velocity/frequency checks**
- Transactions-per-card-per-hour distribution, compared fraud vs non-fraud via Mann-Whitney U (section 4.1) — card-testing fraud shows abnormal velocity.

---

## 8. Feature Importance / Relevance Pre-Screening

Not a formal statistical test, but standard practice before committing to a feature set:

- **ANOVA F-test** (`sklearn.feature_selection.f_classif`) — quick linear-relevance ranking of numeric features against the target.
- **Permutation importance from a quick baseline model** (e.g., shallow Random Forest) — sanity check that aligns with the univariate test rankings above; large disagreement signals interaction effects worth engineering.

---

## 9. Train/Test Distribution Consistency

Before modeling, confirm the split itself is valid:

- **Kolmogorov-Smirnov two-sample test** — compare the distribution of each key numeric feature between train and test/holdout sets. A significant KS statistic means your split isn't representative (common in fraud data if split naively by time and fraud patterns shift).
- **Population Stability Index (PSI)** — standard fraud/credit-risk industry metric for the same purpose; PSI > 0.25 signals meaningful drift between train and test populations.

---

## Summary Table

| Question | Test/Method | Data type |
|---|---|---|
| Is a feature normally distributed? | Shapiro-Wilk, D'Agostino-Pearson, KS | Numeric |
| Does a numeric feature differ by fraud class? | Mann-Whitney U, Welch's t-test, point-biserial r | Numeric vs binary |
| Does a categorical feature associate with fraud? | Chi-square, Cramér's V | Categorical vs binary |
| Are features linearly/monotonically correlated? | Pearson, Spearman | Numeric vs numeric |
| Is there multicollinearity? | VIF, correlation heatmap | Numeric |
| Is there non-linear dependence on target? | Mutual information | Any vs binary |
| Are there outliers, and are they fraud signal? | IQR, modified Z-score, Isolation Forest, LOF | Numeric |
| Is fraud rate stable over time? | ADF test, ACF/PACF | Time series |
| Is the train/test split representative? | KS two-sample test, PSI | Numeric, cross-set |

---

**Sequencing recommendation:** run sections 1 → 2 → 3 → 4/5 in parallel → 6 → 7 → 8 → 9. Sections 1–2 gate whether you proceed at all; 3 determines which flavor of tests to use in 4–5; 6–7 feed directly into feature engineering; 8–9 are the final gate before model selection/cross-validation.
