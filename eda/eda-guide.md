# Exploratory Data Analysis (EDA) in Python — End-to-End Guide

A practical, syntax-rich reference for EDA on classification problems, covering both balanced and imbalanced target variables (e.g. churn, fraud). Each section pairs the "why" with the "how" — the theory behind a technique, followed by the code that implements it.

### Why EDA matters before modeling

EDA is the process of characterizing a dataset's structure, quality, and relationships *before* any modeling decision is made. Its purpose is threefold:

1. **Validate assumptions** — many models (linear/logistic regression, LDA) assume things about feature distributions, independence, or linearity. EDA tells you whether those assumptions hold, or whether a tree-based model like XGBoost (which makes no distributional assumptions) is the safer default.
2. **Surface data quality issues early** — missingness, duplicates, outliers, and leakage are far cheaper to fix at the EDA stage than after a model has already been trained on them.
3. **Build intuition for feature-target relationships** — understanding *which* features separate classes and *how* guides feature engineering, and gives you a sanity check against a model that "works" for the wrong reasons (e.g. leakage).

On imbalanced targets specifically, EDA carries extra weight: standard descriptive statistics and plots are computed across the whole dataset, so they are dominated by the majority class by construction. A histogram of 100,000 non-fraud transactions and 200 fraud transactions will visually erase the fraud cases entirely unless you deliberately condition on the class. Section 11 covers the adjustments needed.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Data Loading & First Look](#2-data-loading--first-look)
3. [Structural Checks](#3-structural-checks)
4. [Univariate Analysis — Numerical](#4-univariate-analysis--numerical)
5. [Univariate Analysis — Categorical](#5-univariate-analysis--categorical)
6. [Target Variable Analysis](#6-target-variable-analysis)
7. [Bivariate Analysis — Features vs Target](#7-bivariate-analysis--features-vs-target)
8. [Multivariate Analysis](#8-multivariate-analysis)
9. [Missing Values](#9-missing-values)
10. [Outlier Detection](#10-outlier-detection)
11. [Imbalanced Data — Special EDA Techniques](#11-imbalanced-data--special-eda-techniques)
12. [Time-Based EDA](#12-time-based-eda)
13. [Feature Relationships & Leakage Checks](#13-feature-relationships--leakage-checks)
14. [Automated EDA Tools](#14-automated-eda-tools)
15. [EDA Checklist](#15-eda-checklist)

---

## 1. Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style('whitegrid')
%matplotlib inline
```

**Core libraries by purpose:**

| Purpose | Library |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Static plots | `matplotlib`, `seaborn` |
| Interactive plots | `plotly` |
| Statistical tests | `scipy.stats`, `statsmodels` |
| Automated profiling | `ydata-profiling`, `sweetviz`, `dtale` |
| Imbalance-specific | `imbalanced-learn` |

---

## 2. Data Loading & First Look

```python
df = pd.read_csv('data.csv')

df.shape                 # (rows, columns)
df.head()
df.tail()
df.sample(5)
df.info()                # dtypes, non-null counts, memory
df.describe()             # numeric summary
df.describe(include='object')   # categorical summary
df.describe(include='all')
```

Quick column split by type:

```python
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
```

---

## 3. Structural Checks

```python
df.dtypes
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.nunique()                     # cardinality per column
df.columns[df.nunique() == 1]    # constant columns — drop candidates
df.columns[df.nunique() == len(df)]  # likely ID columns
```

Memory optimization (useful on large datasets):

```python
df.memory_usage(deep=True).sum() / 1e6   # MB

def downcast(df):
    for col in df.select_dtypes(include='int').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include='float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

---

## 4. Univariate Analysis — Numerical

Univariate analysis examines one variable in isolation — its shape, spread, and central tendency — before looking at relationships. This matters because:

- **Skewness** measures asymmetry. A skew near 0 is roughly symmetric; positive skew means a long right tail (common in monetary/transaction-amount features — most values are small with a few large outliers, typical in fraud data). Highly skewed features often benefit from a log or Box-Cox transform before use in distance-based or linear models.
- **Kurtosis** measures tail heaviness relative to a normal distribution. High kurtosis ("leptokurtic") means more extreme outliers than a normal distribution would predict — a signal to inspect the tails before assuming Gaussian-based methods (like Z-score outlier detection) apply cleanly.

```python
for col in num_cols:
    print(col, df[col].skew(), df[col].kurt())
```

**Distribution plot:**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[col], kde=True, ax=axes[0])
sns.boxplot(x=df[col], ax=axes[1])
plt.suptitle(col)
plt.show()
```

**Batch plotting all numeric columns:**

```python
df[num_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()
```

**Normality test:**

The Shapiro-Wilk test formally checks whether a sample plausibly came from a normal distribution. The null hypothesis is "the data is normal"; a p-value below 0.05 rejects that, meaning the data significantly deviates from normality. This is capped at a 5,000-row sample because the test grows extremely sensitive at large `n` — with millions of rows it will reject normality for almost any real-world feature, even mild deviations, so treat it as a directional signal rather than a hard rule.

```python
stat, p = stats.shapiro(df[col].dropna().sample(min(5000, len(df))))
print('Normal' if p > 0.05 else 'Not normal')
```

---

## 5. Univariate Analysis — Categorical

Categorical features are summarized by frequency rather than moments (mean/variance don't apply to unordered labels). The key concept here is **cardinality** — the number of distinct levels a category has. Low-cardinality features (e.g. `gender`, `subscription_tier`) are cheap to one-hot encode and visualize directly. High-cardinality features (e.g. `zip_code`, `merchant_id`) blow up dimensionality under one-hot encoding and are usually better handled with frequency or target encoding — but target encoding on a high-cardinality feature also raises leakage risk if not done inside cross-validation folds.

```python
for col in cat_cols:
    print(df[col].value_counts(normalize=True).head(10))
    print('-' * 40)
```

```python
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(y=df[col], order=df[col].value_counts().index[:15], ax=ax)
plt.title(col)
plt.show()
```

Flag high-cardinality categoricals (candidates for target/frequency encoding):

```python
cardinality = df[cat_cols].nunique().sort_values(ascending=False)
high_card = cardinality[cardinality > 50]
```

---

## 6. Target Variable Analysis

This is the single most important step in the entire EDA workflow for a classification problem, because the target distribution determines almost every downstream decision: which evaluation metric is meaningful, whether resampling is needed, and how much you can trust visual intuition from later plots.

**Why accuracy fails on imbalanced targets:** if 99% of customers don't churn, a model that always predicts "no churn" scores 99% accuracy while being useless. This is why imbalance ratio is computed first — it flags when accuracy must be abandoned in favor of metrics like PR-AUC, recall, or F1 that are sensitive to minority-class performance.

**Why PR-AUC over ROC-AUC for imbalanced data:** ROC-AUC plots true positive rate against false positive rate, and the false positive rate denominator (true negatives) is huge when the majority class dominates — so ROC-AUC can look deceptively good even when precision on the minority class is poor. PR-AUC (precision-recall AUC) doesn't reference true negatives at all, making it far more diagnostic when the positive class (churn/fraud) is rare.

```python
target = 'churn'   # or 'is_fraud'

df[target].value_counts()
df[target].value_counts(normalize=True) * 100
```

```python
sns.countplot(x=df[target])
plt.title('Class Distribution')
for i, v in enumerate(df[target].value_counts()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()
```

**Imbalance ratio:**

```python
counts = df[target].value_counts()
imbalance_ratio = counts.max() / counts.min()
print(f'Imbalance ratio: {imbalance_ratio:.1f} : 1')
```

Rule of thumb:
- `< 3:1` → mild, standard metrics/models are fine
- `3:1 – 20:1` → moderate, use PR-AUC, class weights, resampling
- `> 20:1` → severe, needs specialized techniques (Section 11)

---

## 7. Bivariate Analysis — Features vs Target

Bivariate analysis asks: does this feature actually carry information about the target? This is where visual inspection is paired with formal statistical tests, because a difference that "looks" real in a plot can arise from sampling noise, especially when one class is small.

### Numerical feature vs categorical target

A boxplot compares the median and spread of a numeric feature across target classes at a glance — non-overlapping boxes suggest the feature separates classes well. A KDE (kernel density estimate) plot shows the full distribution shape rather than just quartiles, which matters when the separation is in the tails rather than the center (common in fraud amounts).

```python
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f'{col} vs {target}')
    plt.show()
```

```python
for col in num_cols:
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(col)
    plt.show()
```

**Statistical test (mean difference across classes):**

Welch's t-test (`equal_var=False`) checks whether the mean of a numeric feature genuinely differs between the two target classes, rather than the difference being due to chance. Welch's variant is preferred over the standard t-test because it does not assume equal variance between groups — a reasonable default here, since the minority class in an imbalanced problem often does have different variance from the majority class. A small p-value (< 0.05) indicates the feature's mean is significantly different across classes; a large p-value does not rule out a relationship, only a *linear/mean-based* one — the KDE plot above catches distributional differences that the t-test misses (e.g. a feature that separates classes via variance rather than mean).

```python
group0 = df[df[target] == 0][col].dropna()
group1 = df[df[target] == 1][col].dropna()
t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
```

### Categorical feature vs categorical target

```python
pd.crosstab(df[cat_col], df[target], normalize='index')
```

```python
ct = pd.crosstab(df[cat_col], df[target], normalize='index')
ct.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title(f'{cat_col} vs {target}')
plt.show()
```

**Chi-square test of independence:**

This test measures whether the distribution of a categorical feature's levels differs across target classes more than chance would predict. It works by comparing observed cell counts in the contingency table against the counts you'd expect if the feature and target were independent; a large discrepancy (and resulting small p-value) means the feature's category is informative about the target. It requires reasonably sized cell counts (rule of thumb: expected count ≥ 5 per cell) to be reliable — sparse categories should be grouped into an "other" bucket first.

```python
contingency = pd.crosstab(df[cat_col], df[target])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print('Associated' if p < 0.05 else 'Not associated')
```

### Feature importance proxy — Cramér's V (categorical) and point-biserial (numeric)

Chi-square tells you *whether* an association is statistically significant, but not *how strong* it is — with enough rows, even a trivially weak association becomes "significant." Cramér's V normalizes the chi-square statistic into a 0–1 scale (0 = no association, 1 = perfect association), making it comparable across features regardless of sample size or number of categories. This is the categorical analogue of a correlation coefficient.

```python
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
```

---

## 8. Multivariate Analysis

Bivariate analysis looks at one feature against the target; multivariate analysis looks at how features relate to *each other*, which matters for two reasons — redundant features add noise without adding signal, and highly correlated features can destabilize coefficient estimates in linear models (though tree-based models like XGBoost are comparatively robust to this).

**Correlation heatmap:**

Pearson correlation measures linear association between two numeric features, ranging from -1 (perfect inverse) to +1 (perfect direct relationship). It only captures *linear* relationships — two features can be strongly related in a nonlinear way (e.g. U-shaped) and still show near-zero Pearson correlation, so a flat heatmap doesn't guarantee independence.

```python
plt.figure(figsize=(10, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.show()
```

**Correlation with target only (numeric target-encoded):**

```python
df.corr(numeric_only=True)[target].sort_values(ascending=False)
```

**Pairplot (use a sample — expensive on large data):**

```python
sns.pairplot(df.sample(1000), hue=target, vars=num_cols[:5], corner=True)
```

**Multicollinearity check (VIF):**

Variance Inflation Factor quantifies how much a feature's variance is "inflated" because it can be linearly predicted from the other features — in other words, how redundant it is. A VIF of 1 means no correlation with other features; a VIF of 10 means the feature's variance is 10x what it would be if uncorrelated, implying roughly 90% of its variability is explained by other features already in the dataset. High VIF mainly matters for coefficient interpretability and stability in linear/logistic regression; it is far less of a concern for tree-based models like XGBoost, which split on features independently and handle redundancy natively.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[num_cols].dropna()
vif = pd.DataFrame()
vif['feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif.sort_values('VIF', ascending=False)
```

VIF > 10 → strong multicollinearity, consider dropping/combining features.

---

## 9. Missing Values

Missing data matters because *why* a value is missing changes how you should handle it. The standard taxonomy:

- **MCAR (Missing Completely At Random)** — missingness is unrelated to any variable, observed or not. Safe to drop or impute with simple statistics.
- **MAR (Missing At Random)** — missingness depends on *other observed* variables (e.g. income is missing more often for a certain age group). Imputation conditioned on those variables works well.
- **MNAR (Missing Not At Random)** — missingness depends on the *unobserved value itself* (e.g. high-fraud-risk transactions are more likely to have an incomplete merchant profile because fraudsters obscure details). This is the most dangerous case for churn/fraud problems, because the missingness pattern is itself predictive — which is why Section 9's "missingness vs target" check exists below rather than just measuring missingness in isolation.

```python
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False)
```

**Visual pattern:**

```python
import missingno as msno
msno.matrix(df)
msno.heatmap(df)     # correlation of missingness between columns
msno.bar(df)
```

**Is missingness related to the target? (important for fraud/churn — missingness can itself be signal)**

```python
for col in missing_df[missing_df['count'] > 0].index:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    print(col, pd.crosstab(df[f'{col}_missing'], df[target], normalize='index'))
```

---

## 10. Outlier Detection

An outlier is a value that deviates markedly from the rest of the data. Three different methods are used together here because each has a different notion of "far":

**IQR method:**

The IQR (interquartile range) method flags any point outside 1.5x the range between the 25th and 75th percentiles. It's distribution-free — it makes no assumption about normality, which makes it robust for skewed features like transaction amounts. It's a univariate method: it looks at one column at a time and cannot detect a point that is unusual only in combination with another feature.

```python
def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]

for col in num_cols:
    n_out = len(iqr_outliers(df[col]))
    print(col, n_out, f'{n_out/len(df)*100:.2f}%')
```

**Z-score method:**

The Z-score expresses how many standard deviations a point is from the mean; a common threshold of |Z| > 3 flags extreme values. Unlike IQR, this method assumes the feature is roughly normally distributed — on skewed features it can under- or over-flag outliers, so it's best used alongside (not instead of) IQR.

```python
z_scores = np.abs(stats.zscore(df[num_cols].dropna()))
outliers = (z_scores > 3).sum(axis=0)
```

**Isolation Forest (multivariate outliers — good pre-check for fraud EDA):**

IQR and Z-score are univariate — they can miss a point that is unremarkable in every individual feature but unusual in *combination* (e.g. a transaction with a normal amount at a normal hour, but an abnormal amount-for-that-hour combination). Isolation Forest works by randomly partitioning the feature space; anomalies require fewer random splits to isolate than normal points do, because they sit in sparse regions of the data. This makes it a genuinely multivariate outlier detector and a natural fit for fraud-style problems where individual features look unremarkable in isolation.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
df['outlier_flag'] = iso.fit_predict(df[num_cols].fillna(0))
df['outlier_flag'].value_counts()
```

**Outliers vs target — do outliers correlate with the positive class?**

```python
pd.crosstab(df['outlier_flag'], df[target], normalize='index')
```

> In fraud/churn contexts, don't blindly remove outliers — they are frequently the signal, not noise. Always check outlier rate by class before treating them.

---

## 11. Imbalanced Data — Special EDA Techniques

Standard EDA can mislead on imbalanced targets because majority-class patterns dominate visuals and stats. Every plot and statistic computed over the whole dataset is, by construction, a weighted average dominated by whichever class has more rows — so a "typical" value or shape reported without conditioning on the target is really just describing the majority class. Adjust as follows.

### 11.1 Always report class-conditional stats, not just marginal stats

A **marginal** statistic (e.g. `df[col].mean()`) collapses across both classes. A **class-conditional** statistic (`df.groupby(target)[col].mean()`) reports the value separately per class. On a severely imbalanced target, the marginal mean is nearly identical to the majority-class mean, and any minority-class signal is invisible unless you condition on the target explicitly — which is why every summary in this section is computed per group rather than once overall.

```python
df.groupby(target)[num_cols].agg(['mean', 'median', 'std'])
```

### 11.2 Use normalized/overlaid plots, never raw counts, when comparing classes

`stat='density'` rescales each class's histogram so its bars integrate to 1, and `common_norm=False` ensures each class is normalized *independently* rather than relative to the combined dataset. Without both settings, the minority class's bars stay tiny relative to the majority class's simply because there are fewer of them — a visualization artifact of class size, not of the actual shape difference you're trying to see.

```python
sns.histplot(data=df, x=col, hue=target, stat='density', common_norm=False, kde=True)
```

Raw-count histograms will make the minority class invisible — always normalize.

### 11.3 Stratified sampling for any exploratory sampling

A plain random sample (`df.sample()`) preserves the overall class ratio *in expectation*, but on a rare positive class a 10% random sample can easily end up with too few minority examples to draw any reliable conclusion from — or in the worst case, none at all. Stratified sampling explicitly preserves the exact class proportion in the sample by sampling within each class group separately, guaranteeing minority representation regardless of sample size.

```python
from sklearn.model_selection import train_test_split

sample, _ = train_test_split(df, train_size=0.1, stratify=df[target], random_state=42)
```

### 11.4 Rank features by class-separation power

**Numeric — AUC of single feature as a quick univariate signal:**

Passing a raw feature directly into `roc_auc_score` (instead of model predictions) treats the feature's own value as if it were a ranking score, and measures how well that ranking alone separates the two classes. An AUC of 0.5 means the feature has no ranking power — knowing its value tells you nothing about class membership. An AUC far from 0.5 (in either direction) means the feature alone is a decent classifier, which is a fast way to pre-screen hundreds of features before formal modeling.

```python
from sklearn.metrics import roc_auc_score

scores = {}
for col in num_cols:
    try:
        scores[col] = roc_auc_score(df[target], df[col].fillna(df[col].median()))
    except ValueError:
        pass
pd.Series(scores).sort_values(ascending=False)
```

AUC near 0.5 → weak standalone signal. AUC near 0 or 1 → strong signal (or leakage — verify).

### 11.5 Minority-class deep dive

```python
minority = df[df[target] == 1]
majority = df[df[target] == 0]

minority[num_cols].describe()
majority[num_cols].describe()
```

Look specifically for:
- Features where minority-class distribution is narrow/tight (easy separators)
- Features where minority class has more missingness or more extreme outliers
- Categorical levels that appear disproportionately in the minority class

```python
for col in cat_cols:
    minority_rate = df.groupby(col)[target].mean().sort_values(ascending=False)
    print(minority_rate.head(10))
```

### 11.6 Visualize class overlap in reduced dimensions

With more than 2–3 features, you can't directly plot the class separation in the original feature space. Dimensionality reduction projects the data into 2D while trying to preserve structure, so you can visually judge whether the classes occupy distinguishable regions.

- **PCA** finds the directions (linear combinations of features) that capture the most variance in the data. It's fast and deterministic, but since it optimizes for variance rather than class separation, two classes can look overlapped in a PCA plot even if they're actually well-separated by a nonlinear boundary.
- **t-SNE** is a nonlinear technique that preserves local neighborhood structure — points close together in the original space stay close in the 2D projection. It generally shows class clusters more clearly than PCA but is computationally expensive (hence the sampling below) and its axes have no direct interpretable meaning — only relative distances/clusters matter, not absolute position.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(df[num_cols].fillna(0))
pca = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(pca[:, 0], pca[:, 1], c=df[target], cmap='coolwarm', alpha=0.4, s=10)
plt.title('PCA — class separability')
plt.show()
```

For a sharper (nonlinear) view on a sampled subset:

```python
from sklearn.manifold import TSNE

sample = df.sample(5000, random_state=42)
X_s = StandardScaler().fit_transform(sample[num_cols].fillna(0))
tsne = TSNE(n_components=2, random_state=42).fit_transform(X_s)

plt.scatter(tsne[:, 0], tsne[:, 1], c=sample[target], cmap='coolwarm', alpha=0.5, s=10)
plt.title('t-SNE — class separability (sampled)')
plt.show()
```

### 11.7 Don't compute global correlation heatmaps only — compute them per class

Correlation structure often differs between fraud/non-fraud or churned/retained populations. A single global heatmap masks this.

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(majority[num_cols].corr(), ax=axes[0], cmap='coolwarm', center=0)
axes[0].set_title('Majority class correlation')
sns.heatmap(minority[num_cols].corr(), ax=axes[1], cmap='coolwarm', center=0)
axes[1].set_title('Minority class correlation')
plt.show()
```

### 11.8 Check for near-duplicate / synthetic-looking minority records

Especially relevant for fraud data, where minority classes are sometimes clustered (e.g. one fraud ring generating many similar transactions).

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=2).fit(minority[num_cols].fillna(0))
dist, _ = nn.kneighbors(minority[num_cols].fillna(0))
plt.hist(dist[:, 1], bins=50)
plt.title('Distance to nearest neighbor within minority class')
```

---

## 12. Time-Based EDA

Relevant when churn/fraud has a temporal dimension (transaction timestamps, subscription dates).

```python
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
```

**Target rate over time:**

```python
daily_rate = df.groupby(df['date'].dt.date)[target].mean()
daily_rate.plot(figsize=(14, 4), title='Daily target rate')
```

**Seasonality / hour-of-day patterns (common in fraud):**

```python
hourly = df.groupby('hour')[target].mean()
hourly.plot(kind='bar', title='Target rate by hour')
```

**Check for concept drift — is the relationship between features and target stable over time?**

Concept drift is when the statistical relationship between features and the target changes over time — e.g. a fraud pattern that worked last year gets patched and stops being predictive. It matters because a model validated on historical data can silently degrade in production if the feature-target relationship has shifted. Comparing single-feature AUC between an early and late time window is a lightweight way to catch this at the EDA stage, before it shows up as unexplained live performance decay later.

```python
early = df[df['date'] < df['date'].median()]
late = df[df['date'] >= df['date'].median()]

for col in num_cols[:5]:
    print(col, 'early AUC:', roc_auc_score(early[target], early[col].fillna(0)),
               'late AUC:', roc_auc_score(late[target], late[col].fillna(0)))
```

---

## 13. Feature Relationships & Leakage Checks

**Data leakage** occurs when a feature contains information that would not actually be available at prediction time — most often because it is a byproduct of the outcome itself (e.g. a "days since cancellation" field that only exists for customers who already churned). Leakage is especially dangerous on imbalanced targets because a leaky feature tends to produce a *near-perfect* single-feature AUC, which can look like an exciting discovery rather than the red flag it actually is. Any feature that separates classes suspiciously well deserves scrutiny before being trusted.

```python
# Any feature with near-1.0 single-feature AUC deserves scrutiny
suspicious = pd.Series(scores).sort_values(ascending=False)
suspicious[suspicious > 0.95]
```

```python
# Check if a feature is only populated after the outcome is known
# e.g. 'cancellation_reason' only exists for churned customers
df.groupby(target)[col].apply(lambda x: x.notnull().mean())
```

```python
# Check for ID-like or post-outcome timestamp columns
df.columns[df.columns.str.contains('id|date_closed|resolution', case=False)]
```

---

## 14. Automated EDA Tools

Useful for a fast first pass — not a substitute for the targeted class-conditional analysis above.

```python
# ydata-profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title='EDA Report', explorative=True)
profile.to_file('eda_report.html')
```

```python
# Sweetviz — supports direct target-based comparison
import sweetviz as sv
report = sv.analyze(df, target_feat=target)
report.show_html('sweetviz_report.html')
```

```python
# D-Tale — interactive browser-based EDA
import dtale
dtale.show(df).open_browser()
```

---

## 15. EDA Checklist

**Structure**
- [ ] Shape, dtypes, duplicates checked
- [ ] Constant / ID-like columns flagged
- [ ] Memory usage reasonable for dataset size

**Univariate**
- [ ] Distributions plotted for all numeric columns
- [ ] Skew/kurtosis noted; transform candidates identified
- [ ] Categorical cardinality checked; high-cardinality columns flagged

**Target**
- [ ] Class distribution and imbalance ratio computed
- [ ] Imbalance severity classified (mild/moderate/severe)

**Bivariate / class-conditional**
- [ ] Features compared using normalized plots (density, not raw counts)
- [ ] Statistical tests run (t-test / chi-square) where relevant
- [ ] Single-feature AUC ranking computed
- [ ] Minority-class-specific deep dive done

**Multivariate**
- [ ] Global correlation heatmap done
- [ ] Per-class correlation heatmaps compared
- [ ] Multicollinearity (VIF) checked
- [ ] PCA/t-SNE class-separability plot done

**Data quality**
- [ ] Missingness quantified and visualized
- [ ] Missingness-vs-target relationship checked
- [ ] Outliers detected (IQR/Z-score/Isolation Forest)
- [ ] Outlier rate compared by class

**Time (if applicable)**
- [ ] Target rate over time plotted
- [ ] Seasonality/time-of-day patterns checked
- [ ] Concept drift (early vs late AUC) checked

**Integrity**
- [ ] Leakage check — no suspiciously high single-feature AUC left unexplained
- [ ] No post-outcome features present

---

*Companion references: cross-validation guide (model selection, imbalance handling in CV), MLflow guide (experiment tracking), ML lifecycle guide (end-to-end workflow).*
