# Exploratory Data Analysis (EDA) in Python — End-to-End Guide

A practical, syntax-first reference for EDA on classification problems, covering both balanced and imbalanced target variables (e.g. churn, fraud).

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

```python
stat, p = stats.shapiro(df[col].dropna().sample(min(5000, len(df))))
print('Normal' if p > 0.05 else 'Not normal')
```

---

## 5. Univariate Analysis — Categorical

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

### Numerical feature vs categorical target

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

```python
contingency = pd.crosstab(df[cat_col], df[target])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print('Associated' if p < 0.05 else 'Not associated')
```

### Feature importance proxy — Cramér's V (categorical) and point-biserial (numeric)

```python
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
```

---

## 8. Multivariate Analysis

**Correlation heatmap:**

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

**IQR method:**

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

```python
z_scores = np.abs(stats.zscore(df[num_cols].dropna()))
outliers = (z_scores > 3).sum(axis=0)
```

**Isolation Forest (multivariate outliers — good pre-check for fraud EDA):**

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

Standard EDA can mislead on imbalanced targets because majority-class patterns dominate visuals and stats. Adjust as follows.

### 11.1 Always report class-conditional stats, not just marginal stats

```python
df.groupby(target)[num_cols].agg(['mean', 'median', 'std'])
```

### 11.2 Use normalized/overlaid plots, never raw counts, when comparing classes

```python
sns.histplot(data=df, x=col, hue=target, stat='density', common_norm=False, kde=True)
```

Raw-count histograms will make the minority class invisible — always normalize.

### 11.3 Stratified sampling for any exploratory sampling

```python
from sklearn.model_selection import train_test_split

sample, _ = train_test_split(df, train_size=0.1, stratify=df[target], random_state=42)
```

### 11.4 Rank features by class-separation power

**Numeric — AUC of single feature as a quick univariate signal:**

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

```python
early = df[df['date'] < df['date'].median()]
late = df[df['date'] >= df['date'].median()]

for col in num_cols[:5]:
    print(col, 'early AUC:', roc_auc_score(early[target], early[col].fillna(0)),
               'late AUC:', roc_auc_score(late[target], late[col].fillna(0)))
```

---

## 13. Feature Relationships & Leakage Checks

Critical before modeling — imbalanced problems are especially prone to leakage because a leaky feature can look like a "perfect" separator.

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
