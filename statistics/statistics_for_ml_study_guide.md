# Statistics for Machine Learning — Study Guide

A consolidated reference covering distribution transformations, central tendency & imputation, the Central Limit Theorem, Chebyshev's Inequality, correlation measures, and hypothesis testing.

---

## 1. Converting a Log-Normal Distribution to a Standard Normal Distribution

### The Concept

If a random variable `X` follows a **log-normal distribution** (common for skewed data like income, fraud transaction amounts, or website session durations), its natural logarithm `Y = ln(X)` follows a **normal distribution**.

This means you cannot z-score `X` directly using its own mean/std — you must log-transform first, *then* standardize.

### The Formula

```
Z = (ln(X) - μ) / σ
```

| Symbol | Name | Meaning |
|---|---|---|
| `X` | Original variable | The raw log-normally distributed data |
| `ln(X)` | Natural logarithm | Transforms `X` into a normally distributed variable `Y` |
| `μ` (mu) | Mean | Mean of the **log-transformed** data `ln(X)`, not raw `X` |
| `σ` (sigma) | Standard deviation | Std dev of the **log-transformed** data `ln(X)` |
| `Z` | Z-score / standard score | Standardized value; result is `N(0, 1)` — mean 0, std 1 |

**Common mistake:** computing `μ` and `σ` from the raw `X` values instead of `ln(X)`. This silently breaks the transformation because raw log-normal data is skewed — its mean and std don't describe a symmetric bell curve.

### Python Implementation

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Example raw log-normal data (e.g., transaction amounts)
X = np.array([[1.0], [2.5], [5.0], [10.0]])

# Step 1: Take the natural log (transforms it to a normal distribution)
X_log = np.log(X)

# Step 2: Apply StandardScaler to the LOG values, not the raw values
scaler = StandardScaler()
Z = scaler.fit_transform(X_log)

print(Z)
```

**Pipeline-friendly version** using `FunctionTransformer` (chains cleanly into `sklearn.Pipeline`):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

log_standardize_pipeline = Pipeline([
    ("log", FunctionTransformer(np.log, validate=True)),
    ("scale", StandardScaler())
])

Z = log_standardize_pipeline.fit_transform(X)
```

### Why This Matters in ML
Many models (linear/logistic regression, PCA, k-NN, anything relying on Euclidean distance or Gaussian assumptions) perform better when skewed features are log-transformed and standardized first. Skipping the log step and standardizing raw skewed data leaves outliers dominating the scale.

---

## 2. Central Tendency & Outlier-Robust Imputation

When filling missing values, the right statistic depends on how sensitive it is to outliers.

### Mean, Median, Mode — Behavior Summary

| Metric | Impact of Outliers | Best Used When |
|---|---|---|
| **Mean** | High — gets dragged toward the outlier | Data is symmetric and normally distributed (no major outliers) |
| **Median** | Low — resistant to extreme values | Data is skewed, contains outliers, or is ordinal/continuous |
| **Mode** | None — ignores magnitude, focuses on frequency | Categorical data or discrete data with heavy repetition |

### Why the Median Resists Outliers
The median is the exact middle value once data is sorted — it only cares about **position**, not magnitude.

Example: `[1, 2, 3, 4, 100]`
- Mean = 22 (dragged toward 100)
- Median = 3 (completely unaffected by the outlier's size)

If `100` were a typo, the median still gives a value representative of the typical data points — making it the safer default for imputation on skewed features.

### Why the Mode Resists Outliers
The mode only counts **frequency**, never magnitude. It's the standard choice for categorical variables (e.g., filling a missing "payment_method" with the most common category) and can also apply to discrete numeric data with heavy repetition.

### Python Implementation

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "transaction_amount": [10, 12, 11, 9, 5000, np.nan, 13],  # numeric, has outlier
    "payment_method": ["card", "card", "upi", np.nan, "card", "upi", "card"]  # categorical
})

# Median imputation for skewed/outlier-prone numeric columns
median_val = df["transaction_amount"].median()
df["transaction_amount"] = df["transaction_amount"].fillna(median_val)

# Mode imputation for categorical columns
mode_val = df["payment_method"].mode()[0]
df["payment_method"] = df["payment_method"].fillna(mode_val)

print(df)
```

Using `sklearn.impute.SimpleImputer` in a pipeline:

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")  # mode

df[["transaction_amount"]] = num_imputer.fit_transform(df[["transaction_amount"]])
df[["payment_method"]] = cat_imputer.fit_transform(df[["payment_method"]])
```

**Rule of thumb for your fraud detection work:** transaction amounts, account age, and similar heavy-tailed features → median. Categorical fields like merchant category or device type → mode.

---

## 3. The Central Limit Theorem (CLT)

### Core Idea
> The CLT lets you use normal-distribution statistics (z-tests, t-tests, confidence intervals) even when your original data is completely non-normal (skewed, bimodal, etc.) — **as long as you're looking at the distribution of sample means**, not the raw data.

If you draw many random samples (`S1, S2, ..., S100`) of size `n` from *any* population, the sample means (`x̄1, x̄2, ...`) will themselves be approximately normally distributed around the true population mean `μ`, regardless of the shape of the original population.

### Why It Matters
1. **Breaks the "normal data" requirement** — real-world data (income, website traffic, fraud amounts) is rarely normal, but the *sampling distribution of the mean* still behaves normally.
2. **Bridges samples and populations** — lets you make inferences about `μ` from a sample.
3. **Powers hypothesis testing** — p-values, z-tests, t-tests, and confidence intervals all rely on the sampling distribution being well-behaved (bell-shaped).

### The Conditions (n ≥ 30 Is Not Enough on Its Own)

The classic "n ≥ 30" rule of thumb only holds if these are also true:

| Condition | Rule | Why It Matters |
|---|---|---|
| **Independence** | Each observation must be sampled independently | If one data point influences another (time-series, tight-knit groups), the math breaks down. If sampling without replacement, sample size should be < 10% of the population |
| **Random sampling** | Sample must be chosen randomly | Biased selection means sample means won't center on the true `μ`, no matter how large `n` is |
| **Shape of the underlying population** | `n ≥ 30` assumes the data isn't *infinitely* messy | Heavily/violently skewed data (extreme Pareto, severe log-normal) may need `n = 50, 100,` or more. If the original population is already normal, CLT applies instantly even for small `n` |

### Python Implementation — Simulating the CLT

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 1. Generate a heavily skewed population (e.g., simulating fraud amounts)
population = rng.exponential(scale=50, size=100_000)  # exponential = skewed, non-normal

# 2. Draw many random samples and record their means
sample_means = []
n = 40          # sample size
num_samples = 2000

for _ in range(num_samples):
    sample = rng.choice(population, size=n, replace=False)
    sample_means.append(sample.mean())

# 3. Plot: raw population (skewed) vs. distribution of sample means (normal-ish)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(population, bins=50, color="steelblue")
axes[0].set_title("Original Population (Skewed)")

axes[1].hist(sample_means, bins=50, color="salmon")
axes[1].set_title(f"Distribution of Sample Means (n={n})")
plt.tight_layout()
plt.show()

print("Population mean:", population.mean())
print("Mean of sample means:", np.mean(sample_means))
```

Running this shows the left histogram heavily right-skewed, while the right histogram (sample means) looks approximately bell-shaped and centers on the true population mean — the CLT in action.

---

## 4. Chebyshev's Inequality

### When to Use It
The empirical rule (68–95–99.7%) **only applies to normal distributions**. When your data is skewed or you don't know its shape (fraud amounts, website traffic), Chebyshev's Inequality gives a distribution-free guarantee.

### The Formula

```
Pr(μ - kσ < x < μ + kσ) ≥ 1 - 1/k²
```

| Symbol | Name | Meaning |
|---|---|---|
| `k` | Number of standard deviations | Any number greater than 1 |
| `1 - 1/k²` | Minimum guaranteed proportion | The floor on how much data must fall within `±kσ` of the mean, for **any** distribution shape |

### Worked Examples

**k = 2 (within ±2σ):**
```
1 - 1/2² = 1 - 1/4 = 3/4 = 75%
```
At least 75% of data falls within 2 std devs of the mean — no matter how skewed. (Compare to ~95% for a true normal distribution — Chebyshev is a conservative worst-case guarantee.)

**k = 3 (within ±3σ):**
```
1 - 1/3² = 1 - 1/9 = 8/9 ≈ 88.9%
```
At least ~89% of data is guaranteed within 3 std devs, regardless of skew.

### Python Implementation

```python
import numpy as np

def chebyshev_bound(k):
    """Minimum guaranteed proportion of data within k standard deviations."""
    if k <= 1:
        raise ValueError("Chebyshev's Inequality requires k > 1")
    return 1 - 1 / k**2

for k in [2, 3, 4]:
    print(f"k={k}: at least {chebyshev_bound(k):.1%} of data within ±{k}σ")

# Applying it to a real, skewed dataset
rng = np.random.default_rng(0)
fraud_amounts = rng.exponential(scale=200, size=10_000)  # skewed

mu = fraud_amounts.mean()
sigma = fraud_amounts.std()

k = 3
lower, upper = mu - k * sigma, mu + k * sigma
within_range = np.mean((fraud_amounts > lower) & (fraud_amounts < upper))

print(f"Empirically, {within_range:.1%} of data falls within ±{k}σ")
print(f"Chebyshev guarantees at least {chebyshev_bound(k):.1%}")
```

Because Chebyshev is a *worst-case* bound, the empirical percentage on real data will typically be much higher than the guarantee — but the guarantee never fails, unlike the empirical rule on non-normal data.

---

## 5. Covariance vs. Correlation

Both measure how two variables move together, but they answer different questions.

### Covariance — Direction Only

```
Cov(X, Y) = (1/n) * Σ (xᵢ - μx)(yᵢ - μy)
```

| Symbol | Name | Meaning |
|---|---|---|
| `xᵢ, yᵢ` | Individual data points | Observations for variables X and Y |
| `μx, μy` | Means | Average of X and Y respectively |
| `n` | Sample size | Number of observations |

**Reading it:**
- **Positive:** both variables move together (e.g., size ↑, price ↑)
- **Negative:** variables move oppositely (e.g., fuel efficiency ↑, fuel consumption ↓)
- **Zero:** no linear relationship

**The Big Flaw:** covariance is **unbounded and scale-dependent**. Measuring house prices in dollars vs. thousands of dollars changes the covariance value drastically — you can't tell if `Cov = 500` is a strong or weak relationship just by looking at it.

### Correlation (Pearson's `r`) — Direction *and* Strength

```
r = Cov(X, Y) / (σx * σy)
```

Correlation standardizes covariance by dividing by both standard deviations, producing a value strictly bounded between **-1 and +1**.

| Value | Meaning |
|---|---|
| `+1` | Perfect positive linear relationship |
| `-1` | Perfect negative linear relationship |
| `0` | No linear relationship |

### Key Differences at a Glance

| Feature | Covariance | Correlation |
|---|---|---|
| **Scale** | Unbounded (-∞ to +∞) | Bounded (-1 to +1) |
| **Units** | Dependent on units of X and Y (e.g., dollars × sq. ft.) | Unitless (pure number) |
| **Interpretability** | Hard to judge magnitude (is 1,500 big or small?) | Easy — e.g., 0.85 is a strong positive relationship |
| **Use case** | Intermediate step for other metrics (e.g., variance-covariance matrices) | Widely used for feature selection, EDA, model diagnostics |

### Python Implementation

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "house_size_sqft": [1200, 1500, 1800, 2200, 2800],
    "price_thousands": [150, 190, 210, 270, 320]
})

# Covariance (matrix — diagonal = variances, off-diagonal = covariance)
cov_matrix = df.cov()
print("Covariance matrix:\n", cov_matrix)

# Pearson correlation
corr_matrix = df.corr(method="pearson")
print("\nPearson correlation matrix:\n", corr_matrix)

# Manually, matching the formula
cov_xy = np.cov(df["house_size_sqft"], df["price_thousands"], ddof=0)[0, 1]
r = cov_xy / (df["house_size_sqft"].std(ddof=0) * df["price_thousands"].std(ddof=0))
print(f"\nManual r = {r:.4f}")
```

---

## 6. Spearman Rank Correlation

### Core Idea
Pearson's `r` measures **linear** relationships. Spearman's `ρ` (or `rs`) measures **monotonic** relationships — where one variable consistently increases or decreases as the other does, but not necessarily at a constant rate (e.g., an exponential curve is monotonic but not linear).

Instead of using raw values, Spearman converts each variable to **ranks**, then computes Pearson's correlation on those ranks. Because ranks compress outliers to "just the highest/lowest position," Spearman is much less sensitive to extreme values than Pearson.

### The Formula (no tied ranks)

```
rs = 1 - (6 * Σ dᵢ²) / (n * (n² - 1))
```

| Symbol | Name | Meaning |
|---|---|---|
| `dᵢ` | Rank difference | `rank(xᵢ) - rank(yᵢ)` for each observation |
| `dᵢ²` | Squared rank difference | Squaring removes sign, penalizes larger gaps more |
| `n` | Number of data points | Sample size |
| `rs` (or `ρ`) | Spearman's coefficient | Ranges strictly between -1 and +1 |

### Step-by-Step Process
1. Sort by `X`, assign ranks `1, 2, ..., n` → column `xᵢ`
2. Sort by `Y`, assign ranks `1, 2, ..., n` → column `yᵢ`
3. Compute `dᵢ = xᵢ - yᵢ` for each row
4. Compute `dᵢ²`
5. Plug `Σdᵢ²` into the formula

**Worked example** (IQ vs. Hours of TV per week):

| IQ | Hours TV | rank xᵢ | rank yᵢ | dᵢ | dᵢ² |
|---|---|---|---|---|---|
| 86 | 0 | 1 | 1 | 0 | 0 |
| 97 | 20 | 2 | 6 | -4 | 16 |
| 99 | 28 | 3 | 8 | -5 | 25 |
| 100 | 27 | 4 | 7 | -3 | 9 |
| 101 | 50 | 5 | 10 | -5 | 25 |
| 103 | 29 | 6 | 9 | -3 | 9 |
| 106 | 7 | 7 | 3 | 4 | 16 |
| 110 | 17 | 8 | 5 | 3 | 9 |
| 112 | 6 | 9 | 2 | 7 | 49 |
| 113 | 12 | 10 | 4 | 6 | 36 |

`Σdᵢ² = 194`, `n = 10` → `rs = 1 - (6 × 194) / (10 × 99) = 1 - 1164/990 ≈ -0.176`

A weak negative monotonic relationship — as IQ rises, TV hours tend to fall slightly.

### Reading the Output

| Value | Meaning |
|---|---|
| `+1` | Perfect monotonic increasing relationship |
| `-1` | Perfect monotonic decreasing relationship |
| `0` | No monotonic relationship |

### Pearson vs. Spearman — When to Use Which

| Feature | Pearson | Spearman |
|---|---|---|
| **Data type** | Continuous, normally distributed | Ordinal, or continuous with outliers/skew |
| **Relationship type** | Strictly linear | Monotonic (can be curved) |
| **Sensitivity to outliers** | High | Low (ranks flatten extreme values) |

### Python Implementation

```python
import pandas as pd
from scipy import stats

data = pd.DataFrame({
    "IQ": [86, 97, 99, 100, 101, 103, 106, 110, 112, 113],
    "Hours_TV": [0, 20, 28, 27, 50, 29, 7, 17, 6, 12]
})

# Using scipy directly
rho, p_value = stats.spearmanr(data["IQ"], data["Hours_TV"])
print(f"Spearman's rho: {rho:.4f}, p-value: {p_value:.4f}")

# Using pandas
print("Spearman via pandas:", data.corr(method="spearman").iloc[0, 1])

# Manual implementation matching the formula, for intuition
ranks_iq = data["IQ"].rank()
ranks_tv = data["Hours_TV"].rank()
d = ranks_iq - ranks_tv
n = len(data)
rs_manual = 1 - (6 * (d**2).sum()) / (n * (n**2 - 1))
print(f"Manual rs: {rs_manual:.4f}")
```

**When to reach for this in fraud/churn work:** feature relationships with heavy-tailed variables (transaction amount, tenure, account age) where a few extreme values would distort Pearson's `r`.

---

## 7. Hypothesis Testing Fundamentals (from the whiteboard notes)

### The Coin-Flip Example
Setup: flip a coin 100 times. If it's fair, you'd expect ~50 heads. You observe 60 heads. Is the coin still fair, or is 60 heads evidence it's biased?

### Key Terms

| Term | Symbol | Meaning |
|---|---|---|
| **Null Hypothesis** | `H₀` | The default, "boring" claim — treats everything as the same/equal. Here: *the coin is fair* (p = 0.5) |
| **Alternative Hypothesis** | `H₁` (or `Hₐ`) | The claim you're testing for — *the coin is not fair* |
| **P-value** | `p` | The probability, **assuming H₀ is true**, of observing a result at least as extreme as what you got (60+ heads out of 100) |
| **Significance level** | `α` (alpha) | The threshold you decide *before* testing — commonly `0.05` — below which you reject `H₀` |
| **Two-tailed test** | — | Splits the significance level across both tails of the distribution (here, 2.5% in each tail for a 5% total `α`), since "not fair" could mean biased toward *either* heads or tails |

### The Logic Flow
1. State `H₀`: the coin is fair.
2. State `H₁`: the coin is not fair.
3. Run the experiment: 100 tosses → 60 heads observed.
4. Compute the p-value: the probability of getting 60+ (or equivalently ≤40, for a two-tailed test) heads out of 100 tosses **if the coin really were fair**.
5. Compare `p` to `α` (0.05):
   - If `p < α` → reject `H₀` (the result is unlikely enough under fairness that you conclude the coin is probably biased)
   - If `p ≥ α` → fail to reject `H₀` (not enough evidence to say the coin is unfair — this does **not** prove it's fair, just that 60 heads isn't surprising enough)

### Important Nuance
- **Failing to reject `H₀` is not proving `H₀`.** It only means your data didn't give you enough evidence against it.
- The p-value is *not* "the probability the null hypothesis is true." It's the probability of the observed (or more extreme) data, **given that** `H₀` is true.
- `α = 0.05` is a convention, not a law of nature — in high-stakes or high-cost-of-error contexts (like fraud detection), you might use a stricter threshold (e.g., 0.01).

### Python Implementation

```python
from scipy import stats

# H0: coin is fair (p = 0.5)
# H1: coin is not fair (p != 0.5)  -> two-tailed test

n_trials = 100
n_heads = 60
p_null = 0.5
alpha = 0.05

# Binomial test: exact test for a proportion
result = stats.binomtest(n_heads, n_trials, p_null, alternative="two-sided")
p_value = result.pvalue

print(f"Observed: {n_heads}/{n_trials} heads")
print(f"P-value: {p_value:.4f}")
print(f"Significance level (alpha): {alpha}")

if p_value < alpha:
    print("Reject H0: evidence suggests the coin is NOT fair.")
else:
    print("Fail to reject H0: not enough evidence the coin is biased.")
```

```python
# Visualizing the sampling distribution under H0 with rejection regions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

n, p = 100, 0.5
mu, sigma = n * p, np.sqrt(n * p * (1 - p))

x = np.arange(0, 101)
pmf = binom.pmf(x, n, p)

plt.figure(figsize=(8, 4))
plt.bar(x, pmf, color="lightgray", label="Binomial(n=100, p=0.5)")

# Two-tailed 5% rejection region (2.5% each tail)
lower_crit = binom.ppf(0.025, n, p)
upper_crit = binom.ppf(0.975, n, p)

plt.axvline(lower_crit, color="red", linestyle="--", label="Critical values (α=0.05, two-tailed)")
plt.axvline(upper_crit, color="red", linestyle="--")
plt.axvline(n_heads, color="blue", label=f"Observed = {n_heads}")
plt.legend()
plt.title("Hypothesis Test: Is the Coin Fair?")
plt.xlabel("Number of Heads")
plt.show()
```

### Connecting It Back to CLT and Chebyshev
This is exactly why the earlier sections matter:
- The CLT is why the binomial distribution of "number of heads in 100 flips" looks approximately **normal**, letting you use z-based critical values as an approximation to the exact binomial test.
- Chebyshev's Inequality gives you a fallback distribution-free bound if you couldn't assume normality at all.

---

## 8. Confidence Intervals

### Core Idea
A **Confidence Interval (CI)** gives a range of plausible values for an unknown population parameter (like the true mean `μ`), built from sample data, along with a stated confidence level (e.g., 95%).

```
CI = Point Estimate ± Margin of Error
```

A 95% CI does **not** mean "there's a 95% chance the true mean is in this range." It means: if you repeated this sampling process many times and built a CI each time, ~95% of those intervals would contain the true population mean.

### The Formula (known population σ)

```
CI = x̄ ± Z(α/2) * (σ / √n)
```

| Symbol | Name | Meaning |
|---|---|---|
| `x̄` | sample mean | Your point estimate of the population mean |
| `α` (alpha) | significance level | `1 - confidence level` (e.g., 0.05 for a 95% CI) |
| `Z(α/2)` | critical Z-value | The z-score marking off `α/2` in each tail of the standard normal curve |
| `σ` | population standard deviation | Spread of the population (assumed known here) |
| `n` | sample size | Number of observations in your sample |
| `σ/√n` | standard error | How much the sample mean is expected to vary from the true mean |

### Worked Example — Average Size of Sharks

Given: `σ = 100`, `n = 30`, `x̄ = 500`, confidence level = 95% → `α = 0.05`

**Step 1 — Find the critical Z-value.**
For a two-tailed 95% CI, each tail holds `α/2 = 0.025`.
```
Z(0.025) = 1 - 0.025 = 0.975  →  from the Z-table, Z = 1.96
```

**Step 2 — Compute the margin of error and bounds.**
```
Lower limit = 500 - 1.96 × (100 / √30) = 386
Upper limit = 500 + 1.96 × (100 / √30) = 613
```

**Result:** We are 95% confident the true average shark size lies between **386 and 613** (units as given, e.g., cm).

### Common Z-values for Confidence Levels

| Confidence Level | α | α/2 | Z(α/2) |
|---|---|---|---|
| 90% | 0.10 | 0.05 | 1.645 |
| 95% | 0.05 | 0.025 | 1.96 |
| 99% | 0.01 | 0.005 | 2.576 |

### Python Implementation

```python
import numpy as np
from scipy import stats

x_bar = 500      # sample mean
sigma = 100       # known population std dev
n = 30            # sample size
confidence = 0.95
alpha = 1 - confidence

# Critical Z-value for a two-tailed interval
z_crit = stats.norm.ppf(1 - alpha / 2)
print(f"Z-critical: {z_crit:.3f}")

margin_of_error = z_crit * (sigma / np.sqrt(n))
lower = x_bar - margin_of_error
upper = x_bar + margin_of_error

print(f"{confidence:.0%} CI: ({lower:.0f}, {upper:.0f})")

# When population sigma is UNKNOWN, use the sample std dev and a t-distribution instead:
sample_data = np.array([480, 510, 495, 530, 470, 505, 520, 490, 500, 515])
sample_mean = sample_data.mean()
sample_sem = stats.sem(sample_data)  # standard error of the mean
ci_t = stats.t.interval(confidence, df=len(sample_data) - 1,
                         loc=sample_mean, scale=sample_sem)
print(f"t-based {confidence:.0%} CI: ({ci_t[0]:.1f}, {ci_t[1]:.1f})")
```

**When to use Z vs. t:** use the Z-distribution when population `σ` is known (or `n` is large, ~30+); use the **t-distribution** when `σ` is unknown and estimated from the sample — this is the far more common real-world case, and it produces a slightly wider (more conservative) interval.

---

## 9. The Bernoulli Distribution

### Core Idea
The **Bernoulli distribution** models a single trial with exactly two possible outcomes: success (`X = 1`) or failure (`X = 0`) — e.g., a fraud/not-fraud flag, a coin flip, a churn/no-churn event.

### The Probability Mass Function (PMF)

```
P(X = x) = p^x * (1 - p)^(1 - x)
```

| Symbol | Name | Meaning |
|---|---|---|
| `X` | random variable | Takes value 0 or 1 |
| `x` | specific outcome | Either 0 (failure) or 1 (success) |
| `p` | probability of success | `P(X = 1)` |
| `1 - p` (often `q`) | probability of failure | `P(X = 0)` |

Plugging in the two possible outcomes:
- `P(X = 1) = p`
- `P(X = 0) = 1 - p`

### Mean, Variance, and Standard Deviation

| Statistic | Formula | Meaning |
|---|---|---|
| **Mean (Mode too)** | `p` | The expected value equals the success probability; the mode is whichever of 0/1 has higher probability |
| **Variance** | `pq` (i.e., `p(1-p)`) | Spread of the outcomes |
| **Standard Deviation** | `√(pq)` | Square root of variance |

Note variance is maximized at `p = 0.5` (maximum uncertainty) and shrinks to 0 as `p` approaches 0 or 1 (outcome becomes near-certain).

### Worked Examples (from the three PMF bars)

| Case | P(X=0) | P(X=1) | Mean (p) | Variance (pq) | Std Dev |
|---|---|---|---|---|---|
| Red | 0.2 | 0.8 | 0.8 | 0.16 | 0.40 |
| Blue | 0.8 | 0.2 | 0.2 | 0.16 | 0.40 |
| Green | 0.5 | 0.5 | 0.5 | 0.25 | 0.50 |

### Python Implementation

```python
import numpy as np
from scipy import stats

p = 0.2  # e.g., probability a transaction is fraudulent

bernoulli = stats.bernoulli(p)

print("Mean:", bernoulli.mean())          # p
print("Variance:", bernoulli.var())       # p * (1 - p)
print("Std Dev:", bernoulli.std())        # sqrt(p * (1 - p))

# PMF at each outcome
print("P(X=0):", bernoulli.pmf(0))        # 1 - p
print("P(X=1):", bernoulli.pmf(1))        # p

# Simulate 10,000 Bernoulli trials (e.g., fraud flags)
rng = np.random.default_rng(0)
samples = rng.binomial(n=1, p=p, size=10_000)
print("Empirical mean:", samples.mean())
print("Empirical variance:", samples.var())
```

**Why it matters for fraud/churn ML:** the target label in a binary classifier (`fraud`/`not fraud`, `churn`/`not churn`) is a Bernoulli random variable. Class imbalance (e.g., `p = 0.02` for fraud) is exactly why fraud datasets have low variance around 0 and need techniques like resampling or class weighting.

---

## 10. Five-Number Summary, Percentiles & IQR (Outlier Detection)

### The Five-Number Summary

| # | Statistic | Meaning |
|---|---|---|
| 1 | **Minimum** | Smallest value in the sorted dataset |
| 2 | **Q1 (25th percentile)** | Value below which 25% of the data falls |
| 3 | **Median (Q2 / 50th percentile)** | Middle value — 50% of data falls below it |
| 4 | **Q3 (75th percentile)** | Value below which 75% of the data falls |
| 5 | **Maximum** | Largest value in the sorted dataset |

This is exactly what a **box plot** visualizes.

### The Percentile Formula

```
Percentile rank position = (P / 100) * (n + 1)
```

| Symbol | Name | Meaning |
|---|---|---|
| `P` | desired percentile | e.g., 25 for Q1, 75 for Q3 |
| `n` | number of data points | Sample size |
| Result | rank/position | Which position in the *sorted* list to read off (interpolating between positions if it's not a whole number) |

### Worked Example
Dataset (sorted, `n = 10`): `1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10` *(11 values in the board example)*

- **Minimum = 1**
- **Q1 (25%)**: position `= (25/100) × (11+1) = 3` → value at position 3 = **3**
- **Median (50%)**: position `= (50/100) × 12 = 6` → value at position 6 = **6**
- **Q3 (75%)**: position `= (75/100) × 12 = 9` → value at position 9 = **9**
- **Maximum = 10**

A fractional example: for `P = 25`, `n = 13` → position `= (25/100) × 14 = 3.5` → interpolate between the 3rd and 4th sorted values (board rounds this to ≈3rd/4th).

### Interquartile Range (IQR) — Outlier Detection

```
IQR = Q3 - Q1
```

Using the worked example: `IQR = 9 - 3 = 6`

**Outlier fences** (the standard rule used with box plots):
```
Lower fence = Q1 - 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```
Any data point outside `[Lower fence, Upper fence]` is flagged as an outlier.

| Symbol | Name | Meaning |
|---|---|---|
| `Q1` | first quartile | 25th percentile |
| `Q3` | third quartile | 75th percentile |
| `IQR` | interquartile range | Spread of the middle 50% of the data — robust to outliers (same spirit as the median) |

### Python Implementation

```python
import numpy as np
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10])

q1 = data.quantile(0.25)
median = data.quantile(0.50)
q3 = data.quantile(0.75)
iqr = q3 - q1

print(f"Min: {data.min()}, Q1: {q1}, Median: {median}, Q3: {q3}, Max: {data.max()}")
print(f"IQR: {iqr}")

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = data[(data < lower_fence) | (data > upper_fence)]
print(f"Outlier fences: [{lower_fence}, {upper_fence}]")
print("Outliers found:", outliers.tolist())

# Quick visual: box plot
import matplotlib.pyplot as plt
plt.boxplot(data, vert=False)
plt.title("Five-Number Summary / Box Plot")
plt.show()
```

**Applying this to your fraud detection work:** IQR-based fencing is a standard first-pass outlier flag on transaction amounts — but remember from Section 2 that flagged "outliers" in fraud data are often the *signal itself* (large fraudulent transactions), not noise to remove. Use domain judgment before dropping them.

---

## 11. Sampling Techniques

### Why Sampling Matters
You rarely have access to an entire population (e.g., every shark in the sea, every voter in a state). Sampling techniques determine *how* you pick a subset — and a bad technique produces a **biased** sample no matter how large it is (this connects back to the CLT's "random sampling" condition in Section 3).

### The Four Core Techniques

| # | Technique | How It Works | Example |
|---|---|---|---|
| 1 | **Random Sampling** | Every member of the population has an equal chance of selection | Randomly drawing 1,000 names from a full voter roll |
| 2 | **Stratified Sampling** | Population is split into meaningful subgroups ("strata"), then randomly sampled *within* each group, often proportionally | Sampling 700 men and 300 women to match a population that's 70/30 male/female |
| 3 | **Systematic Sampling** | Pick every *k*-th member from an ordered list, starting from a random point | Surveying every 10th customer who walks through the door |
| 4 | **Cluster Sampling** | Population is divided into clusters (often by domain/geography), then entire clusters are randomly selected | Randomly picking 5 hospitals (clusters) and surveying every patient within them, rather than sampling patients across all hospitals |

### Worked Example — Exit Poll Bias
Setup: predicting an election outcome (Party A vs. Party B vs. Party C) using an exit poll.

- Population = 100,000 voters; sample size drawn = 1,000
- If the sample is built with a **1:2 ratio** favoring one gender/group (e.g., 500 men : 80 women) instead of matching the true population ratio, the resulting estimate (e.g., "Party A wins 7:3") will be **biased** — it reflects the sampling skew, not the true population preference.
- The fix is **stratified sampling**: match the sample's subgroup proportions (e.g., men/women, or region splits) to the population's true proportions before drawing randomly within each stratum.

This is precisely why real exit polls can mispredict election outcomes — an unrepresentative sample breaks the CLT's independence/random-sampling assumptions, so the sample mean no longer reliably estimates the population mean, regardless of sample size.

### Python Implementation

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Simulated population: 100,000 voters with a gender field and a preferred party
population = pd.DataFrame({
    "gender": rng.choice(["M", "F"], size=100_000, p=[0.5, 0.5]),
    "party": rng.choice(["A", "B", "C"], size=100_000, p=[0.45, 0.35, 0.20])
})

# 1. Random Sampling — equal chance for everyone
random_sample = population.sample(n=1000, random_state=42)
print("Random sample party split:\n", random_sample["party"].value_counts(normalize=True))

# 2. Stratified Sampling — preserve the true gender proportions
stratified_sample = population.groupby("gender", group_keys=False).apply(
    lambda g: g.sample(frac=1000 / len(population), random_state=42)
)
print("\nStratified sample gender split:\n", stratified_sample["gender"].value_counts(normalize=True))

# 3. Systematic Sampling — every k-th row after a random start
k = len(population) // 1000
start = rng.integers(0, k)
systematic_sample = population.iloc[start::k]
print("\nSystematic sample size:", len(systematic_sample))

# 4. Cluster Sampling — randomly select whole clusters (e.g., simulate 10 regions)
population["region"] = rng.integers(0, 10, size=len(population))
selected_clusters = rng.choice(population["region"].unique(), size=3, replace=False)
cluster_sample = population[population["region"].isin(selected_clusters)]
print("\nCluster sample size (3 of 10 regions):", len(cluster_sample))
```

**Practical takeaway for churn/fraud analytics:** when building train/test splits or evaluation cohorts from imbalanced data (e.g., a small fraud class), **stratified sampling** (`train_test_split(..., stratify=y)` in scikit-learn) is essential — otherwise a random split can under-represent the minority class and give you a misleading evaluation.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 12. Z-Score and Its Applications

### Core Idea
The **Z-score** tells you how many standard deviations a specific data point is away from the mean. It's the same formula used in Section 1, but here the focus is on its two big real-world uses: **standardization** for ML models, and **comparing values across different distributions**.

### The Formula

```
Z = (xᵢ - μ) / σ
```

| Symbol | Name | Meaning |
|---|---|---|
| `xᵢ` | individual data point | The specific value you're converting |
| `μ` | mean | Mean of the distribution `xᵢ` belongs to |
| `σ` | standard deviation | Spread of that same distribution |
| `Z` | Z-score | How many std devs `xᵢ` is from the mean (sign shows direction: + above, - below) |

### Worked Example — Single Value
Given `μ = 120`, `σ = 12`, `x = 144`:
```
Z = (144 - 120) / 12 = 24 / 12 = 2
```
This point sits exactly **2 standard deviations above the mean**.

### Application 1 — Standardization (Feature Scaling in ML)
Before feeding features like **age, weight, height** into a machine learning model, they're often on very different scales. Standardization (Z-score scaling) rescales every feature to have mean 0 and std dev 1, so no single feature dominates a distance-based or gradient-based algorithm just because of its raw scale.

This is exactly what `sklearn.preprocessing.StandardScaler` implements — and it's the same transformation used "in-stream" (i.e., applied consistently to both training and incoming production data) in real ML pipelines.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([
    [25, 70, 175],   # age, weight (kg), height (cm)
    [40, 85, 180],
    [30, 60, 165],
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```

### Application 2 — Comparing Scores Across Different Distributions
Z-scores let you fairly compare two values that come from **different distributions** (different means and std devs), because both get converted to the same standard scale.

**Worked example — India's batting average, 2020 vs. 2021:**

| Year | Avg (μ) | Std Dev (σ) | Player's Score |
|---|---|---|---|
| 2020 | 181 | 12 | 187 |
| 2021 | 182 | 5 | 185 |

```
Z(2020) = (187 - 181) / 12 = 6/12  = 0.5
Z(2021) = (185 - 182) / 5  = 3/5   = 0.6
```

**Interpretation:** even though the raw 2021 score (185) is lower than the raw 2020 score (187), the **2021 performance is relatively better** — it sits 0.6 standard deviations above that year's mean, compared to only 0.5 for 2020. Raw scores alone would have given the wrong conclusion, because the two years had different team-average performance and different consistency (spread).

### Python Implementation

```python
def z_score(x, mu, sigma):
    return (x - mu) / sigma

# Single-value example
print(z_score(144, mu=120, sigma=12))   # 2.0

# Comparing across two different distributions
z_2020 = z_score(187, mu=181, sigma=12)
z_2021 = z_score(185, mu=182, sigma=5)

print(f"2020 Z-score: {z_2020:.2f}")
print(f"2021 Z-score: {z_2021:.2f}")
print("Better relative performance:", "2021" if z_2021 > z_2020 else "2020")
```

### Why This Matters for ML
1. **Standardization** — nearly every distance-based (k-NN, k-means), gradient-based (linear/logistic regression, neural nets), or regularized (Ridge/Lasso) model benefits from or requires standardized features.
2. **Cross-distribution comparison** — useful for anomaly/fraud scoring: a transaction amount that's "high" needs to be judged relative to *that customer's or that merchant category's* mean and std dev, not a global raw threshold. A Z-score-based anomaly flag naturally adapts to each group's distribution.

---

## 13. Power Law Distribution (Pareto Distribution & the 80-20 Rule)

### Core Idea
A **Power Law Distribution** describes situations where **a relative change in one quantity results in a proportional change in another quantity** — producing a curve with a sharp spike near the origin and a long, slowly-decaying tail. This is fundamentally different from the symmetric bell curve of a Gaussian (normal) distribution.

| Distribution | Shape | Typical Examples |
|---|---|---|
| **Gaussian (Normal)** | Symmetric bell curve | Age, weight, height |
| **Log-Normal** | Right-skewed, single peak, moderate tail | Income (moderate skew), some fraud amounts |
| **Power Law / Pareto** | Extreme spike near the minimum, very long tail | Wealth distribution, sales concentration, bug/crash concentration |

### The Pareto Distribution (Pareto Type I)

The most common power law distribution. Its probability density function (PDF) is controlled by a shape parameter `α` (alpha):

| Symbol | Name | Meaning |
|---|---|---|
| `α` (alpha) | shape parameter | Controls how "heavy" the tail is — smaller `α` = heavier tail, more extreme inequality |
| `xₘ` | scale parameter (minimum value) | The smallest possible value the variable can take |
| `Pr(X = x)` | probability density | Height of the curve at value `x` |
| `Pr(X ≤ x)` | cumulative probability | Probability that the variable is at most `x` (the CDF) |

**Reading the PDF chart:** as `α` decreases (1, 2, 3, ...→∞), the curve becomes less extreme — at `α = ∞` the distribution collapses to a single spike (a Dirac delta) at `xₘ`, meaning zero variability. Smaller `α` values (like `α = 1`) produce the classic "few dominate, many trail off" shape.

**Reading the CDF chart:** the cumulative distribution function shows how quickly probability accumulates. Larger `α` reaches close to 1.0 (100%) very quickly (concentrated near the minimum), while smaller `α` climbs more slowly (heavier tail, more spread into large values).

### The 80-20 Rule (Pareto Principle)
A famous special case / rule of thumb derived from Pareto-like distributions:

> **80% of the effects come from 20% of the causes.**

| # | Real-World Example |
|---|---|
| 1 | 80% of a company's **sales** come from 20% of its overall **products** |
| 2 | 80% of **software crashes** are caused by 20% of the overall **bugs** |
| 3 | 80% of a population's **wealth** is held by 20% of the people |
| 4 | (generalized) 80% of outcomes trace back to 20% of inputs, in many business/ops contexts |

This is why the 80-20 rule shows up constantly in **prioritization**: fixing the top 20% of bugs eliminates most crashes; focusing on the top 20% of products/customers drives most revenue.

### Distinguishing the Three Curve Shapes

| Shape | Peak Location | Tail Behavior | Typical Cause |
|---|---|---|---|
| **Gaussian** | Centered, symmetric | Thin, symmetric on both sides | Additive processes (many small independent effects summing up) |
| **Log-Normal** | Off-center, single peak, moderate right skew | Moderate tail | Multiplicative processes (effects multiply rather than add) |
| **Power Law / Pareto** | Sharp spike near the minimum | Very long, "fat" tail | Preferential attachment / "rich get richer" dynamics (wealth, popularity, network connections) |

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

alpha = 2.0    # shape parameter
xm = 1.0       # minimum value (scale)

# Generate Pareto Type I distributed data
rng = np.random.default_rng(0)
pareto_data = (rng.pareto(alpha, size=10_000) + 1) * xm

# PDF and CDF from scipy
x = np.linspace(1, 10, 500)
pareto_dist = stats.pareto(b=alpha, scale=xm)
pdf = pareto_dist.pdf(x)
cdf = pareto_dist.cdf(x)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(x, pdf, color="crimson")
axes[0].set_title(f"Pareto PDF (alpha={alpha})")

axes[1].plot(x, cdf, color="steelblue")
axes[1].set_title(f"Pareto CDF (alpha={alpha})")
plt.tight_layout()
plt.show()

# Verifying an 80-20-style concentration in simulated "sales" data
sales = np.sort(pareto_data)[::-1]              # descending
cum_sales = np.cumsum(sales) / sales.sum()
top_20_pct_count = int(0.2 * len(sales))
print(f"Top 20% of items generate {cum_sales[top_20_pct_count - 1]:.1%} of total sales")
```

### Why This Matters for ML and Analytics
- **Feature engineering:** power-law-distributed features (transaction counts per customer, revenue per merchant, connections per node in a fraud ring graph) usually need a log transform before use in linear models — similar to the log-normal case in Section 1, but often needing more aggressive transforms (e.g., `log1p`) because of the extreme spike near the minimum.
- **Fraud/churn relevance:** fraud amounts and merchant-level fraud counts frequently follow power-law-like concentration — a small fraction of merchants or accounts can account for the majority of fraud losses, which is a direct analog of the 80-20 rule and useful for prioritizing investigation resources.
- **Distinguishing from log-normal:** if the histogram shows a moderate right skew with one clear peak, treat it as log-normal (Section 1's method). If it shows an extreme spike at the minimum with a very long flat tail, treat it as power-law/Pareto — the imputation, scaling, and modeling choices differ.

---

## Quick-Reference Symbol Glossary

| Symbol | Name | Common Meaning |
|---|---|---|
| `μ` | mu | Population/sample mean |
| `σ` | sigma | Standard deviation |
| `σ²` | sigma squared | Variance |
| `Z` | Z-score | Standardized value, `(x - μ)/σ` |
| `X̄` / `x̄` | x-bar | Sample mean |
| `n` | sample size | Number of observations |
| `k` | — | Generic multiplier (e.g., number of std devs in Chebyshev) |
| `ρ` / `rs` | rho | Spearman's rank correlation coefficient |
| `r` | — | Pearson correlation coefficient |
| `Cov(X,Y)` | covariance | Joint variability of X and Y |
| `dᵢ` | rank difference | `rank(xᵢ) - rank(yᵢ)` in Spearman's formula |
| `H₀` | H-naught | Null hypothesis |
| `H₁` / `Hₐ` | H-one / H-a | Alternative hypothesis |
| `p` (p-value) | — | Probability of the observed data under `H₀` |
| `α` | alpha | Significance level / threshold for rejecting `H₀` (also `1 - confidence level` in a CI) |
| `Pr(...)` | probability | Probability of an event |
| `Z(α/2)` | critical Z-value | Z-score cutting off `α/2` in each tail, used to build a confidence interval |
| `p` (Bernoulli) | probability of success | `P(X = 1)` in a Bernoulli trial |
| `q` | — | `1 - p`, probability of failure in a Bernoulli trial |
| `Q1, Q2, Q3` | quartiles | 25th, 50th (median), and 75th percentiles |
| `IQR` | interquartile range | `Q3 - Q1`; spread of the middle 50% of data |
| `α` (Pareto) | shape parameter | Controls tail heaviness of a Pareto/power-law distribution (smaller = heavier tail) |
| `xₘ` | scale parameter | Minimum possible value in a Pareto distribution |

---

## Suggested Practice Path
1. Implement each Python snippet above on your own data (start with the Kaggle fraud dataset you're already using).
2. For your fraud/churn features: check skew → decide log-transform + Z vs. Chebyshev bounds → choose median/mode for imputation → check Pearson vs. Spearman for feature relationships → run a hypothesis test on a business question (e.g., "is the fraud rate significantly different between two merchant categories?").
3. Re-derive the Spearman worked example by hand once, then verify against `scipy.stats.spearmanr`.
4. Build a 95% confidence interval around your fraud rate estimate from a sample, then re-derive it using the t-distribution version.
5. Treat your binary fraud/churn label as a Bernoulli variable — compute its mean (base rate) and variance by hand, then confirm with `scipy.stats.bernoulli`.
6. Run the five-number summary + IQR fencing on a raw transaction-amount column before deciding whether flagged "outliers" are noise or genuine fraud signal.
7. Audit your train/test split: confirm you used stratified sampling on the target label, not plain random sampling.
8. Standardize a mixed-scale feature set (e.g., age, transaction amount, account tenure) with `StandardScaler`, and manually verify one Z-score by hand.
9. Plot the distribution of transaction amounts or fraud counts per merchant — check whether it looks Gaussian, log-normal, or power-law/Pareto, and let that decide your transform and imputation strategy.
