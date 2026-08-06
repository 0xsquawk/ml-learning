# The Fraud Analyst's Math Guide
### A practical, code-first path from probability to production fraud models

**Who this is for:** You already understand classification algorithms conceptually (logistic regression, trees, boosting, etc.) but want the underlying math intuition and the statistical toolkit specific to fraud work.

**How to use this guide:** Each phase has (1) the core concepts, (2) why it matters for fraud specifically, (3) runnable Python code, and (4) curated references. Work through phases in order — later phases assume earlier ones.

---

## Table of Contents
1. [Phase 1: Probability & Statistics Foundations](#phase-1-probability--statistics-foundations)
2. [Phase 2: Imbalanced Classification & Evaluation](#phase-2-imbalanced-classification--evaluation)
3. [Phase 3: Linear Algebra Intuition](#phase-3-linear-algebra-intuition)
4. [Phase 4: Calculus Intuition](#phase-4-calculus-intuition)
5. [Phase 5: Applied Fraud-Specific Topics](#phase-5-applied-fraud-specific-topics)
6. [Capstone Project](#capstone-project)
7. [Master Reference List](#master-reference-list)

---

## Phase 1: Probability & Statistics Foundations

### 1.1 Conditional Probability & Bayes' Theorem

This is the single most-used piece of math in fraud analytics. Every fraud score is really a statement about conditional probability: *P(fraud | features)*.

**Bayes' Theorem:**

```
P(Fraud | Signal) = [ P(Signal | Fraud) * P(Fraud) ] / P(Signal)
```

Where:
- `P(Fraud)` = base rate (prior) — often very small, e.g. 0.5%–2%
- `P(Signal | Fraud)` = likelihood — how often fraudsters trigger this signal
- `P(Signal)` = overall probability of the signal firing (fraud + non-fraud combined)

**Why it matters:** A rule that "catches 95% of fraud" sounds great, but if it also fires on 10% of legitimate transactions and fraud is only 1% of volume, most flagged transactions are actually legitimate. Bayes' theorem is how you catch this before it becomes a customer-experience disaster.

**Code — Bayes' theorem for a fraud rule:**

```python
def bayes_fraud_probability(p_fraud_prior, p_signal_given_fraud, p_signal_given_legit):
    """
    p_fraud_prior: base fraud rate, e.g. 0.01
    p_signal_given_fraud: true positive rate of the signal (sensitivity)
    p_signal_given_legit: false positive rate of the signal
    """
    p_legit_prior = 1 - p_fraud_prior
    p_signal = (p_signal_given_fraud * p_fraud_prior) + (p_signal_given_legit * p_legit_prior)
    p_fraud_given_signal = (p_signal_given_fraud * p_fraud_prior) / p_signal
    return p_fraud_given_signal

# Example: fraud is 1% of transactions, rule catches 95% of fraud,
# but also fires on 10% of legit transactions
result = bayes_fraud_probability(
    p_fraud_prior=0.01,
    p_signal_given_fraud=0.95,
    p_signal_given_legit=0.10
)
print(f"P(Fraud | Flagged) = {result:.2%}")
# Output: P(Fraud | Flagged) = 8.75%
# Most flagged transactions are NOT fraud -- this is the base-rate fallacy in action
```

### 1.2 Probability Distributions

Focus on these four:

| Distribution | Use case in fraud |
|---|---|
| **Bernoulli** | Single transaction: fraud (1) or not (0) |
| **Binomial** | Number of fraudulent transactions out of N attempts |
| **Poisson** | Modeling rare-event counts, e.g. fraud alerts per hour |
| **Normal (Gaussian)** | Transaction amounts, z-score anomaly detection |

**Code — simulating and visualizing distributions:**

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Binomial: probability of exactly k fraud cases out of n transactions
n, p = 1000, 0.01  # 1000 transactions, 1% fraud rate
k = 15
prob_exact = stats.binom.pmf(k, n, p)
prob_at_least = 1 - stats.binom.cdf(k - 1, n, p)
print(f"P(exactly {k} fraud cases) = {prob_exact:.4f}")
print(f"P(at least {k} fraud cases) = {prob_at_least:.4f}")

# Poisson: modeling fraud alerts per hour (lambda = average rate)
lam = 3.2  # average 3.2 alerts/hour
prob_5_alerts = stats.poisson.pmf(5, lam)
print(f"P(exactly 5 alerts in an hour) = {prob_5_alerts:.4f}")

# Normal: z-score for anomaly detection on transaction amount
amounts = np.random.normal(loc=50, scale=20, size=10000)  # simulate spending
mean, std = amounts.mean(), amounts.std()
new_transaction = 340
z_score = (new_transaction - mean) / std
print(f"Z-score of ${new_transaction} transaction: {z_score:.2f}")
# z > 3 is a common (naive) anomaly threshold
```

### 1.3 Hypothesis Testing & Confidence Intervals

Used constantly when comparing fraud rates across segments, A/B testing new rules, or validating whether a model change actually improved detection.

**Code — two-proportion z-test (e.g., comparing fraud rates before/after a new rule):**

```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Before: 120 fraud cases out of 50,000 transactions
# After:   80 fraud cases out of 48,000 transactions
counts = np.array([120, 80])
nobs = np.array([50000, 48000])

z_stat, p_value = proportions_ztest(counts, nobs)
print(f"Z-statistic: {z_stat:.3f}, p-value: {p_value:.5f}")

if p_value < 0.05:
    print("Statistically significant reduction in fraud rate")
else:
    print("Not statistically significant -- could be noise")
```

**Code — confidence interval for a fraud rate:**

```python
from statsmodels.stats.proportion import proportion_confint

fraud_count = 120
total = 50000
ci_low, ci_high = proportion_confint(fraud_count, total, alpha=0.05, method='wilson')
print(f"95% CI for fraud rate: [{ci_low:.4%}, {ci_high:.4%}]")
```

### 1.4 Bayesian Thinking Beyond the Formula

Read about priors, posteriors, and updating beliefs sequentially — this maps directly to how real-time fraud engines update risk scores as new signals arrive (device fingerprint → login pattern → transaction amount → velocity check), each step refining the posterior.

### Phase 1 References
- Wheelan, Charles. *Naked Statistics: Stripping the Dread from the Data* — intuitive, non-technical intro. [Publisher page](https://wwnorton.com/books/9780393347777)
- Harvard Stat 110 (Joe Blitzstein) — free, rigorous, excellent for Bayes: https://projects.iq.harvard.edu/stat110/home
- Seeing Theory (visual probability primer, Brown University): https://seeing-theory.brown.edu/
- StatsModels documentation: https://www.statsmodels.org/stable/index.html
- SciPy `stats` module docs: https://docs.scipy.org/doc/scipy/reference/stats.html

---

## Phase 2: Imbalanced Classification & Evaluation

### 2.1 Why Accuracy Lies to You

If fraud is 1% of transactions, a model that predicts "not fraud" for everything is 99% accurate and completely useless. This is the first thing to internalize.

### 2.2 Confusion Matrix & Core Metrics

```
                Predicted Fraud   Predicted Legit
Actual Fraud         TP                FN
Actual Legit         FP                TN
```

- **Precision** = TP / (TP + FP) — of what you flagged, how much was real fraud?
- **Recall (Sensitivity)** = TP / (TP + FN) — of all real fraud, how much did you catch?
- **F1 Score** = harmonic mean of precision and recall
- **ROC-AUC** — good for balanced classes, misleading for rare-event problems
- **PR-AUC (Precision-Recall AUC)** — the metric that actually matters for fraud, because it's not inflated by the large number of true negatives

**Code — full evaluation workflow:**

```python
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                              roc_curve, auc, classification_report,
                              average_precision_score)
import matplotlib.pyplot as plt

# y_true: actual labels (0/1), y_scores: predicted probabilities
def evaluate_fraud_model(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud']))

    # PR-AUC -- the key metric for imbalanced fraud data
    pr_auc = average_precision_score(y_true, y_scores)
    print(f"PR-AUC: {pr_auc:.4f}")

    # ROC-AUC for comparison
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plot PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    return pr_auc, roc_auc
```

### 2.3 Handling Class Imbalance

| Technique | What it does |
|---|---|
| **Class weights** | Penalize the model more for missing fraud cases |
| **SMOTE** | Synthetically oversample the minority (fraud) class |
| **Random undersampling** | Downsample the majority (legit) class |
| **Threshold tuning** | Adjust the decision boundary instead of resampling |

**Code — class weighting (usually the first thing to try):**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# class_weight='balanced' auto-adjusts weights inversely proportional to class frequency
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# For tree-based models
rf = RandomForestClassifier(class_weight='balanced', n_estimators=300)
rf.fit(X_train, y_train)
```

**Code — SMOTE:**

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE only to training data -- never to test data (data leakage otherwise)
smote = SMOTE(random_state=42, sampling_strategy=0.3)  # bring fraud up to 30% of legit count
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
```

### 2.4 Cost-Sensitive Threshold Tuning

In fraud, a false negative (missed fraud) usually costs far more than a false positive (annoyed customer). Encode that explicitly.

**Code — finding the cost-optimal threshold:**

```python
import numpy as np

def find_optimal_threshold(y_true, y_scores, cost_fn=100, cost_fp=5):
    """
    cost_fn: cost of missing a fraud case (chargeback, loss, etc.)
    cost_fp: cost of a false positive (manual review, customer friction)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)

    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]

best_threshold, min_cost = find_optimal_threshold(y_test, y_scores, cost_fn=100, cost_fp=5)
print(f"Optimal threshold: {best_threshold:.2f}, minimum total cost: ${min_cost}")
```

### Phase 2 References
- Google ML Crash Course — Classification: https://developers.google.com/machine-learning/crash-course/classification
- imbalanced-learn documentation: https://imbalanced-learn.org/stable/
- scikit-learn Model Evaluation guide: https://scikit-learn.org/stable/modules/model_evaluation.html
- Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" — *PLOS ONE*: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

---

## Phase 3: Linear Algebra Intuition

You don't need to derive anything here — just build geometric intuition for what's happening inside models you already use.

### 3.1 Core Concepts
- **Vectors** — a transaction's feature set is a vector (amount, time-since-last-txn, merchant category, etc.)
- **Dot product** — the core operation behind logistic regression's linear combination of features
- **Matrix multiplication** — how a batch of transactions gets scored at once
- **Eigenvectors/eigenvalues** — underlie PCA, useful when you need to reduce a large feature set

**Code — logistic regression is just a dot product + sigmoid:**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# A trained logistic regression is just weights (a vector) and a bias
weights = np.array([0.8, -0.3, 1.2, 0.05])   # learned coefficients
bias = -2.1

# A transaction's features (vector)
transaction = np.array([1, 0, 3.5, 200])  # [is_new_device, is_domestic, velocity_score, amount/100]

# The "linear algebra" step: dot product
z = np.dot(weights, transaction) + bias
fraud_probability = sigmoid(z)
print(f"Fraud probability: {fraud_probability:.4f}")

# Scoring a whole batch is just matrix multiplication
batch = np.array([
    [1, 0, 3.5, 200],
    [0, 1, 0.2, 45],
    [1, 1, 5.0, 900],
])
z_batch = batch @ weights + bias   # @ is matrix multiplication in numpy
probs_batch = sigmoid(z_batch)
print(probs_batch)
```

**Code — PCA for feature reduction (dimensionality reduction on eigenvectors):**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fraud datasets often have many engineered features -- PCA compresses them
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # keep 95% of variance
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### Phase 3 References
- 3Blue1Brown, "Essence of Linear Algebra" (YouTube series, ~2.5 hrs total, purely visual): https://www.3blue1brown.com/topics/linear-algebra
- Khan Academy Linear Algebra: https://www.khanacademy.org/math/linear-algebra
- NumPy linear algebra docs: https://numpy.org/doc/stable/reference/routines.linalg.html

---

## Phase 4: Calculus Intuition

Again — intuition, not hand computation. You want to be able to say "the model is learning by nudging weights in the direction that reduces error" and know what that means.

### 4.1 Core Concept: Gradient Descent

A derivative tells you the slope of the loss function at a point. Gradient descent repeatedly takes small steps in the direction that reduces loss.

**Code — gradient descent from scratch (for intuition, not production use):**

```python
import numpy as np

# Simple 1-feature logistic regression trained by hand with gradient descent
np.random.seed(0)
X = np.random.normal(0, 1, 200)          # one feature, e.g. normalized transaction velocity
y = (X + np.random.normal(0, 0.5, 200) > 0.5).astype(int)  # fraud label

w, b = 0.0, 0.0          # start with weight and bias at zero
lr = 0.1                 # learning rate
epochs = 500

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    z = w * X + b
    pred = sigmoid(z)

    # gradients (this is where calculus lives -- derivative of log-loss)
    error = pred - y
    dw = np.mean(error * X)
    db = np.mean(error)

    # the "descent" step -- move opposite the gradient
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        loss = -np.mean(y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.3f}, b={b:.3f}")

print(f"\nFinal learned weight: {w:.3f}, bias: {b:.3f}")
```

This is exactly what `sklearn.LogisticRegression` and every gradient-boosted tree's leaf-weight updates are doing under the hood, just far more optimized.

### Phase 4 References
- 3Blue1Brown, "Essence of Calculus": https://www.3blue1brown.com/topics/calculus
- Google ML Crash Course — Gradient Descent: https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent

---

## Phase 5: Applied Fraud-Specific Topics

### 5.1 Time-Series Anomaly Detection

**Code — rolling z-score anomaly detection:**

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2026-01-01', periods=500, freq='H'),
    'transaction_amount': np.random.normal(50, 15, 500)
})
df.loc[480, 'transaction_amount'] = 800  # inject an anomaly

window = 48  # 48-hour rolling window
df['rolling_mean'] = df['transaction_amount'].rolling(window).mean()
df['rolling_std'] = df['transaction_amount'].rolling(window).std()
df['z_score'] = (df['transaction_amount'] - df['rolling_mean']) / df['rolling_std']

anomalies = df[df['z_score'].abs() > 3]
print(anomalies[['timestamp', 'transaction_amount', 'z_score']])
```

**Code — velocity feature (a fraud-analytics staple):**

```python
# Count of transactions per user in the trailing 1 hour / 24 hours
df = df.sort_values(['user_id', 'timestamp'])
df = df.set_index('timestamp')

df['txn_count_1h'] = (
    df.groupby('user_id')['transaction_amount']
      .rolling('1h').count()
      .reset_index(level=0, drop=True)
)
df['txn_count_24h'] = (
    df.groupby('user_id')['transaction_amount']
      .rolling('24h').count()
      .reset_index(level=0, drop=True)
)
```

### 5.2 Graph-Based Fraud Detection (Fraud Rings)

Fraudsters often share devices, IPs, or payment instruments across "unrelated" accounts. Graph features expose this.

**Code — basic graph construction and centrality with NetworkX:**

```python
import networkx as nx

G = nx.Graph()

# Edges connect entities that share an attribute (device_id, ip, card_bin, etc.)
edges = [
    ('user_1', 'device_A'), ('user_2', 'device_A'),   # shared device -- suspicious
    ('user_3', 'device_B'),
    ('user_1', 'ip_100'), ('user_4', 'ip_100'), ('user_2', 'ip_100'),
]
G.add_edges_from(edges)

# Degree centrality: entities connected to unusually many others are suspicious
centrality = nx.degree_centrality(G)
sorted_centrality = sorted(centrality.items(), key=lambda x: -x[1])
print("Most connected entities (potential fraud ring hubs):")
for node, score in sorted_centrality[:5]:
    print(f"  {node}: {score:.3f}")

# Connected components reveal clusters -- potential fraud rings
components = list(nx.connected_components(G))
print(f"\nFound {len(components)} connected clusters")
for i, comp in enumerate(components):
    if len(comp) > 2:
        print(f"Cluster {i}: {comp}")
```

### 5.3 Feature Engineering Patterns Specific to Fraud

- **Velocity checks**: transaction count/sum in trailing N minutes/hours/days
- **Aggregation windows**: average spend, deviation from personal baseline
- **Device/IP fingerprinting**: entropy of devices per user, geolocation jumps ("impossible travel")
- **Behavioral biometrics**: typing cadence, session duration (advanced, often vendor-provided)
- **Network/graph features**: shared attributes across accounts, clustering coefficient

**Code — "impossible travel" feature:**

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def implied_speed_kmh(lat1, lon1, t1, lat2, lon2, t2):
    distance = haversine_km(lat1, lon1, lat2, lon2)
    hours = (t2 - t1).total_seconds() / 3600
    return distance / hours if hours > 0 else float('inf')

# If implied speed > ~900 km/h (commercial flight speed), flag as impossible travel
speed = implied_speed_kmh(40.7128, -74.0060, pd.Timestamp('2026-01-01 10:00'),
                           51.5074, -0.1278, pd.Timestamp('2026-01-01 10:30'))
print(f"Implied travel speed: {speed:.0f} km/h")
if speed > 900:
    print("FLAG: Impossible travel detected")
```

### Phase 5 References
- Kaggle Credit Card Fraud Detection dataset (real anonymized data, great for practice): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- IEEE-CIS Fraud Detection (Kaggle competition, richer feature set): https://www.kaggle.com/competitions/ieee-fraud-detection
- NetworkX documentation: https://networkx.org/documentation/stable/
- Stripe Engineering Blog (radar/fraud systems): https://stripe.com/blog/engineering
- PayPal Engineering Blog: https://medium.com/paypal-tech

---

## Capstone Project

Once you've worked through all five phases, apply everything on the Kaggle Credit Card Fraud dataset:

1. **EDA**: compute the fraud base rate, visualize amount distributions by class (Phase 1)
2. **Baseline model**: logistic regression with `class_weight='balanced'`, evaluate with PR-AUC not accuracy (Phase 2)
3. **Improve imbalance handling**: try SMOTE vs. class weights, compare PR-AUC (Phase 2)
4. **Threshold tuning**: assign realistic cost values to FN/FP and find the cost-optimal threshold (Phase 2)
5. **Interpret the model**: inspect coefficients/feature importances as a dot product, or reduce dimensionality with PCA if you engineer many features (Phase 3)
6. **Explain training**: write a short note on how the model's weights were learned via gradient descent (Phase 4)
7. **Add engineered features**: velocity, rolling z-scores, and (if you construct synthetic linked accounts) graph centrality (Phase 5)
8. **Write it up**: a one-page summary as if presenting to a non-technical fraud-ops stakeholder — this tests whether you actually understand it or just ran the code

---

## Master Reference List

**Books**
- Wheelan, C. — *Naked Statistics* — https://wwnorton.com/books/9780393347777

**Free courses**
- Harvard Stat 110 — https://projects.iq.harvard.edu/stat110/home
- Google ML Crash Course — https://developers.google.com/machine-learning/crash-course
- Khan Academy Linear Algebra — https://www.khanacademy.org/math/linear-algebra
- 3Blue1Brown, Essence of Linear Algebra — https://www.3blue1brown.com/topics/linear-algebra
- 3Blue1Brown, Essence of Calculus — https://www.3blue1brown.com/topics/calculus
- Seeing Theory (visual probability) — https://seeing-theory.brown.edu/

**Documentation**
- scikit-learn — https://scikit-learn.org/stable/
- imbalanced-learn — https://imbalanced-learn.org/stable/
- statsmodels — https://www.statsmodels.org/stable/
- SciPy stats — https://docs.scipy.org/doc/scipy/reference/stats.html
- NetworkX — https://networkx.org/documentation/stable/
- pandas — https://pandas.pydata.org/docs/

**Datasets for practice**
- Kaggle Credit Card Fraud Detection — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- IEEE-CIS Fraud Detection — https://www.kaggle.com/competitions/ieee-fraud-detection

**Papers**
- Saito & Rehmsmeier (2015), PR curves vs ROC on imbalanced data — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

**Industry blogs (real-world fraud system design)**
- Stripe Engineering — https://stripe.com/blog/engineering
- PayPal Engineering — https://medium.com/paypal-tech

---

*End of guide. Suggested pace: 8–10 weeks part-time, moving faster through Phases 3–4 and spending the most time in Phases 1, 2, and 5.*
