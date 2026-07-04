# Supervised ML Algorithms & Metrics Reference

Industry-preferred ranges for classification and regression algorithms, along with required dependencies, usage syntax, and theoretical context.

---

## Table of Contents

**Classification**
- [Logistic Regression](#logistic-regression)
- [Decision Tree](#decision-tree)
- [Random Forest](#random-forest)
- [XGBoost / LightGBM](#xgboost--lightgbm)
- [Support Vector Machine](#support-vector-machine)
- [Neural Network (MLP)](#neural-network-mlp)
- [Naive Bayes](#naive-bayes)
- [K-Nearest Neighbors](#k-nearest-neighbors)

**Regression**
- [Linear Regression](#linear-regression)
- [Ridge / Lasso](#ridge--lasso)
- [Decision Tree Regressor](#decision-tree-regressor)
- [Random Forest Regressor](#random-forest-regressor)
- [XGBoost Regressor](#xgboost-regressor)
- [Support Vector Regressor](#support-vector-regressor)

---

## Classification

### Logistic Regression

**Theoretical summary:** Models the log-odds of a binary (or multinomial) outcome as a linear combination of input features, squashed through a sigmoid function into a probability. Assumes a roughly linear decision boundary in feature space. Best used as an interpretable baseline or when the relationship between features and the log-odds of the outcome is approximately linear — e.g. credit approval, churn propensity, click-through prediction.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
```

**Metrics & preferred ranges:** AUC-ROC ≥ 0.80 · F1 ≥ 0.75 · Accuracy ≥ 85% · Log Loss < 0.50
**Notes:** Interpretable baseline; weakens on imbalanced data.

---

### Decision Tree

**Theoretical summary:** Recursively partitions the feature space into regions that minimize impurity (Gini or entropy) within each region. Produces a human-readable set of if/else rules. Naturally handles non-linear relationships and mixed data types but is prone to high variance — small changes in data can produce very different trees. Suited to problems where interpretability/rule extraction matters (e.g. regulatory-compliant credit scoring, medical triage rules).

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** Accuracy ≥ 82% · F1 ≥ 0.72 · AUC-ROC ≥ 0.78 · Gini Impurity: lower = better
**Notes:** Prone to overfit; excellent for rule extraction; prune aggressively.

---

### Random Forest

**Theoretical summary:** An ensemble of decision trees trained on bootstrapped samples with random feature subsets at each split (bagging). Averages predictions across trees to reduce variance relative to a single tree while retaining the ability to capture non-linear interactions. Well suited to tabular data with moderate size, mixed feature types, and where feature importance rankings are useful (e.g. customer churn drivers, fraud signal ranking).

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
importances = model.feature_importances_
```

**Metrics & preferred ranges:** AUC-ROC ≥ 0.87 · F1 ≥ 0.80 · Accuracy ≥ 88% · OOB Error < 10%
**Notes:** Robust ensemble; built-in feature importance; slower at scale.

---

### XGBoost / LightGBM

**Theoretical summary:** Gradient-boosted tree ensembles that build trees sequentially, each correcting the residual errors of the previous ensemble via gradient descent on a differentiable loss function. Includes built-in L1/L2 regularization, handling of missing values, and support for weighted classes — making it the standard choice for structured/tabular prediction tasks such as fraud detection, credit risk, and churn modeling at scale.

**Dependencies:**
```
pip install xgboost
# or
pip install lightgbm
```

**Syntax:**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio_neg_to_pos,
    eval_metric="aucpr"
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
```

**Metrics & preferred ranges:** AUC-ROC ≥ 0.90 · PR-AUC ≥ 0.72 · F1 ≥ 0.82 · Log Loss < 0.40
**Notes:** Industry gold standard for tabular data; tune regularization to avoid overfit.

---

### Support Vector Machine

**Theoretical summary:** Finds the hyperplane that maximizes the margin between classes, optionally projecting data into higher-dimensional space via a kernel function (RBF, polynomial) to capture non-linear boundaries. Effective in high-dimensional spaces (e.g. text classification with TF-IDF features) but scales poorly with dataset size due to quadratic/cubic training complexity.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** Accuracy ≥ 85% · F1 ≥ 0.80 · AUC-ROC ≥ 0.85
**Notes:** Excellent on high-dimensional data; slow training at large scale; kernel choice critical.

---

### Neural Network (MLP)

**Theoretical summary:** A multi-layer perceptron stacks layers of weighted linear transformations and non-linear activations, trained via backpropagation to approximate arbitrarily complex decision boundaries. Requires larger datasets and careful regularization (dropout, weight decay) to avoid overfitting, but can outperform tree ensembles when feature interactions are highly non-linear or when combined with embeddings for categorical/text data.

**Dependencies:**
```
pip install scikit-learn
# or for larger-scale deep learning
pip install tensorflow
pip install torch
```

**Syntax:**
```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, max_iter=500)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** AUC-ROC ≥ 0.90 · PR-AUC ≥ 0.74 · F1 ≥ 0.83 · Log Loss < 0.35
**Notes:** High capacity; needs large data, regularization, and careful hyperparameter tuning.

---

### Naive Bayes

**Theoretical summary:** Applies Bayes' theorem under the (naive) assumption that features are conditionally independent given the class label. Despite the simplifying assumption, it performs surprisingly well on high-dimensional sparse data such as bag-of-words text representations, and trains almost instantly. A strong, cheap baseline for spam filtering, sentiment analysis, and document categorization.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** Accuracy ≥ 75% · F1 ≥ 0.70 · AUC-ROC ≥ 0.75
**Notes:** Fast and simple; assumes feature independence; strong baseline for text classification.

---

### K-Nearest Neighbors

**Theoretical summary:** A non-parametric, instance-based method that classifies a point by majority vote among its *k* closest neighbors in feature space, under a chosen distance metric (usually Euclidean). Has no explicit training phase (lazy learning), which makes inference expensive at scale. Sensitive to feature scaling and the curse of dimensionality; well suited to smaller, well-behaved datasets like recommender cold-start heuristics.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_train)
model = KNeighborsClassifier(n_neighbors=15, weights="distance")
model.fit(X_scaled, y_train)
preds = model.predict(X_test_scaled)
```

**Metrics & preferred ranges:** Accuracy ≥ 80% · F1 ≥ 0.75 · AUC-ROC ≥ 0.78
**Notes:** No training phase; inference slow at scale; feature scaling is mandatory.

---

## Regression

### Linear Regression

**Theoretical summary:** Models the target as a linear combination of input features, fit by minimizing squared residuals (ordinary least squares). Assumes linearity, homoscedasticity, and normally distributed residuals. Serves as the interpretable baseline for any continuous prediction task — e.g. price forecasting, demand estimation — before reaching for more flexible models.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** R² ≥ 0.75 · MAPE < 20% · MAE / RMSE: domain-dependent
**Notes:** Interpretable baseline; assumes linearity and normal residuals.

---

### Ridge / Lasso

**Theoretical summary:** Extends linear regression with L2 (Ridge) or L1 (Lasso) regularization penalties on the coefficients. Ridge shrinks correlated coefficients together, stabilizing estimates under multicollinearity; Lasso drives some coefficients to exactly zero, performing automatic feature selection. Useful when the feature set is large or highly correlated, such as in genomic or high-cardinality marketing-mix models.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
preds = ridge.predict(X_test)
```

**Metrics & preferred ranges:** R² ≥ 0.78 · MAPE < 18% · RMSE: lower vs. OLS
**Notes:** Ridge handles multicollinearity; Lasso performs feature selection via sparsity.

---

### Decision Tree Regressor

**Theoretical summary:** Applies the same recursive-partitioning logic as a classification tree, but splits are chosen to minimize variance (e.g. mean squared error) within each leaf, and leaf predictions are the mean target value of training samples in that region. Captures non-linear relationships and interactions without requiring feature scaling, at the cost of high variance if left unpruned.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=20)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** R² ≥ 0.76 · MAPE < 22% · MAE: domain-dependent
**Notes:** Captures non-linearity; high variance without pruning; interpretable splits.

---

### Random Forest Regressor

**Theoretical summary:** Bags many regression trees trained on bootstrapped samples and random feature subsets, averaging their outputs to reduce variance. Handles non-linearities, interactions, and outliers robustly without needing feature scaling, and produces useful feature-importance estimates — a common workhorse for demand forecasting and price-elasticity modeling.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** R² ≥ 0.85 · MAPE < 15% · MAE: lowest in class
**Notes:** Handles non-linearity and outliers well; feature importance built in.

---

### XGBoost Regressor

**Theoretical summary:** Sequentially builds regression trees, each fit to the gradient of the loss function (typically squared error) with respect to the current ensemble's predictions, with built-in regularization to control overfitting. Consistently among the top performers on structured/tabular regression benchmarks — used widely for revenue forecasting, risk-adjusted pricing, and demand modeling where marginal accuracy gains matter.

**Dependencies:**
```
pip install xgboost
```

**Syntax:**
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror"
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** R² ≥ 0.88 · MAPE < 10% · RMSE: lowest in class
**Notes:** Top performer on structured regression; early stopping prevents overfit.

---

### Support Vector Regressor

**Theoretical summary:** Extends SVM to regression by fitting a function that keeps predictions within an ε-insensitive margin of the true values, only penalizing errors beyond that margin. This makes it robust to small deviations and outliers, at the cost of scaling poorly to large datasets. Applicable where the cost function tolerates small errors but must control large deviations, such as sensor-calibration or short-horizon financial forecasting.

**Dependencies:**
```
pip install scikit-learn
```

**Syntax:**
```python
from sklearn.svm import SVR

model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

**Metrics & preferred ranges:** R² ≥ 0.80 · MAPE < 18%
**Notes:** ε-insensitive loss gives robustness to outliers; kernel selection is key.

---

## Range Key

- **Good** — target commonly met by well-tuned production models
- **Acceptable** — within tolerance, may need improvement depending on use case
- **Lower = better** — minimize this metric
- **Domain-dependent** — no universal threshold; judge against your specific business context
