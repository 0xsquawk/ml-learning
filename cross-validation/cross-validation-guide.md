# Cross-Validation for Model Selection — End-to-End Guide

A practical reference for using cross-validation (CV) to select and tune models, with runnable `scikit-learn` syntax throughout.

---

## Table of Contents

1. [Why Cross-Validation](#1-why-cross-validation)
2. [Core Concept](#2-core-concept)
3. [CV Strategies](#3-cv-strategies)
4. [Basic Syntax: cross_val_score](#4-basic-syntax-cross_val_score)
5. [cross_validate — Multiple Metrics](#5-cross_validate--multiple-metrics)
6. [Cross-Validation for Model Selection](#6-cross-validation-for-model-selection)
7. [Hyperparameter Tuning with CV](#7-hyperparameter-tuning-with-cv)
8. [Pipelines: Avoiding Data Leakage](#8-pipelines-avoiding-data-leakage)
9. [Cross-Validation with Imbalanced Data](#9-cross-validation-with-imbalanced-data)
10. [Nested Cross-Validation](#10-nested-cross-validation)
11. [Custom Scoring Functions](#11-custom-scoring-functions)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Quick Reference Cheat Sheet](#13-quick-reference-cheat-sheet)

---

## 1. Why Cross-Validation

A single train/test split gives one noisy estimate of model performance — it depends heavily on which rows happen to land in the test set. Cross-validation solves this by repeatedly splitting the data, training and evaluating on each split, and averaging the results. This gives:

- A more **reliable estimate** of how a model generalizes to unseen data
- A way to **compare models fairly** (same splits, same conditions)
- A way to **tune hyperparameters** without touching the final held-out test set
- An estimate of **variance** in performance (not just the mean)

**Rule of thumb:** CV is for model selection and tuning. A final held-out test set is still needed for the last, unbiased performance check.

---

## 2. Core Concept

```
Full Dataset
   ├── Train/Test Split (test set locked away, untouched until the very end)
   │
   └── Training Set → split into K folds
         Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5

         Iteration 1: Train on [2,3,4,5] → Validate on [1]
         Iteration 2: Train on [1,3,4,5] → Validate on [2]
         Iteration 3: Train on [1,2,4,5] → Validate on [3]
         Iteration 4: Train on [1,2,3,5] → Validate on [4]
         Iteration 5: Train on [1,2,3,4] → Validate on [5]

         Final CV score = average (and std) across the 5 validation scores
```

Only after model selection/tuning is finalized using CV do you touch the original locked-away test set — once.

---

## 3. CV Strategies

| Strategy | Class | When to use |
|---|---|---|
| K-Fold | `KFold` | Generic regression, balanced data |
| Stratified K-Fold | `StratifiedKFold` | Classification, especially **imbalanced classes** (default recommendation for churn/fraud) |
| Group K-Fold | `GroupKFold` | Data has groups that must not leak across folds (e.g., same customer/user appears multiple times) |
| Time Series Split | `TimeSeriesSplit` | Temporal data — never train on the future to predict the past |
| Leave-One-Out | `LeaveOneOut` | Very small datasets |
| Repeated K-Fold | `RepeatedKFold` / `RepeatedStratifiedKFold` | Want lower variance in the estimate, repeats K-Fold with different shuffles |
| Shuffle Split | `ShuffleSplit` / `StratifiedShuffleSplit` | Want independent control over train/test size and number of iterations |

For a **customer churn** use case specifically (imbalanced target, no strict time-leakage concern, but customers should not repeat across folds if your data has multiple rows per customer): use `StratifiedKFold`, or `GroupKFold`/`StratifiedGroupKFold` if a customer can appear more than once in the dataset.

### Syntax

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    TimeSeriesSplit, RepeatedStratifiedKFold
)

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified (preserves class ratio in every fold) — use for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group-aware (prevents the same entity appearing in train and validation)
gkf = GroupKFold(n_splits=5)

# Time-ordered data
tscv = TimeSeriesSplit(n_splits=5)

# Repeated stratified — more stable estimate, costs more compute
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

Inspecting the folds manually:

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
    print(f"  val class balance: {y[val_idx].mean():.3f}")
```

---

## 4. Basic Syntax: cross_val_score

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

scores = cross_val_score(
    model, X, y,
    cv=5,                  # int, or a CV splitter object like StratifiedKFold(...)
    scoring='roc_auc',     # metric name (see sklearn scoring docs)
    n_jobs=-1               # parallelize across folds
)

print(f"Fold scores: {scores}")
print(f"Mean AUC: {scores.mean():.4f}  |  Std: {scores.std():.4f}")
```

**Always pass a CV object explicitly for classification** rather than a bare integer, so stratification is guaranteed:

```python
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='average_precision')
```

---

## 5. cross_validate — Multiple Metrics

`cross_val_score` only returns one metric. `cross_validate` gives train/test scores for several metrics at once, plus timing.

```python
from sklearn.model_selection import cross_validate

scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',   # preferred metric for imbalanced churn/fraud problems
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

results = cross_validate(
    model, X, y,
    cv=cv_strategy,
    scoring=scoring,
    return_train_score=True,   # lets you check overfitting: train vs. val gap
    n_jobs=-1
)

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df[[c for c in results_df.columns if 'test_' in c]].mean())
```

A large train-score vs. test-score gap across folds is a classic overfitting signal — useful when comparing something like tuned XGBoost vs. a shallow baseline.

---

## 6. Cross-Validation for Model Selection

The typical workflow: evaluate several candidate model families under identical CV conditions, then pick the best mean score (with variance considered, not just the top mean).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

candidates = {
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
    'gradient_boosting': GradientBoostingClassifier(random_state=42),
    'xgboost': XGBClassifier(
        n_estimators=300, eval_metric='aucpr',
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42
    )
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

summary = []
for name, clf in candidates.items():
    scores = cross_val_score(clf, X, y, cv=cv_strategy, scoring='average_precision', n_jobs=-1)
    summary.append({
        'model': name,
        'mean_pr_auc': scores.mean(),
        'std_pr_auc': scores.std(),
        'min_pr_auc': scores.min(),
        'max_pr_auc': scores.max()
    })

summary_df = pd.DataFrame(summary).sort_values('mean_pr_auc', ascending=False)
print(summary_df)
```

### How to pick a winner (not just "highest mean")

- Prefer the model with the best mean **and** acceptably low std — a high mean with wild fold-to-fold swings is less trustworthy.
- If two models are within one standard deviation of each other, prefer the simpler/cheaper one (Occam's razor) unless interpretability or latency needs override that.
- Use a **paired statistical comparison** across the same folds rather than eyeballing means, when the decision is close:

```python
from scipy.stats import ttest_rel

scores_a = cross_val_score(candidates['xgboost'], X, y, cv=cv_strategy, scoring='average_precision')
scores_b = cross_val_score(candidates['random_forest'], X, y, cv=cv_strategy, scoring='average_precision')

t_stat, p_value = ttest_rel(scores_a, scores_b)
print(f"p-value: {p_value:.4f}")   # p < 0.05 suggests a real difference, not noise
```

---

## 7. Hyperparameter Tuning with CV

### GridSearchCV — exhaustive

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='aucpr', random_state=42),
    param_grid=param_grid,
    scoring='average_precision',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
    refit=True             # refits best model on the full training data automatically
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV PR-AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

### RandomizedSearchCV — faster for large search spaces

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'n_estimators': randint(100, 800),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.29),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='aucpr', random_state=42),
    param_distributions=param_distributions,
    n_iter=50,              # number of random combinations to try
    scoring='average_precision',
    cv=cv_strategy,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
print(random_search.best_params_, random_search.best_score_)
```

### Optuna with CV (common with XGBoost, more control than RandomizedSearchCV)

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'aucpr',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='average_precision', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.study_name = 'churn_xgb_cv_tuning'
study.optimize(objective, n_trials=50)

print(study.best_params, study.best_value)
```

---

## 8. Pipelines: Avoiding Data Leakage

Any preprocessing that "learns" from data (scaling, imputation, encoding, SMOTE, feature selection) must be **fit only on the training fold**, never on the full dataset before splitting. `Pipeline` enforces this automatically inside CV.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(eval_metric='aucpr', random_state=42))
])

scores = cross_val_score(pipe, X, y, cv=cv_strategy, scoring='average_precision', n_jobs=-1)
print(scores.mean())
```

When tuning, prefix parameters with the pipeline step name using `__`:

```python
param_grid = {
    'model__n_estimators': [300, 500],
    'model__max_depth': [4, 6, 8]
}
grid_search = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='average_precision', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

**Why this matters:** fitting a `StandardScaler` or SMOTE on the full dataset before CV leaks information about the validation folds into training, silently inflating your CV scores.

---

## 9. Cross-Validation with Imbalanced Data

For a churn/fraud-style problem, two things need care: (1) folds should preserve class ratio, (2) any resampling (SMOTE, undersampling) must happen **inside** each fold, not before splitting.

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

imb_pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),     # only applied to the training fold at fit time
    ('model', XGBClassifier(eval_metric='aucpr', random_state=42))
])

scores = cross_val_score(
    imb_pipe, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='average_precision',   # PR-AUC — preferred over ROC-AUC under class imbalance
    n_jobs=-1
)
print(f"Mean PR-AUC: {scores.mean():.4f}")
```

Use `imblearn.pipeline.Pipeline`, not `sklearn.pipeline.Pipeline`, whenever a resampler is included — the sklearn version doesn't know how to skip resampling at transform/predict time.

---

## 10. Nested Cross-Validation

When you tune hyperparameters with CV **and** want an unbiased estimate of the tuned model's performance, a single CV loop is optimistic (you've indirectly fit hyperparameters to the validation folds). Nested CV fixes this with an outer loop for evaluation and an inner loop for tuning.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

clf = GridSearchCV(
    estimator=XGBClassifier(eval_metric='aucpr', random_state=42),
    param_grid={'max_depth': [3, 5, 7], 'n_estimators': [200, 400]},
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1
)

nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='average_precision', n_jobs=-1)
print(f"Nested CV PR-AUC: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

Use nested CV when reporting a final, defensible performance number (e.g., in a model card or stakeholder report). Use plain CV + a held-out test set for day-to-day iteration — it's much cheaper.

---

## 11. Custom Scoring Functions

Built-in metrics don't always match business objectives (e.g., a cost-weighted churn metric). Wrap a custom function with `make_scorer`.

```python
from sklearn.metrics import make_scorer, fbeta_score

# F2 score: weights recall higher than precision — useful when missing a churner is costlier
f2_scorer = make_scorer(fbeta_score, beta=2)

scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=f2_scorer, n_jobs=-1)
```

```python
# Fully custom business-cost function
def churn_cost_score(y_true, y_pred):
    # Example: cost of missing a churner is 5x the cost of a false alarm
    fn_cost = 5
    fp_cost = 1
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    total_cost = fn * fn_cost + fp * fp_cost
    return -total_cost   # negative because sklearn scorers maximize

cost_scorer = make_scorer(churn_cost_score)
scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=cost_scorer, n_jobs=-1)
```

---

## 12. Common Pitfalls

| Pitfall | Why it's a problem | Fix |
|---|---|---|
| Scaling/encoding before splitting | Leaks validation-fold statistics into training | Put preprocessing inside a `Pipeline` |
| SMOTE/resampling before CV split | Validation folds contain synthetic points derived from training data | Use `imblearn.Pipeline` so resampling only touches the training fold |
| Plain `KFold` on imbalanced classification | Some folds may end up with very few/no positive cases | Use `StratifiedKFold` |
| Ignoring group structure (e.g. repeated customer rows) | Same entity in train and validation inflates scores | Use `GroupKFold` / `StratifiedGroupKFold` |
| Shuffling time-series data | Trains on future to predict the past | Use `TimeSeriesSplit`, never shuffle |
| Picking "best" model by mean score alone | Ignores variance/instability across folds | Look at std, consider a paired t-test |
| Reusing the same CV folds to tune and then report final performance | Optimistic bias | Use nested CV or a separate locked test set |
| Using accuracy on imbalanced data | Misleading — a model predicting all-majority-class looks "good" | Use PR-AUC, F1, recall, or a cost-based metric |

---

## 13. Quick Reference Cheat Sheet

```python
# --- Imports ---
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- Standard imbalanced-classification CV setup ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(eval_metric='aucpr', random_state=42))
])

# 1. Quick single-metric check
cross_val_score(pipe, X, y, cv=cv, scoring='average_precision', n_jobs=-1)

# 2. Multi-metric check with overfitting diagnostics
cross_validate(pipe, X, y, cv=cv, scoring=['roc_auc','average_precision','f1'],
                return_train_score=True, n_jobs=-1)

# 3. Hyperparameter tuning
GridSearchCV(pipe, param_grid={'model__max_depth':[3,5,7]},
             cv=cv, scoring='average_precision', n_jobs=-1).fit(X_train, y_train)

# 4. Final, unbiased performance estimate
# Nested CV (expensive) OR fit on X_train, evaluate once on locked X_test
```

**Metric choice reminder for imbalanced churn/fraud problems:** prefer `average_precision` (PR-AUC) over `roc_auc` as the primary CV scoring metric — it's far more sensitive to performance on the minority (positive/churn) class.

---

## Further Practice

- Re-run the model-selection comparison (Section 6) on a real churn dataset and plot fold-level PR-AUC as a boxplot per model.
- Try nested CV on a small subset first to sanity-check runtime before scaling up.
- Log every CV run (params, scores, fold-level detail) to MLflow for traceability across experiments.
