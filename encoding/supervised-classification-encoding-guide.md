# Supervised Classification Algorithms — Encoding Requirements

Reference guide mapping common supervised classification algorithms to the categorical encoding they require.

## Tree-based models

No true ordinality issue since trees split on thresholds, but sklearn/XGBoost still need numeric input.

| Algorithm | Categorical Encoding Required |
|---|---|
| Decision Tree | Label/Ordinal encoding sufficient |
| Random Forest | Label/Ordinal encoding sufficient |
| XGBoost | Label encoding, or native categorical support (`enable_categorical=True` with `category` dtype) — one-hot not needed |
| LightGBM | Native categorical support (pass column indices/dtype `category`) — no encoding needed |
| CatBoost | Native categorical support — handles raw categorical strings internally via ordered target statistics |

## Linear / distance-based / probabilistic models

Need meaningful numeric encoding.

| Algorithm | Categorical Encoding Required |
|---|---|
| Logistic Regression | One-hot encoding (drop first level to avoid multicollinearity) |
| Linear SVM / SVC | One-hot encoding; also requires feature scaling (StandardScaler) |
| K-Nearest Neighbors (KNN) | One-hot encoding; requires scaling since it's distance-based |
| Naive Bayes (Gaussian) | One-hot encoding; continuous features assumed normally distributed |
| Naive Bayes (Categorical/Multinomial) | Label/ordinal encoding acceptable — works natively on categorical counts |
| Linear Discriminant Analysis (LDA) | One-hot encoding; requires scaling |

## Neural networks

| Algorithm | Categorical Encoding Required |
|---|---|
| MLP / Feedforward NN | One-hot encoding for low-cardinality; embedding layers for high-cardinality categoricals |
| Deep tabular models (TabNet, FT-Transformer) | Embedding layers (learned dense representations) preferred over one-hot |

## Ensemble/boosting variants

| Algorithm | Categorical Encoding Required |
|---|---|
| AdaBoost | Depends on base estimator — one-hot if linear base learner, label encoding if decision stumps |
| Gradient Boosting (sklearn `GradientBoostingClassifier`) | One-hot encoding (no native categorical support, unlike XGBoost/LightGBM) |

## Encoding method quick reference

- **One-hot encoding** — nominal categories, low cardinality (<~15 unique values), no ordinal relationship
- **Label/ordinal encoding** — ordinal categories with true order, or tree-based models where split logic doesn't assume distance
- **Target/mean encoding** — high-cardinality categoricals (e.g. `merchant_id`, `device_id` in fraud/churn work) — must be computed inside CV folds to avoid leakage
- **Frequency encoding** — high-cardinality, when frequency itself is predictive (common in fraud detection for rare-category signals)
- **Embedding encoding** — high-cardinality categoricals in neural network pipelines

## Notes for imbalanced classification (churn/fraud)

Target and frequency encoding tend to outperform one-hot for high-cardinality identifiers, since one-hot on a field like `merchant_id` would blow up dimensionality.
