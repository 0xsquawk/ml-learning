# Supervised ML Algorithms & Metrics Reference

Industry-preferred ranges for classification and regression algorithms.

## Classification

| Algorithm | Key Metrics & Preferred Ranges | Notes |
|---|---|---|
| Logistic Regression | AUC-ROC ≥ 0.80 · F1 ≥ 0.75 · Accuracy ≥ 85% · Log Loss < 0.50 | Interpretable baseline; weakens on imbalanced data |
| Decision Tree | Accuracy ≥ 82% · F1 ≥ 0.72 · AUC-ROC ≥ 0.78 · Gini Impurity: lower = better | Prone to overfit; excellent for rule extraction; prune aggressively |
| Random Forest | AUC-ROC ≥ 0.87 · F1 ≥ 0.80 · Accuracy ≥ 88% · OOB Error < 10% | Robust ensemble; built-in feature importance; slower at scale |
| XGBoost / LightGBM | AUC-ROC ≥ 0.90 · PR-AUC ≥ 0.72 · F1 ≥ 0.82 · Log Loss < 0.40 | Industry gold standard for tabular data; tune regularization to avoid overfit |
| Support Vector Machine | Accuracy ≥ 85% · F1 ≥ 0.80 · AUC-ROC ≥ 0.85 | Excellent on high-dimensional data; slow training at large scale; kernel choice critical |
| Neural Network (MLP) | AUC-ROC ≥ 0.90 · PR-AUC ≥ 0.74 · F1 ≥ 0.83 · Log Loss < 0.35 | High capacity; needs large data, regularization, careful tuning |
| Naive Bayes | Accuracy ≥ 75% · F1 ≥ 0.70 · AUC-ROC ≥ 0.75 | Fast and simple; assumes feature independence; strong text-classification baseline |
| K-Nearest Neighbors | Accuracy ≥ 80% · F1 ≥ 0.75 · AUC-ROC ≥ 0.78 | No training phase; inference slow at scale; feature scaling is mandatory |

## Regression

| Algorithm | Key Metrics & Preferred Ranges | Notes |
|---|---|---|
| Linear Regression | R² ≥ 0.75 · MAPE < 20% · MAE / RMSE: domain-dependent | Interpretable baseline; assumes linearity and normal residuals |
| Ridge / Lasso | R² ≥ 0.78 · MAPE < 18% · RMSE: lower vs. OLS | Ridge handles multicollinearity; Lasso performs feature selection via sparsity |
| Decision Tree Regressor | R² ≥ 0.76 · MAPE < 22% · MAE: domain-dependent | Captures non-linearity; high variance without pruning; interpretable splits |
| Random Forest Regressor | R² ≥ 0.85 · MAPE < 15% · MAE: lowest in class | Handles non-linearity and outliers well; feature importance built in |
| XGBoost Regressor | R² ≥ 0.88 · MAPE < 10% · RMSE: lowest in class | Top performer on structured regression; early stopping prevents overfit |
| Support Vector Regressor | R² ≥ 0.80 · MAPE < 18% | ε-insensitive loss gives robustness to outliers; kernel selection is key |

## Range Key

- **Good** — target commonly met by well-tuned production models
- **Acceptable** — within tolerance, may need improvement depending on use case
- **Lower = better** — minimize this metric
- **Domain-dependent** — no universal threshold; judge against your specific business context
