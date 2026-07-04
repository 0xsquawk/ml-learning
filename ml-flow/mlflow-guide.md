# MLflow Learning Guide

A practical reference for tracking experiments, logging models, and managing the ML lifecycle with MLflow — including syntax for common algorithms (scikit-learn, XGBoost, CatBoost, LightGBM).

---

## Table of Contents

1. [What is MLflow](#what-is-mlflow)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Experiment Tracking Basics](#experiment-tracking-basics)
5. [Autologging](#autologging)
6. [Manual Logging by Algorithm](#manual-logging-by-algorithm)
7. [Model Registry](#model-registry)
8. [Loading & Serving Models](#loading--serving-models)
9. [MLflow Projects](#mlflow-projects)
10. [Useful CLI Commands](#useful-cli-commands)
11. [Best Practices](#best-practices)

---

## What is MLflow

MLflow is an open-source platform for managing the ML lifecycle. Four main components:

| Component | Purpose |
|---|---|
| **Tracking** | Log parameters, metrics, artifacts, and code versions per run |
| **Projects** | Package code in a reusable, reproducible format |
| **Models** | Standard format for packaging models for deployment |
| **Model Registry** | Centralized store for model versioning, staging, and lifecycle |

---

## Installation & Setup

```bash
pip install mlflow

# Optional extras
pip install mlflow[extras]        # adds serving, R support, etc.
pip install xgboost catboost lightgbm scikit-learn
```

Start the tracking UI (default: local `mlruns/` folder):

```bash
mlflow ui --port 5000
# visit http://localhost:5000
```

Point to a remote tracking server (e.g., a shared server or SQLite backend):

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
# or a file/db backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

---

## Core Concepts

| Term | Meaning |
|---|---|
| **Run** | A single execution of model training code |
| **Experiment** | A named collection of runs |
| **Parameters** | Inputs to your training code (hyperparameters) |
| **Metrics** | Outputs measured during/after training (accuracy, PR-AUC, etc.) |
| **Artifacts** | Output files (models, plots, data samples) |
| **Tags** | Key-value metadata for organizing runs |

```python
mlflow.set_experiment("ravenstack-churn-model")

with mlflow.start_run(run_name="xgb-baseline"):
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", value)
    mlflow.log_artifact("path/to/file")
    mlflow.set_tag("stage", "baseline")
```

---

## Experiment Tracking Basics

```python
import mlflow

mlflow.set_experiment("my-experiment")

with mlflow.start_run(run_name="run-01"):
    # single param/metric
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("pr_auc", 0.87)

    # batch logging (preferred for many values)
    mlflow.log_params({"n_estimators": 300, "max_depth": 6})
    mlflow.log_metrics({"precision": 0.81, "recall": 0.76})

    # log at a specific training step (for curves)
    for epoch, loss in enumerate(loss_history):
        mlflow.log_metric("loss", loss, step=epoch)

    # artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifacts("output_dir/")   # log a whole directory

    # tags
    mlflow.set_tag("model_type", "XGBoost")
    mlflow.set_tags({"dataset": "ravenstack_v2", "owner": "charchit"})
```

Nested runs (useful for cross-validation folds or hyperparameter sweeps):

```python
with mlflow.start_run(run_name="parent-sweep"):
    for fold in range(5):
        with mlflow.start_run(run_name=f"fold-{fold}", nested=True):
            mlflow.log_metric("fold_auc", fold_score)
```

---

## Autologging

The fastest way to get started — MLflow automatically logs params, metrics, and the model artifact for supported frameworks.

```python
import mlflow

mlflow.autolog()  # generic, detects the framework in use

# or framework-specific (more control, recommended for production)
mlflow.sklearn.autolog()
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()
mlflow.catboost.autolog()
```

Common `autolog()` parameters (available across most flavors):

```python
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    silent=False,
    max_tuning_runs=5,          # for GridSearchCV/RandomizedSearchCV
)
```

> Note: autologging must be called **before** `.fit()` and typically before `mlflow.start_run()` (it will create its own run if none is active).

---

## Manual Logging by Algorithm

Manual logging gives full control — useful when autolog doesn't capture custom metrics (e.g., PR-AUC on an imbalanced fraud dataset).

### Scikit-learn (e.g., Logistic Regression, RandomForest)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

with mlflow.start_run(run_name="rf-model"):
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
    }
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    mlflow.log_metric("pr_auc", auc(recall, precision))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
        registered_model_name="churn-rf-model",  # optional: registers directly
    )
```

### XGBoost

```python
import mlflow
import mlflow.xgboost
import xgboost as xgb

with mlflow.start_run(run_name="xgb-fraud-model"):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "eta": 0.05,
        "scale_pos_weight": 65,   # for ~1.5% fraud rate imbalance
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    mlflow.log_params(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=False,
    )

    # log per-round metrics as a curve
    for i, val in enumerate(evals_result["valid"]["aucpr"]):
        mlflow.log_metric("valid_aucpr", val, step=i)

    mlflow.log_metric("best_iteration", booster.best_iteration)

    mlflow.xgboost.log_model(
        xgb_model=booster,
        artifact_path="model",
        registered_model_name="fraud-xgb-model",
    )
```

Using the scikit-learn wrapper API instead:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=65,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    use_label_encoder=False,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
```

### CatBoost

```python
import mlflow
from catboost import CatBoostClassifier

with mlflow.start_run(run_name="catboost-baseline"):
    params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "class_weights": [1, 65],
        "random_seed": 42,
        "verbose": False,
    }
    mlflow.log_params(params)

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_feature_indices,
        early_stopping_rounds=30,
    )

    mlflow.log_metric("best_score", model.get_best_score()["validation"]["PRAUC"])

    mlflow.catboost.log_model(model, artifact_path="model")
```

### LightGBM

```python
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

with mlflow.start_run(run_name="lgbm-model"):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "scale_pos_weight": 65,
        "seed": 42,
    }
    mlflow.log_params(params)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    mlflow.log_metric("best_iteration", model.best_iteration)
    mlflow.lightgbm.log_model(model, artifact_path="model")
```

### Prophet (time series forecasting)

```python
import mlflow
from prophet import Prophet
from prophet.serialize import model_to_json

with mlflow.start_run(run_name="prophet-forecast"):
    params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
    }
    mlflow.log_params(params)

    model = Prophet(**params)
    model.fit(df_train)

    forecast = model.predict(df_future)
    mape = compute_mape(df_test["y"], forecast["yhat"])
    mlflow.log_metric("mape", mape)

    # Prophet models need custom serialization
    with open("prophet_model.json", "w") as f:
        f.write(model_to_json(model))
    mlflow.log_artifact("prophet_model.json")
```

### Hyperparameter Sweeps (GridSearchCV / Optuna)

```python
import mlflow
from sklearn.model_selection import GridSearchCV

mlflow.sklearn.autolog(max_tuning_runs=10)

with mlflow.start_run(run_name="grid-search-parent"):
    grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={"n_estimators": [100, 200], "max_depth": [5, 10, None]},
        scoring="average_precision",
        cv=5,
    )
    grid.fit(X_train, y_train)
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_cv_score", grid.best_score_)
```

With Optuna:

```python
import mlflow
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
    }
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_valid, y_valid)
        mlflow.log_metric("pr_auc", score)
    return score

with mlflow.start_run(run_name="optuna-sweep"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_pr_auc", study.best_value)
```

---

## Model Registry

Register, version, and stage models for deployment.

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# Register a logged model
result = mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="churn-xgb-model",
)

# Transition stages (Staging -> Production -> Archived)
client.transition_model_version_stage(
    name="churn-xgb-model",
    version=result.version,
    stage="Staging",
)

# Add description / tags
client.update_model_version(
    name="churn-xgb-model",
    version=result.version,
    description="XGBoost model trained on Q2 churn dataset, PR-AUC 0.89",
)
```

> Newer MLflow versions (2.9+) favor **Aliases** over stage names:

```python
client.set_registered_model_alias(
    name="churn-xgb-model",
    alias="champion",
    version=result.version,
)
```

---

## Loading & Serving Models

```python
import mlflow

# Load by run ID
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# Load by registered name + version/stage/alias
model = mlflow.pyfunc.load_model("models:/churn-xgb-model/1")
model = mlflow.pyfunc.load_model("models:/churn-xgb-model@champion")

predictions = model.predict(X_new)
```

Serve as a local REST API:

```bash
mlflow models serve -m "models:/churn-xgb-model/1" --port 1234 --no-conda
```

```bash
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["feature_1","feature_2"], "data": [[0.5, 1.2]]}}'
```

Build a Docker image:

```bash
mlflow models build-docker -m "models:/churn-xgb-model/1" -n churn-model-image
```

---

## MLflow Projects

Package reproducible runs with an `MLproject` file:

```yaml
name: ravenstack-churn

python_env: python_env.yaml

entry_points:
  main:
    parameters:
      max_depth: {type: int, default: 6}
      learning_rate: {type: float, default: 0.05}
    command: "python train.py --max_depth {max_depth} --learning_rate {learning_rate}"
```

Run it:

```bash
mlflow run . -P max_depth=8 -P learning_rate=0.03
mlflow run https://github.com/user/repo.git -P max_depth=8
```

---

## Useful CLI Commands

```bash
mlflow ui                                  # launch tracking UI
mlflow experiments list                    # list experiments
mlflow runs list --experiment-id 1         # list runs in an experiment
mlflow runs describe --run-id <id>         # inspect a single run
mlflow models serve -m <model_uri>         # serve a model
mlflow gc                                  # remove deleted runs permanently
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

---

## Best Practices

- **Name experiments and runs meaningfully** — e.g. `fraud-xgb-scale_pos_weight_sweep` rather than `run1`.
- **Log the evaluation metric that matches your problem** — PR-AUC over accuracy for highly imbalanced classes like fraud (~1.5% positive rate).
- **Always log an `input_example`** when calling `log_model()` — it enables automatic signature inference and makes serving errors easier to debug.
- **Use nested runs** for CV folds and hyperparameter sweeps to keep the parent run as the summary.
- **Set a remote/shared tracking URI** early if working in a team, so runs are visible to collaborators instead of siloed in local `mlruns/` folders.
- **Register only validated models** — treat the registry as the source of truth for what's deployable, not a dumping ground for every run.
- **Version your training data** alongside the model run (log a hash, path, or `mlflow.log_input()` dataset reference) so experiments stay reproducible.
- **Prefer aliases over legacy stage names** (`@champion`, `@challenger`) — Stages are being phased out in newer MLflow releases.

---

## Reference Links

- Official docs: https://mlflow.org/docs/latest/index.html
- Python API reference: https://mlflow.org/docs/latest/python_api/index.html
- Model flavors: https://mlflow.org/docs/latest/models.html#built-in-model-flavors
