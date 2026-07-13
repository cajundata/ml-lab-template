`src/ml_lab/models/logistic_smoke.py` — a `LogisticRegression` and, more importantly, the `SmokeModel` wrapper that keeps the estimator, its **schema**, and its **metadata** together as one artifact.

The schema is not decoration. `predict()` and `predict_proba()` both call `_validate_features()` first, which raises `SchemaValidationError` unless the column list matches *exactly* (order included) and every feature column is numeric. A model that silently accepts reordered columns is a model that silently returns garbage.

`save()` writes three files as a unit — `model.joblib`, `schema.json`, `metadata.json` — and `load()` reads all three back. Metadata carries `model_name`, `model_type`, `version` (`0.1.0`), the feature/target names, `created_at`, and `mlflow_run_id`. That last field starts as `None` and is stamped by `train_smoke()` *after* the MLflow run is created, which is what lets any saved model be traced back to the run that produced it.

This is the Phase-1 seam: swap the estimator, keep the wrapper.
