**Stage 2** — `src/ml_lab/models/logistic_smoke.py`, driven by `cli.train_smoke()`.

Fits `LogisticRegression(random_state=SMOKE_SEED)` on `train.csv` and wraps it in a **`SmokeModel`** — the estimator, its **schema**, and its **metadata**, kept together as one artifact.

The schema is enforced, not decorative: `predict()` and `predict_proba()` both validate first, raising `SchemaValidationError` unless the column list matches *exactly* (order included) and every feature is numeric.

`save()` writes three files as a unit — `model.joblib`, `schema.json`, `metadata.json`. Metadata carries `model_name`, `model_type`, `version`, features, target, `created_at`, and `mlflow_run_id`. That last field starts `None` and is **stamped after the MLflow run is created**, which is what lets any saved model be traced back to the run that produced it.

The stage also scores the model on its **own training split** (`train_accuracy`, `train_f1`) and logs those with the params. That is a smoke signal, not a generalization estimate — the held-out numbers come from stage 3.

**Outputs:** `models/smoke/latest/{model.joblib, schema.json, metadata.json}` + `reports/smoke/latest/train_run_id.txt`.

**This is the Phase-1 seam: swap the estimator, keep the wrapper.**
