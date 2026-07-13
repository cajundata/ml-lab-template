`src/ml_lab/tracking/mlflow_utils.py` — 38 lines wrapping MLflow's local file store (`./mlruns`).

Two functions, and the relationship between them is the whole design:
- `start_training_run(params, metrics)` creates a run, logs training params + metrics, and **returns the run id**.
- `log_evaluation_metrics(run_id, metrics)` **reopens that same run by id** and appends the evaluation metrics to it.

So training and evaluation are not two runs — they are one run, written in two passes. The run id is handed between the two CLI invocations through a file on disk: `reports/smoke/latest/train_run_id.txt`. `evaluate_smoke` refuses to run if that file is absent ("run train first"), and the id is also stamped into the model's `metadata.json`, so a saved model can always be traced back to the run that produced it.

Note the `MLFLOW_ALLOW_FILE_STORE` opt-in at import time: MLflow 3.14 put the local file store into maintenance mode and raises without it. Phase 0 uses the file store deliberately, so the module sets the flag for both the app and any test that imports it.
