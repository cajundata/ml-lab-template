**One run, two passes.** This is the piece of the pipeline most worth understanding, and the piece most likely to be broken by a well-meaning refactor.

`src/ml_lab/tracking/mlflow_utils.py` is 38 lines with two functions:
- **`start_training_run(params, metrics)`** — creates a run, logs the training params and metrics, **returns the run id**.
- **`log_evaluation_metrics(run_id, metrics)`** — **reopens that same run by id** and appends the evaluation metrics to it.

So `train-smoke` and `evaluate-smoke` — two separate CLI invocations, two separate processes — write to **one** MLflow run. The run id travels between them through a file on disk (`reports/smoke/latest/train_run_id.txt`) and is simultaneously stamped into the model's `metadata.json`.

The payoff is a single MLflow row that tells the whole story of a model: the seed and checksummed row counts it trained on, its params, its training metrics, and its held-out metrics. And it round-trips — from a saved `model.joblib` you can find the run; from the run you can find the exact bytes.

Two runs would break that trace. Keep it one.

**Backend:** the local file store at `./mlruns` (`make mlflow-ui` serves it at 127.0.0.1:5000). MLflow 3.14 put the file store into maintenance mode and raises without an opt-out, so the module sets `MLFLOW_ALLOW_FILE_STORE` at import time — Phase 0 uses it deliberately.
