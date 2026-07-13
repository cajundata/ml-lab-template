`src/ml_lab/evaluation/logistic_eval.py` — one function, `evaluate_smoke_model()`, which loads the *saved* model and the *saved* test split from disk rather than receiving them in memory. That is deliberate: it evaluates what was actually persisted, so a broken `save()`/`load()` round-trip cannot pass silently.

It fails loud in two directions before computing anything — a missing `test.csv` raises `FileNotFoundError` ("run training first"), and a test split missing feature or target columns raises `SchemaValidationError` naming exactly which ones.

Outputs are accuracy, precision, recall, F1, the confusion matrix, the evaluated row count, and the `mlflow_run_id` carried in from the model's metadata. It writes `metrics.json` and a per-row `predictions.csv` (`y_true`, `y_pred`, `y_score`), then returns the metrics dict so `cli.evaluate_smoke()` can push the four scalar metrics into the existing MLflow run.
