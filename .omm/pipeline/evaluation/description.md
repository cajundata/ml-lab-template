**Stage 3** — `src/ml_lab/evaluation/logistic_eval.py`, driven by `cli.evaluate_smoke()`.

It loads the **saved** model and the **saved** test split *from disk* rather than receiving them in memory. That is deliberate: it evaluates what was actually persisted, so a broken `save()`/`load()` round-trip cannot pass silently.

Fails loud in two directions before computing anything — a missing `test.csv` raises `FileNotFoundError` ("run training first"), and a test split missing feature or target columns raises `SchemaValidationError` naming exactly which ones.

Computes accuracy, precision, recall, F1, the confusion matrix, the evaluated row count, and carries through the `mlflow_run_id` from the model's metadata. Writes `metrics.json` and a per-row `predictions.csv` (`y_true`, `y_pred`, `y_score`) — the per-row file matters because aggregate metrics on 60 rows hide everything interesting.

Then `cli.evaluate_smoke()` **reopens the training run by id** and appends the four scalar metrics to it.

**Outputs:** `reports/smoke/latest/{metrics.json, predictions.csv}`.
