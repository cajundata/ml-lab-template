# Reports directory

This directory holds generated evaluation outputs. Nothing here is committed except this README and the `.gitkeep` placeholder. Reports are produced mechanically from a saved model and a saved test split, so they are always reproducible.

## Layout

In Phase 0, the smoke lifecycle writes to `reports/smoke/latest/`:

- `train_run_id.txt` is written by `make train` and holds the MLflow run id for the lifecycle. `make evaluate` reads this file and logs evaluation metrics to that same run.
- `metrics.json` is written by `make evaluate` and records accuracy, precision, recall, F1, the confusion matrix, the evaluated row count, and the model run id.
- `predictions.csv` is written by `make evaluate` and contains `y_true`, `y_pred`, and `y_score` columns, one row per evaluated test row.

`make clean` removes the contents of `reports/smoke/`. Regenerate with `make train` followed by `make evaluate`.