The odd one out — a "backend" with **no network at all**. MLflow's local file store at `./mlruns`, wrapped by `src/ml_lab/tracking/mlflow_utils.py`.

Listed among the integrations because it is a genuine external system with its own semantics and its own failure modes; it just happens to live on disk. There is no tracking server, no database, no credentials. `make mlflow-ui` serves it at 127.0.0.1:5000.

The usage pattern that matters: **one run, written in two passes.** `start_training_run` creates the run and returns its id; `log_evaluation_metrics` reopens **that same run by id** and appends to it. The id crosses the process boundary between `train-smoke` and `evaluate-smoke` through a file on disk (`reports/smoke/latest/train_run_id.txt`) and is also stamped into the model's `metadata.json`.

**Version gotcha:** MLflow 3.14 put the local file store into *maintenance mode* and raises unless `MLFLOW_ALLOW_FILE_STORE` is set. Phase 0 uses the file store deliberately, so the module sets that flag at **import time** — which covers both the app and any test that imports it. If a future MLflow removes the file store outright, this is the module that has to change.
