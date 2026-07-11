import os

import mlflow

from ml_lab.config import EXPERIMENT_NAME, MLRUNS_DIR

# MLflow 3.14 put the local file store (./mlruns) into maintenance mode and
# raises unless this documented opt-out is set. Phase 0 deliberately uses the
# local file store, so opt in here (covers both the app and tests importing
# this module).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def start_training_run(
    params: dict,
    metrics: dict,
    tracking_uri=MLRUNS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Create an MLflow run, log params + metrics, and return its run id."""
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    return run_id
