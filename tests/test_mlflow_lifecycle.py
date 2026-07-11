import mlflow

from ml_lab.config import EXPERIMENT_NAME
from ml_lab.tracking.mlflow_utils import start_training_run

PARAMS = {"model_type": "LogisticRegression", "random_seed": 20260706}
TRAIN_METRICS = {"train_accuracy": 1.0, "train_f1": 1.0}


def test_start_training_run_returns_run_id(tmp_path):
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tmp_path / "mlruns",
        experiment_name=EXPERIMENT_NAME,
    )
    assert isinstance(run_id, str)
    assert run_id


def test_start_training_run_logs_params_and_metrics(tmp_path):
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tmp_path / "mlruns",
        experiment_name=EXPERIMENT_NAME,
    )
    run = mlflow.get_run(run_id)
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.params["random_seed"] == "20260706"
    assert run.data.metrics["train_accuracy"] == 1.0
    assert run.data.metrics["train_f1"] == 1.0


def test_start_training_run_uses_named_experiment(tmp_path):
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tmp_path / "mlruns",
        experiment_name=EXPERIMENT_NAME,
    )
    run = mlflow.get_run(run_id)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert run.info.experiment_id == experiment.experiment_id
