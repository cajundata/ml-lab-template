import mlflow

from ml_lab.config import EXPERIMENT_NAME
from ml_lab.tracking.mlflow_utils import (
    log_evaluation_metrics,
    start_training_run,
)

PARAMS = {"model_type": "LogisticRegression", "random_seed": 20260706}
TRAIN_METRICS = {"train_accuracy": 1.0, "train_f1": 1.0}
EVAL_METRICS = {"accuracy": 1.0, "f1": 1.0}


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
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    run = client.get_run(run_id)
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
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    run = client.get_run(run_id)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert run.info.experiment_id == experiment.experiment_id


def test_log_evaluation_metrics_appends_to_run(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )
    log_evaluation_metrics(run_id, EVAL_METRICS, tracking_uri=tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_uri))
    run = client.get_run(run_id)
    assert run.data.metrics["accuracy"] == 1.0
    assert run.data.metrics["f1"] == 1.0


def test_lifecycle_one_run_holds_train_and_eval(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )
    log_evaluation_metrics(run_id, EVAL_METRICS, tracking_uri=tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_uri))
    run = client.get_run(run_id)
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.metrics["train_accuracy"] == 1.0
    assert run.data.metrics["accuracy"] == 1.0


def test_log_evaluation_metrics_creates_no_second_run(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )
    log_evaluation_metrics(run_id, EVAL_METRICS, tracking_uri=tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_uri))
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
