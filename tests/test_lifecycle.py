import json

import mlflow
import pytest
from typer.testing import CliRunner

from ml_lab.cli import app, evaluate_smoke, train_smoke


def _dirs(tmp_path):
    return {
        "data_dir": tmp_path / "data",
        "model_dir": tmp_path / "model",
        "reports_dir": tmp_path / "reports",
        "tracking_uri": tmp_path / "mlruns",
    }


def test_train_smoke_writes_all_artifacts(tmp_path):
    d = _dirs(tmp_path)
    run_id = train_smoke(**d)
    assert isinstance(run_id, str)
    assert run_id
    assert (d["data_dir"] / "train.csv").exists()
    assert (d["data_dir"] / "test.csv").exists()
    assert (d["data_dir"] / "split_manifest.json").exists()
    assert (d["model_dir"] / "model.joblib").exists()
    assert (d["model_dir"] / "metadata.json").exists()
    assert (d["model_dir"] / "schema.json").exists()
    metadata = json.loads((d["model_dir"] / "metadata.json").read_text())
    assert metadata["mlflow_run_id"] == run_id
    run_id_file = d["reports_dir"] / "train_run_id.txt"
    assert run_id_file.exists()
    assert run_id_file.read_text().strip() == run_id


def test_train_smoke_logs_params_and_metrics(tmp_path):
    d = _dirs(tmp_path)
    run_id = train_smoke(**d)
    client = mlflow.tracking.MlflowClient(tracking_uri=str(d["tracking_uri"]))
    run = client.get_run(run_id)
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.params["feature_count"] == "4"
    assert run.data.metrics["train_accuracy"] >= 0.95
    assert run.data.metrics["train_f1"] >= 0.95


def test_lifecycle_one_run_holds_train_and_eval(tmp_path):
    d = _dirs(tmp_path)
    run_id = train_smoke(**d)
    metrics = evaluate_smoke(**d)
    assert (d["reports_dir"] / "metrics.json").exists()
    assert (d["reports_dir"] / "predictions.csv").exists()
    assert metrics["accuracy"] >= 0.95
    assert metrics["f1"] >= 0.95
    client = mlflow.tracking.MlflowClient(tracking_uri=str(d["tracking_uri"]))
    run = client.get_run(run_id)
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.metrics["train_accuracy"] >= 0.95
    assert run.data.metrics["accuracy"] >= 0.95
    assert run.data.metrics["f1"] >= 0.95


def test_evaluate_smoke_without_train_run_id_raises(tmp_path):
    d = _dirs(tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_smoke(**d)


def test_cli_exposes_train_and_evaluate_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train-smoke" in result.output
    assert "evaluate-smoke" in result.output
