import pandas as pd
import pytest

from ml_lab.data.synthetic import write_smoke_data
from ml_lab.evaluation.logistic_eval import evaluate_smoke_model
from ml_lab.models.logistic_smoke import (
    SchemaValidationError,
    train_smoke_model,
)


def _setup(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    reports_dir = tmp_path / "reports"
    write_smoke_data(dest=data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    train_smoke_model(train_df).save(dest=model_dir)
    return model_dir, data_dir, reports_dir


def test_metrics_and_predictions_files_created(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    evaluate_smoke_model(model_dir, data_dir, reports_dir)
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "predictions.csv").exists()


def test_metric_thresholds(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    metrics = evaluate_smoke_model(model_dir, data_dir, reports_dir)
    assert metrics["accuracy"] >= 0.95
    assert metrics["f1"] >= 0.95


def test_precision_recall_numeric_in_range(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    metrics = evaluate_smoke_model(model_dir, data_dir, reports_dir)
    for key in ("precision", "recall"):
        assert isinstance(metrics[key], float)
        assert 0.0 <= metrics[key] <= 1.0


def test_confusion_matrix_is_2x2(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    metrics = evaluate_smoke_model(model_dir, data_dir, reports_dir)
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2
    assert all(len(row) == 2 for row in cm)


def test_evaluated_row_count(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    metrics = evaluate_smoke_model(model_dir, data_dir, reports_dir)
    assert metrics["evaluated_row_count"] == 60


def test_predictions_csv_shape_and_columns(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    evaluate_smoke_model(model_dir, data_dir, reports_dir)
    preds = pd.read_csv(reports_dir / "predictions.csv")
    assert list(preds.columns) == ["y_true", "y_pred", "y_score"]
    assert len(preds) == 60


def test_missing_test_split_raises(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    (data_dir / "test.csv").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_smoke_model(model_dir, data_dir, reports_dir)


def test_schema_mismatch_raises(tmp_path):
    model_dir, data_dir, reports_dir = _setup(tmp_path)
    df = pd.read_csv(data_dir / "test.csv").drop(columns=["x3"])
    df.to_csv(data_dir / "test.csv", index=False)
    with pytest.raises(SchemaValidationError):
        evaluate_smoke_model(model_dir, data_dir, reports_dir)
