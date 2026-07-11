from datetime import datetime

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml_lab.config import FEATURE_NAMES, TARGET_NAME
from ml_lab.data.synthetic import generate_smoke_data, split_smoke_data
from ml_lab.models.logistic_smoke import (
    MODEL_NAME,
    MODEL_VERSION,
    SchemaValidationError,
    SmokeModel,
    train_smoke_model,
)


def _train_split():
    train_df, _ = split_smoke_data(generate_smoke_data())
    return train_df


def test_train_returns_smoke_model_with_fitted_logreg():
    model = train_smoke_model(_train_split())
    assert isinstance(model, SmokeModel)
    assert isinstance(model.estimator, LogisticRegression)
    assert hasattr(model.estimator, "coef_")


def test_train_is_deterministic():
    a = train_smoke_model(_train_split())
    b = train_smoke_model(_train_split())
    np.testing.assert_array_equal(a.estimator.coef_, b.estimator.coef_)


def test_schema_contents():
    schema = train_smoke_model(_train_split()).schema
    assert schema["feature_names"] == list(FEATURE_NAMES)
    assert set(schema["dtypes"].keys()) == set(FEATURE_NAMES)
    assert schema["target_name"] == TARGET_NAME
    assert schema["target_labels"] == [0, 1]


def test_metadata_contents():
    md = train_smoke_model(_train_split()).metadata
    assert md["model_name"] == MODEL_NAME
    assert md["model_type"] == "LogisticRegression"
    assert md["version"] == MODEL_VERSION
    assert md["features"] == list(FEATURE_NAMES)
    assert md["target"] == TARGET_NAME
    assert md["mlflow_run_id"] is None
    parsed = datetime.fromisoformat(md["created_at"])
    assert parsed.tzinfo is not None


def _test_features():
    _, test_df = split_smoke_data(generate_smoke_data())
    return test_df[list(FEATURE_NAMES)]


def test_predict_count_matches_input_rows():
    model = train_smoke_model(_train_split())
    features = _test_features()
    preds = model.predict(features)
    assert len(preds) == len(features)


def test_predict_is_binary():
    model = train_smoke_model(_train_split())
    preds = model.predict(_test_features())
    assert set(preds).issubset({0, 1})


def test_predict_rejects_missing_column():
    model = train_smoke_model(_train_split())
    bad = _test_features()[["x0", "x1", "x2"]]
    with pytest.raises(SchemaValidationError):
        model.predict(bad)


def test_predict_rejects_extra_column():
    model = train_smoke_model(_train_split())
    bad = _test_features().copy()
    bad["x4"] = 0.0
    with pytest.raises(SchemaValidationError):
        model.predict(bad)


def test_predict_rejects_wrong_order():
    model = train_smoke_model(_train_split())
    bad = _test_features()[["x1", "x0", "x2", "x3"]]
    with pytest.raises(SchemaValidationError):
        model.predict(bad)


def test_predict_rejects_nonnumeric():
    model = train_smoke_model(_train_split())
    bad = _test_features().copy()
    bad["x0"] = "not a number"
    with pytest.raises(SchemaValidationError):
        model.predict(bad)


def test_save_creates_artifact_files(tmp_path):
    model = train_smoke_model(_train_split())
    model.save(dest=tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "schema.json").exists()
    assert (tmp_path / "metadata.json").exists()


def test_load_round_trips_predictions(tmp_path):
    model = train_smoke_model(_train_split())
    features = _test_features()
    before = model.predict(features)
    model.save(dest=tmp_path)
    reloaded = SmokeModel.load(dest=tmp_path)
    after = reloaded.predict(features)
    assert (before == after).all()


def test_load_restores_schema_and_metadata(tmp_path):
    model = train_smoke_model(_train_split())
    model.save(dest=tmp_path)
    reloaded = SmokeModel.load(dest=tmp_path)
    assert reloaded.schema == model.schema
    assert reloaded.metadata == model.metadata


def test_predict_proba_length_matches_rows():
    model = train_smoke_model(_train_split())
    features = _test_features()
    proba = model.predict_proba(features)
    assert len(proba) == len(features)


def test_predict_proba_values_in_unit_interval():
    model = train_smoke_model(_train_split())
    proba = model.predict_proba(_test_features())
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_predict_proba_rejects_bad_frame():
    model = train_smoke_model(_train_split())
    bad = _test_features()[["x0", "x1", "x2"]]
    with pytest.raises(SchemaValidationError):
        model.predict_proba(bad)
