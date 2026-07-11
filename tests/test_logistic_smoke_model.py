from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml_lab.config import FEATURE_NAMES, TARGET_NAME
from ml_lab.data.synthetic import generate_smoke_data, split_smoke_data
from ml_lab.models.logistic_smoke import (
    MODEL_NAME,
    MODEL_VERSION,
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
