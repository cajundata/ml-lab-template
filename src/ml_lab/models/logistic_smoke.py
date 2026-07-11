from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_SEED,
    TARGET_NAME,
)

MODEL_NAME = "logistic_smoke"
MODEL_VERSION = "0.1.0"


class SmokeModel:
    """Fitted logistic smoke model plus its schema and metadata."""

    def __init__(self, estimator: LogisticRegression, schema: dict, metadata: dict):
        self.estimator = estimator
        self.schema = schema
        self.metadata = metadata


def _build_schema(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "dtypes": {col: str(X[col].dtype) for col in FEATURE_NAMES},
        "target_name": TARGET_NAME,
        "target_labels": [int(v) for v in sorted(y.unique())],
    }


def _build_metadata(estimator) -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_type": type(estimator).__name__,
        "version": MODEL_VERSION,
        "features": list(FEATURE_NAMES),
        "target": TARGET_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": None,
    }


def train_smoke_model(train_df: pd.DataFrame) -> SmokeModel:
    """Fit LogisticRegression on the train split; return a SmokeModel."""
    X = train_df[FEATURE_NAMES]
    y = train_df[TARGET_NAME]
    estimator = LogisticRegression(random_state=SMOKE_SEED)
    estimator.fit(X, y)
    schema = _build_schema(X, y)
    metadata = _build_metadata(estimator)
    return SmokeModel(estimator=estimator, schema=schema, metadata=metadata)
