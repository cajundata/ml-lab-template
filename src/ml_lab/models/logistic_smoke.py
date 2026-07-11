import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_lab.config import (
    FEATURE_NAMES,
    MODELS_SMOKE_LATEST,
    SMOKE_SEED,
    TARGET_NAME,
)

MODEL_NAME = "logistic_smoke"
MODEL_VERSION = "0.1.0"


class SchemaValidationError(ValueError):
    """Raised when input features do not match the model schema."""


class SmokeModel:
    """Fitted logistic smoke model plus its schema and metadata."""

    def __init__(self, estimator: LogisticRegression, schema: dict, metadata: dict):
        self.estimator = estimator
        self.schema = schema
        self.metadata = metadata

    def predict(self, df: pd.DataFrame):
        self._validate_features(df)
        return self.estimator.predict(df[self.schema["feature_names"]])

    def _validate_features(self, df: pd.DataFrame) -> None:
        expected = self.schema["feature_names"]
        if list(df.columns) != expected:
            raise SchemaValidationError(
                f"Feature columns {list(df.columns)} do not match "
                f"schema {expected}"
            )
        for col in expected:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise SchemaValidationError(
                    f"Feature column '{col}' must be numeric, "
                    f"got dtype {df[col].dtype}"
                )

    def save(self, dest: Path = MODELS_SMOKE_LATEST) -> None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, dest / "model.joblib")
        (dest / "schema.json").write_text(
            json.dumps(self.schema, indent=2) + "\n", encoding="utf-8"
        )
        (dest / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, dest: Path = MODELS_SMOKE_LATEST) -> "SmokeModel":
        dest = Path(dest)
        estimator = joblib.load(dest / "model.joblib")
        schema = json.loads((dest / "schema.json").read_text(encoding="utf-8"))
        metadata = json.loads(
            (dest / "metadata.json").read_text(encoding="utf-8")
        )
        return cls(estimator=estimator, schema=schema, metadata=metadata)


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
