import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml_lab.config import (
    DATA_PROCESSED_SMOKE,
    FEATURE_NAMES,
    MODELS_SMOKE_LATEST,
    REPORTS_SMOKE_LATEST,
    TARGET_NAME,
)
from ml_lab.models.logistic_smoke import SchemaValidationError, SmokeModel


def evaluate_smoke_model(
    model_dir: Path = MODELS_SMOKE_LATEST,
    data_dir: Path = DATA_PROCESSED_SMOKE,
    reports_dir: Path = REPORTS_SMOKE_LATEST,
) -> dict:
    """Evaluate the saved smoke model against the saved test split."""
    data_dir = Path(data_dir)
    test_path = data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_path}; run training first."
        )

    model = SmokeModel.load(model_dir)
    test_df = pd.read_csv(test_path)

    missing = [c for c in FEATURE_NAMES if c not in test_df.columns]
    if TARGET_NAME not in test_df.columns:
        missing = missing + [TARGET_NAME]
    if missing:
        raise SchemaValidationError(
            f"Test data does not match schema; missing columns: {missing}"
        )

    features = test_df[FEATURE_NAMES]
    y_true = test_df[TARGET_NAME]
    y_pred = model.predict(features)
    y_score = model.predict_proba(features)

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": [[int(v) for v in row] for row in cm],
        "evaluated_row_count": int(len(test_df)),
        "mlflow_run_id": model.metadata["mlflow_run_id"],
    }

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred,
            "y_score": y_score,
        }
    ).to_csv(reports_dir / "predictions.csv", index=False)

    return metrics
