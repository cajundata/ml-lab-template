# Phase 0 Evaluation Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add positive-class scores to `SmokeModel` and an `evaluate_smoke_model` that loads the saved model + test split, computes metrics, and writes `metrics.json` + `predictions.csv`.

**Architecture:** Two TDD tasks. Task 1 adds `SmokeModel.predict_proba` (reusing the existing feature validation). Task 2 adds `src/ml_lab/evaluation/logistic_eval.py` with `evaluate_smoke_model`, which guards for a missing/mismatched test split, predicts, computes sklearn metrics, and persists artifacts. No MLflow, CLI, or `make evaluate` wiring in this slice.

**Tech Stack:** Python 3.11+, scikit-learn, pandas, numpy, joblib, pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Modify:** `src/ml_lab/models/logistic_smoke.py` — add `predict_proba` to `SmokeModel` (Task 1).
- **Modify:** `tests/test_logistic_smoke_model.py` — add `predict_proba` tests (Task 1). The method's tests live with the model, next to the existing `predict` tests, for cohesion.
- **Create:** `src/ml_lab/evaluation/logistic_eval.py` — `evaluate_smoke_model` (Task 2). `src/ml_lab/evaluation/__init__.py` already exists.
- **Create:** `tests/test_logistic_evaluation.py` — evaluation tests (Task 2).
- **Depends on (already committed, do not modify):** `config.py` (`MODELS_SMOKE_LATEST`, `DATA_PROCESSED_SMOKE`, `REPORTS_SMOKE_LATEST`, `FEATURE_NAMES`, `TARGET_NAME`), `data/synthetic.py` (`write_smoke_data`), `models/logistic_smoke.py` (`SmokeModel`, `SchemaValidationError`, `train_smoke_model`).

pytest is configured with `pythonpath = ["src"]`. `reports/smoke/` and `models/smoke/` are gitignored, so artifacts are never committed. Imports use `from ml_lab.evaluation.logistic_eval import ...`.

---

### Task 1: `SmokeModel.predict_proba`

**Files:**
- Modify: `src/ml_lab/models/logistic_smoke.py`
- Test: `tests/test_logistic_smoke_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_logistic_smoke_model.py` (no new imports needed — `pytest`, `SchemaValidationError`, `train_smoke_model`, `_train_split`, `_test_features` all already exist in this file):

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_logistic_smoke_model.py -k proba -v`
Expected: FAIL — `AttributeError: 'SmokeModel' object has no attribute 'predict_proba'`.

- [ ] **Step 3: Add `predict_proba` to `SmokeModel`**

In `src/ml_lab/models/logistic_smoke.py`, add this method INSIDE the `SmokeModel` class, immediately after `predict` (and before `_validate_features`):

```python
    def predict_proba(self, df: pd.DataFrame):
        self._validate_features(df)
        return self.estimator.predict_proba(
            df[self.schema["feature_names"]]
        )[:, 1]
```

(`pd` is already imported in this module. The estimator is trained on labels `[0, 1]`, so column `1` of `predict_proba` is `P(y = 1)`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logistic_smoke_model.py -v`
Expected: PASS — all existing model tests plus the three new `predict_proba` tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/models/logistic_smoke.py tests/test_logistic_smoke_model.py
git commit -m "feat: positive-class predict_proba on SmokeModel"
```

---

### Task 2: `evaluate_smoke_model`

**Files:**
- Create: `src/ml_lab/evaluation/logistic_eval.py`
- Test: `tests/test_logistic_evaluation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logistic_evaluation.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_logistic_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.evaluation.logistic_eval'`.

- [ ] **Step 3: Create `logistic_eval.py`**

Create `src/ml_lab/evaluation/logistic_eval.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logistic_evaluation.py -v`
Expected: PASS — all eight evaluation tests green.

- [ ] **Step 5: Sanity-check against the real artifacts (optional, non-committed)**

Run:
```bash
uv run python -c "from ml_lab.data.synthetic import write_smoke_data; from ml_lab.models.logistic_smoke import train_smoke_model; import pandas as pd; write_smoke_data(); tr=pd.read_csv('data/processed/smoke/train.csv'); train_smoke_model(tr).save()"
uv run python -c "from ml_lab.evaluation.logistic_eval import evaluate_smoke_model; import json; print(json.dumps(evaluate_smoke_model(), indent=2))"
```
Expected: prints the metrics dict (accuracy/f1 well above 0.95); `reports/smoke/latest/{metrics.json,predictions.csv}` exist locally (gitignored).

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/evaluation/logistic_eval.py tests/test_logistic_evaluation.py
git commit -m "feat: evaluate smoke model and write metrics + predictions"
```

---

## Definition of Done

- `uv run pytest` is fully green (the model, data, and new evaluation tests).
- `SmokeModel.predict_proba` returns validated positive-class probabilities.
- `evaluate_smoke_model` writes `metrics.json` (accuracy, precision, recall, f1, 2×2 confusion_matrix, evaluated_row_count, `mlflow_run_id: null`) and `predictions.csv` (`y_true,y_pred,y_score`), and returns the metrics dict.
- Raises `FileNotFoundError` for a missing test split and `SchemaValidationError` for a schema mismatch.
- No MLflow, `train_run_id.txt`, CLI, or `make evaluate` added.
