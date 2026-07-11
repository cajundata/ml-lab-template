# Phase 0 Smoke Model Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the deterministic logistic smoke model, wrap it in a schema-validating `SmokeModel`, and persist `model.joblib` + `schema.json` + `metadata.json`, with a full test suite.

**Architecture:** One module `src/ml_lab/models/logistic_smoke.py` built incrementally across three TDD tasks. `train_smoke_model(train_df)` fits `LogisticRegression(random_state=SMOKE_SEED)` and returns a `SmokeModel` carrying the estimator, a schema dict, and a metadata dict. `SmokeModel.predict` validates input features against the schema (raising `SchemaValidationError`); `save`/`load` round-trip the three artifact files. No MLflow, CLI, evaluation, or scores in this slice.

**Tech Stack:** Python 3.11+, scikit-learn, joblib, pandas, numpy, pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Create:** `src/ml_lab/models/logistic_smoke.py` — the whole slice's logic (`SmokeModel`, `SchemaValidationError`, `train_smoke_model`, private `_build_schema`/`_build_metadata`). Built up across Tasks 1–3.
- **Create:** `tests/test_logistic_smoke_model.py` — the Model-tests assertion list. Extended across Tasks 1–3.
- **Depends on (already committed):** `src/ml_lab/config.py` (`FEATURE_NAMES`, `TARGET_NAME`, `SMOKE_SEED`, `MODELS_SMOKE_LATEST`) and `src/ml_lab/data/synthetic.py` (`generate_smoke_data`, `split_smoke_data`) — used by tests to build data in-memory. Do not modify these.

pytest is configured with `pythonpath = ["src"]` and `testpaths = ["tests"]`. `src/ml_lab/models/__init__.py` already exists. `models/smoke/` is gitignored, so saved artifacts are never committed. Imports use `from ml_lab.models.logistic_smoke import ...`.

---

### Task 1: `train_smoke_model` + `SmokeModel` core (schema + metadata)

**Files:**
- Create: `src/ml_lab/models/logistic_smoke.py`
- Test: `tests/test_logistic_smoke_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logistic_smoke_model.py`:

```python
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
    assert (a.estimator.coef_ == b.estimator.coef_).all()


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
    assert md["created_at"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_logistic_smoke_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.models.logistic_smoke'`.

- [ ] **Step 3: Create `logistic_smoke.py` with training + schema/metadata**

Create `src/ml_lab/models/logistic_smoke.py`:

```python
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

    def __init__(self, estimator, schema: dict, metadata: dict):
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logistic_smoke_model.py -v`
Expected: PASS — all four tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/models/logistic_smoke.py tests/test_logistic_smoke_model.py
git commit -m "feat: train deterministic logistic smoke model with schema/metadata"
```

---

### Task 2: `SmokeModel.predict` + schema validation

**Files:**
- Modify: `src/ml_lab/models/logistic_smoke.py`
- Test: `tests/test_logistic_smoke_model.py`

- [ ] **Step 1: Write the failing tests**

First extend the imports at the TOP of `tests/test_logistic_smoke_model.py`: add `import pytest` (with the other imports) and add `SchemaValidationError` to the model import, so it becomes:

```python
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
```

Then append these tests:

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_logistic_smoke_model.py -k "predict or reject" -v`
Expected: FAIL — `ImportError: cannot import name 'SchemaValidationError'`.

- [ ] **Step 3: Add `SchemaValidationError`, `predict`, and validation**

In `src/ml_lab/models/logistic_smoke.py`, add a `SchemaValidationError` class ABOVE `SmokeModel`:

```python
class SchemaValidationError(ValueError):
    """Raised when input features do not match the model schema."""
```

Then add these two methods INSIDE the `SmokeModel` class (after `__init__`):

```python
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
```

(`pandas` is already imported at the top of the module from Task 1.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logistic_smoke_model.py -v`
Expected: PASS — Task 1 tests plus the six new predict/rejection tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/models/logistic_smoke.py tests/test_logistic_smoke_model.py
git commit -m "feat: schema-validating predict on SmokeModel"
```

---

### Task 3: `SmokeModel.save` / `load` round-trip

**Files:**
- Modify: `src/ml_lab/models/logistic_smoke.py`
- Test: `tests/test_logistic_smoke_model.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `tests/test_logistic_smoke_model.py` (no new imports needed — `SmokeModel`, `train_smoke_model`, `_train_split`, `_test_features` already exist; `tmp_path` is a built-in pytest fixture):

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_logistic_smoke_model.py -k "save or load" -v`
Expected: FAIL — `AttributeError: 'SmokeModel' object has no attribute 'save'`.

- [ ] **Step 3: Add `save`/`load` and their imports**

In `src/ml_lab/models/logistic_smoke.py`, extend the top-of-file imports. Add these three stdlib/third-party imports (place `import json` and `from pathlib import Path` with the stdlib imports, and `import joblib` with the third-party imports):

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
```

Add `MODELS_SMOKE_LATEST` to the config import block:

```python
from ml_lab.config import (
    FEATURE_NAMES,
    MODELS_SMOKE_LATEST,
    SMOKE_SEED,
    TARGET_NAME,
)
```

Then add these methods INSIDE the `SmokeModel` class (after `_validate_features`):

```python
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
```

- [ ] **Step 4: Run the full test file to verify it passes**

Run: `uv run pytest tests/test_logistic_smoke_model.py -v`
Expected: PASS — all Task 1/2/3 tests green.

- [ ] **Step 5: Sanity-check the real artifacts (optional, non-committed)**

Run:
```bash
uv run python -c "from ml_lab.data.synthetic import generate_smoke_data, split_smoke_data; from ml_lab.models.logistic_smoke import train_smoke_model; tr,_=split_smoke_data(generate_smoke_data()); m=train_smoke_model(tr); m.save(); import json; print(json.dumps(m.schema, indent=2)); print(json.dumps(m.metadata, indent=2))"
```
Expected: prints schema + metadata; `models/smoke/latest/{model.joblib,schema.json,metadata.json}` exist locally (gitignored).

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/models/logistic_smoke.py tests/test_logistic_smoke_model.py
git commit -m "feat: persist and reload SmokeModel artifacts"
```

---

## Definition of Done

- `uv run pytest tests/test_logistic_smoke_model.py -v` is fully green (and `uv run pytest` overall stays green).
- `logistic_smoke.py` exports `SmokeModel`, `SchemaValidationError`, `train_smoke_model`, `MODEL_NAME`, `MODEL_VERSION`.
- `SmokeModel.predict` rejects missing/extra/wrong-order/non-numeric feature frames with `SchemaValidationError`.
- `save` writes `model.joblib` + `schema.json` + `metadata.json`; `load` round-trips predictions, schema, and metadata.
- `metadata["mlflow_run_id"]` is `None` (tracking deferred); no MLflow, CLI, evaluation, or `predict_proba` added.
