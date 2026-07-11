# Phase 0 CLI Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `train_smoke` / `evaluate_smoke` orchestration and the `train-smoke` / `evaluate-smoke` Typer commands that back `make train` / `make evaluate`, closing the Phase 0 local lifecycle gate (train and evaluate share one MLflow run).

**Architecture:** One module `src/ml_lab/cli.py` built across three TDD tasks. `train_smoke` composes data → model → MLflow run → save (setting `metadata["mlflow_run_id"]`) → `train_run_id.txt`. `evaluate_smoke` reads that run id, runs `evaluate_smoke_model`, and appends eval metrics to the same run. Thin Typer commands wrap the plain functions. No changes to the data/model/eval/tracking modules.

**Tech Stack:** Python 3.11+, Typer, MLflow, scikit-learn, pandas, joblib, pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Create:** `src/ml_lab/cli.py` — `train_smoke` (Task 1), `evaluate_smoke` (Task 2), Typer `app` + commands (Task 3). Imports are added incrementally per task.
- **Create:** `tests/test_lifecycle.py` — lifecycle tests, extended across all three tasks.
- **Depends on (already committed, do not modify):** `config.py`, `data/synthetic.py` (`write_smoke_data`), `models/logistic_smoke.py` (`train_smoke_model`, `SmokeModel`), `evaluation/logistic_eval.py` (`evaluate_smoke_model`), `tracking/mlflow_utils.py` (`start_training_run`, `log_evaluation_metrics`).

pytest is configured with `pythonpath = ["src"]`. `data/processed/`, `models/smoke/`, `reports/smoke/`, `mlruns/` are gitignored. `pyproject.toml` already defines `ml-lab = "ml_lab.cli:app"` and the `Makefile` `train`/`evaluate` targets already call `uv run ml-lab train-smoke` / `evaluate-smoke`. Importing `ml_lab.cli` pulls in `mlflow_utils`, which sets the MLflow file-store opt-in, so tests and the app both work against a file store.

---

### Task 1: `train_smoke` orchestration

**Files:**
- Create: `src/ml_lab/cli.py`
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lifecycle.py`:

```python
import json

import mlflow

from ml_lab.cli import train_smoke


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
```

(`train_smoke`'s `experiment_name` defaults to config `EXPERIMENT_NAME`; `_dirs`
omits it. MLflow stores params as strings, so `feature_count` reads back
`"4"`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.cli'`.

- [ ] **Step 3: Create `cli.py` with `train_smoke`**

Create `src/ml_lab/cli.py`:

```python
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml_lab.config import (
    DATA_PROCESSED_SMOKE,
    EXPERIMENT_NAME,
    FEATURE_NAMES,
    MLRUNS_DIR,
    MODELS_SMOKE_LATEST,
    REPORTS_SMOKE_LATEST,
    SMOKE_SEED,
    TARGET_NAME,
)
from ml_lab.data.synthetic import write_smoke_data
from ml_lab.models.logistic_smoke import train_smoke_model
from ml_lab.tracking.mlflow_utils import start_training_run


def train_smoke(
    data_dir=DATA_PROCESSED_SMOKE,
    model_dir=MODELS_SMOKE_LATEST,
    reports_dir=REPORTS_SMOKE_LATEST,
    tracking_uri=MLRUNS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Generate data, train the smoke model, log to MLflow, persist artifacts."""
    manifest = write_smoke_data(dest=data_dir)
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    model = train_smoke_model(train_df)

    features = train_df[FEATURE_NAMES]
    y_true = train_df[TARGET_NAME]
    y_pred = model.predict(features)
    train_metrics = {
        "train_accuracy": float(accuracy_score(y_true, y_pred)),
        "train_f1": float(f1_score(y_true, y_pred)),
    }

    params = {
        "model_type": model.metadata["model_type"],
        "random_seed": SMOKE_SEED,
        "train_rows": manifest["train_rows"],
        "test_rows": manifest["test_rows"],
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "target_name": TARGET_NAME,
    }

    run_id = start_training_run(
        params,
        train_metrics,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    model.metadata["mlflow_run_id"] = run_id
    model.save(dest=model_dir)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "train_run_id.txt").write_text(
        run_id + "\n", encoding="utf-8"
    )
    return run_id
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: PASS — both tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/cli.py tests/test_lifecycle.py
git commit -m "feat: train_smoke orchestration (data, model, MLflow, artifacts)"
```

---

### Task 2: `evaluate_smoke` + end-to-end lifecycle gate

**Files:**
- Modify: `src/ml_lab/cli.py`
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Extend the imports at the TOP of `tests/test_lifecycle.py`: add `import pytest`
and change the cli import to bring in both functions:

```python
import json

import mlflow
import pytest

from ml_lab.cli import evaluate_smoke, train_smoke
```

Then append these tests:

```python
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


def test_evaluate_smoke_without_train_run_id_raises(tmp_path):
    d = _dirs(tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_smoke(**d)
```

(`evaluate_smoke` accepts `data_dir`, `model_dir`, `reports_dir`,
`tracking_uri` — all keys in `_dirs` — so `evaluate_smoke(**d)` binds by name.
It ignores no keys because `_dirs` returns exactly those four.)

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_lifecycle.py -k "lifecycle or without_train" -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_smoke'`.

- [ ] **Step 3: Add `evaluate_smoke` to `cli.py`**

Extend the imports in `src/ml_lab/cli.py` — add `evaluate_smoke_model` and
`log_evaluation_metrics`:

```python
from ml_lab.data.synthetic import write_smoke_data
from ml_lab.evaluation.logistic_eval import evaluate_smoke_model
from ml_lab.models.logistic_smoke import train_smoke_model
from ml_lab.tracking.mlflow_utils import (
    log_evaluation_metrics,
    start_training_run,
)
```

Then append after `train_smoke`:

```python
def evaluate_smoke(
    data_dir=DATA_PROCESSED_SMOKE,
    model_dir=MODELS_SMOKE_LATEST,
    reports_dir=REPORTS_SMOKE_LATEST,
    tracking_uri=MLRUNS_DIR,
) -> dict:
    """Evaluate the saved model and append eval metrics to the training run."""
    run_id_path = Path(reports_dir) / "train_run_id.txt"
    if not run_id_path.exists():
        raise FileNotFoundError(
            f"train_run_id.txt not found at {run_id_path}; run train first."
        )
    run_id = run_id_path.read_text(encoding="utf-8").strip()

    metrics = evaluate_smoke_model(
        model_dir=model_dir, data_dir=data_dir, reports_dir=reports_dir
    )
    eval_metrics = {
        k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")
    }
    log_evaluation_metrics(run_id, eval_metrics, tracking_uri=tracking_uri)
    return metrics
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: PASS — Task 1 tests plus the lifecycle gate and the missing-run-id
test.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/cli.py tests/test_lifecycle.py
git commit -m "feat: evaluate_smoke appends eval metrics to the training run"
```

---

### Task 3: Typer commands (`train-smoke` / `evaluate-smoke`)

**Files:**
- Modify: `src/ml_lab/cli.py`
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing test**

Extend the imports at the TOP of `tests/test_lifecycle.py`: add the Typer
runner and include `app` in the cli import:

```python
from typer.testing import CliRunner

from ml_lab.cli import app, evaluate_smoke, train_smoke
```

Then append:

```python
def test_cli_exposes_train_and_evaluate_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train-smoke" in result.output
    assert "evaluate-smoke" in result.output
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `uv run pytest tests/test_lifecycle.py -k exposes -v`
Expected: FAIL — `ImportError: cannot import name 'app'`.

- [ ] **Step 3: Add the Typer app and commands to `cli.py`**

Add `import typer` at the TOP of `src/ml_lab/cli.py` (with the third-party
imports, above `import pandas as pd`):

```python
import typer
```

Then append at the END of `src/ml_lab/cli.py`:

```python
app = typer.Typer()


@app.command("train-smoke")
def train_smoke_command() -> None:
    run_id = train_smoke()
    typer.echo(f"Training run: {run_id}")


@app.command("evaluate-smoke")
def evaluate_smoke_command() -> None:
    metrics = evaluate_smoke()
    typer.echo(f"Evaluation metrics: {metrics}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: PASS — all lifecycle tests plus the command-wiring test.

- [ ] **Step 5: Confirm the full suite stays green**

Run: `uv run pytest -q`
Expected: PASS — all prior slices plus the CLI lifecycle tests.

- [ ] **Step 6: Sanity-check the real lifecycle (optional, non-committed)**

Run:
```bash
make train
make evaluate
```
Expected: `make train` prints a `Training run: <id>` line and writes
`models/smoke/latest/*`, `data/processed/smoke/*`, `reports/smoke/latest/train_run_id.txt`;
`make evaluate` prints `Evaluation metrics: {...}` and writes
`reports/smoke/latest/{metrics.json,predictions.csv}` (all gitignored).

- [ ] **Step 7: Commit**

```bash
git add src/ml_lab/cli.py tests/test_lifecycle.py
git commit -m "feat: Typer train-smoke/evaluate-smoke commands"
```

---

## Definition of Done

- `uv run pytest` is fully green.
- `cli.py` exports `train_smoke`, `evaluate_smoke`, and a Typer `app` with `train-smoke` / `evaluate-smoke`.
- `train_smoke` writes data + model (metadata `mlflow_run_id` populated) + `train_run_id.txt`, and logs params + train metrics to a new MLflow run.
- `evaluate_smoke` reads `train_run_id.txt` (raises `FileNotFoundError` if absent), writes `metrics.json` + `predictions.csv`, and appends eval metrics to the same run.
- The lifecycle test proves one MLflow run holds training params, training metrics, and evaluation metrics; accuracy/F1 ≥ 0.95.
- `make train` / `make evaluate` run end-to-end. No changes to the data/model/eval/tracking modules; GPU automation remains out of scope.
