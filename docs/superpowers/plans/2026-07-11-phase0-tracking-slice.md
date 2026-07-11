# Phase 0 Tracking (MLflow) Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide two MLflow helpers — `start_training_run` (create a run, log params+metrics, return its id) and `log_evaluation_metrics` (reopen a run by id and append metrics) — so training and evaluation can share one MLflow run.

**Architecture:** One module `src/ml_lab/tracking/mlflow_utils.py` with two generic helpers built across two TDD tasks. Both set the tracking URI and use `with mlflow.start_run(...)` context managers. Tests run against a per-test isolated file store (`tmp_path`), never the repo's real `./mlruns`. No model/eval/CLI integration in this slice.

**Tech Stack:** Python 3.11+, MLflow, pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Create:** `src/ml_lab/tracking/mlflow_utils.py` — `start_training_run` (Task 1), `log_evaluation_metrics` (Task 2). `src/ml_lab/tracking/__init__.py` already exists.
- **Create:** `tests/test_mlflow_lifecycle.py` — tracking tests, extended across both tasks.
- **Depends on (already committed, do not modify):** `config.py` (`EXPERIMENT_NAME = "phase0-smoke-lifecycle"`, `MLRUNS_DIR`).

pytest is configured with `pythonpath = ["src"]`. `mlruns/` is gitignored. Imports use `from ml_lab.tracking.mlflow_utils import ...`. MLflow keeps a process-global tracking URI; every test passes its own `tracking_uri=tmp_path / "mlruns"`, so state never leaks and the real `./mlruns` is untouched.

---

### Task 1: `start_training_run`

**Files:**
- Create: `src/ml_lab/tracking/mlflow_utils.py`
- Test: `tests/test_mlflow_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mlflow_lifecycle.py`:

```python
import mlflow

from ml_lab.config import EXPERIMENT_NAME
from ml_lab.tracking.mlflow_utils import start_training_run

PARAMS = {"model_type": "LogisticRegression", "random_seed": 20260706}
TRAIN_METRICS = {"train_accuracy": 1.0, "train_f1": 1.0}


def test_start_training_run_returns_run_id(tmp_path):
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tmp_path / "mlruns",
        experiment_name=EXPERIMENT_NAME,
    )
    assert isinstance(run_id, str)
    assert run_id


def test_start_training_run_logs_params_and_metrics(tmp_path):
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tmp_path / "mlruns",
        experiment_name=EXPERIMENT_NAME,
    )
    run = mlflow.get_run(run_id)
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.params["random_seed"] == "20260706"
    assert run.data.metrics["train_accuracy"] == 1.0
    assert run.data.metrics["train_f1"] == 1.0


def test_start_training_run_uses_named_experiment(tmp_path):
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tmp_path / "mlruns",
        experiment_name=EXPERIMENT_NAME,
    )
    run = mlflow.get_run(run_id)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert run.info.experiment_id == experiment.experiment_id
```

(Params are stored by MLflow as strings, so `random_seed` reads back as
`"20260706"`. After `start_training_run` sets the tracking URI, `mlflow.get_run`
and `mlflow.get_experiment_by_name` resolve against that same per-test store.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_mlflow_lifecycle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.tracking.mlflow_utils'`.

- [ ] **Step 3: Create `mlflow_utils.py` with `start_training_run`**

Create `src/ml_lab/tracking/mlflow_utils.py`:

```python
import mlflow

from ml_lab.config import EXPERIMENT_NAME, MLRUNS_DIR


def start_training_run(
    params: dict,
    metrics: dict,
    tracking_uri=MLRUNS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Create an MLflow run, log params + metrics, and return its run id."""
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    return run_id
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_mlflow_lifecycle.py -v`
Expected: PASS — all three tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/tracking/mlflow_utils.py tests/test_mlflow_lifecycle.py
git commit -m "feat: start_training_run MLflow helper"
```

---

### Task 2: `log_evaluation_metrics` + one-run lifecycle gate

**Files:**
- Modify: `src/ml_lab/tracking/mlflow_utils.py`
- Test: `tests/test_mlflow_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Extend the import at the TOP of `tests/test_mlflow_lifecycle.py` to include
`log_evaluation_metrics`:

```python
from ml_lab.tracking.mlflow_utils import (
    log_evaluation_metrics,
    start_training_run,
)
```

Then append the eval-metrics constant and tests:

```python
EVAL_METRICS = {"accuracy": 1.0, "f1": 1.0}


def test_log_evaluation_metrics_appends_to_run(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )
    log_evaluation_metrics(run_id, EVAL_METRICS, tracking_uri=tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["accuracy"] == 1.0
    assert run.data.metrics["f1"] == 1.0


def test_lifecycle_one_run_holds_train_and_eval(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )
    log_evaluation_metrics(run_id, EVAL_METRICS, tracking_uri=tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.metrics["train_accuracy"] == 1.0
    assert run.data.metrics["accuracy"] == 1.0


def test_log_evaluation_metrics_creates_no_second_run(tmp_path):
    tracking_uri = tmp_path / "mlruns"
    run_id = start_training_run(
        PARAMS,
        TRAIN_METRICS,
        tracking_uri=tracking_uri,
        experiment_name=EXPERIMENT_NAME,
    )
    log_evaluation_metrics(run_id, EVAL_METRICS, tracking_uri=tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_mlflow_lifecycle.py -k "eval or lifecycle or second_run" -v`
Expected: FAIL — `ImportError: cannot import name 'log_evaluation_metrics'`.

- [ ] **Step 3: Add `log_evaluation_metrics` to `mlflow_utils.py`**

Append to `src/ml_lab/tracking/mlflow_utils.py`:

```python
def log_evaluation_metrics(
    run_id: str,
    metrics: dict,
    tracking_uri=MLRUNS_DIR,
) -> None:
    """Reopen an existing MLflow run by id and log evaluation metrics to it."""
    mlflow.set_tracking_uri(str(tracking_uri))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
```

(Resuming by `run_id` appends to the existing run; the experiment does not need
to be re-set — only the tracking URI must point at the store holding the run.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_mlflow_lifecycle.py -v`
Expected: PASS — all Task 1 + Task 2 tests green.

- [ ] **Step 5: Confirm the full suite stays green**

Run: `uv run pytest -q`
Expected: PASS — the tracking tests plus all prior data/model/eval tests.

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/tracking/mlflow_utils.py tests/test_mlflow_lifecycle.py
git commit -m "feat: log_evaluation_metrics appends to the training run"
```

---

## Definition of Done

- `uv run pytest` is fully green.
- `mlflow_utils.py` exports `start_training_run` and `log_evaluation_metrics`.
- `start_training_run` logs params (string form) + metrics under the named experiment and returns a run id.
- `log_evaluation_metrics` appends metrics to that same run id; no second run is created.
- The lifecycle test proves one run id holds training params, training metrics, and evaluation metrics together.
- No changes to `train_smoke_model`/`evaluate_smoke_model`/`SmokeModel`; no `train_run_id.txt`, no CLI.
