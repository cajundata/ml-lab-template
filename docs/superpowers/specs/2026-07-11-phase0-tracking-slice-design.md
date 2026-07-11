# Phase 0 — Tracking (MLflow) Slice (Design)

Date: 2026-07-11
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00`, sections
"What `make train` does", "What `make evaluate` does", and the "MLflow
lifecycle tests"). Builds on the data, model, and evaluation slices.

**Slice boundary (decided):** MLflow helper library only. This slice delivers
`src/ml_lab/tracking/mlflow_utils.py` with generic helpers and their tests
against an isolated tracking store. Wiring these into
`train_smoke_model`/`evaluate_smoke_model`, populating
`metadata["mlflow_run_id"]`, writing `train_run_id.txt`, and the
`make train`/`make evaluate` CLI are **deferred to the CLI slice**. The model
and evaluation modules stay untouched.

The Phase 0 doc is the spec of record. This design pins the helper API and the
micro-decisions the doc leaves implicit; it does not reopen any locked
decision.

## Scope

In scope:
- `src/ml_lab/tracking/mlflow_utils.py` — `start_training_run`,
  `log_evaluation_metrics`.
- `tests/test_mlflow_lifecycle.py`.

Out of scope (CLI slice):
- Integrating logging into `train_smoke_model` / `evaluate_smoke_model`.
- Populating `metadata["mlflow_run_id"]`; writing
  `reports/smoke/latest/train_run_id.txt`.
- `src/ml_lab/cli.py`, `make train`, `make evaluate`.
- Deciding the concrete Phase 0 param/metric sets (the CLI composes those and
  passes them as dicts).

## Rationale for two purpose-built helpers

The Phase 0 lifecycle has exactly two MLflow moments: training **creates** a
run and logs params + training metrics; evaluation **reopens** that run and
appends evaluation metrics to it. Two functions that mirror those moments make
the "train and evaluate share one run id" gate explicit and directly testable,
and give the CLI two obvious call sites. Thin passthrough wrappers around the
raw MLflow API were rejected as lower-value indirection.

The helpers take generic `params`/`metrics` dicts so this module never
hardcodes the Phase 0 param list; the CLI decides what to log.

## Module design — `src/ml_lab/tracking/mlflow_utils.py`

Imports: `mlflow`; `EXPERIMENT_NAME`, `MLRUNS_DIR` from `ml_lab.config`.

### `start_training_run`

```python
def start_training_run(
    params: dict,
    metrics: dict,
    tracking_uri=MLRUNS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    ...
```

Behavior:
1. `mlflow.set_tracking_uri(str(tracking_uri))`.
2. `mlflow.set_experiment(experiment_name)`.
3. `with mlflow.start_run() as run:` — `mlflow.log_params(params)`;
   `mlflow.log_metrics(metrics)`; capture `run_id = run.info.run_id`.
4. Return `run_id`.

The `with` context guarantees the run is closed even on error.

### `log_evaluation_metrics`

```python
def log_evaluation_metrics(
    run_id: str,
    metrics: dict,
    tracking_uri=MLRUNS_DIR,
) -> None:
    ...
```

Behavior:
1. `mlflow.set_tracking_uri(str(tracking_uri))`.
2. `with mlflow.start_run(run_id=run_id):` — `mlflow.log_metrics(metrics)`.

Reopening by `run_id` appends to the existing run rather than creating a new
one. The experiment does not need to be re-set to resume a run by id; only the
tracking URI must point at the store that holds the run.

## Data notes / gotchas

- **Params are stringified.** MLflow stores params as strings, so a numeric or
  list param value comes back as its `str(...)` form. Tests assert the string
  form (e.g. `"20260706"`), not the original type.
- **Metrics are floats.** `mlflow.log_metrics` requires numeric values;
  they come back as floats.
- **Global state.** MLflow keeps a process-global tracking URI and active-run
  stack. Every test passes its own `tracking_uri=tmp_path` store, and both
  helpers use `with` context managers so no run leaks across tests. No test
  writes to the repo's real `./mlruns`.

## Tests — `tests/test_mlflow_lifecycle.py`

Each test uses a per-test store: `tracking_uri = tmp_path / "mlruns"`, and a
test experiment name (the config `EXPERIMENT_NAME` is fine). Tests read runs
back with `mlflow.get_run(run_id)` and look the experiment up with
`mlflow.get_experiment_by_name(...)`.

- **Training run is created and logged:** `start_training_run(params, metrics,
  tracking_uri=..., experiment_name=...)` returns a non-empty `run_id`; the run
  read back contains the logged params (string form) and metrics; the run's
  `experiment_id` matches the named experiment.
- **Evaluation metrics append to the same run:** after
  `start_training_run(...)` → `log_evaluation_metrics(run_id, eval_metrics,
  tracking_uri=...)`, `mlflow.get_run(run_id).data.metrics` contains the
  evaluation metrics.
- **Lifecycle gate (one run id):** a single run id, after both calls, holds the
  training params, the training metrics, **and** the evaluation metrics
  together.
- **No duplicate run:** logging evaluation metrics does not create a second run
  — the experiment still has exactly one run after both calls.

Illustrative fixtures for tests (generic dicts):
`params = {"model_type": "LogisticRegression", "random_seed": 20260706}`;
`train_metrics = {"train_accuracy": 1.0, "train_f1": 1.0}`;
`eval_metrics = {"accuracy": 1.0, "f1": 1.0}`.

## Non-goals / deferred

- No changes to `train_smoke_model` / `evaluate_smoke_model` /
  `SmokeModel.metadata`.
- No `train_run_id.txt`, no CLI, no `make train` / `make evaluate`.
- No remote/DB-backed MLflow, no model registry (local file store only).
