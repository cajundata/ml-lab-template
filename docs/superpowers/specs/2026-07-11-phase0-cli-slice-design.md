# Phase 0 — CLI Slice (Local Lifecycle Capstone) (Design)

Date: 2026-07-11
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00`, sections
"What `make train` does", "What `make evaluate` does", the "End-to-end
lifecycle test", and the local repo/MLflow exit gates). This is the
**integration slice**: it composes the data, model, evaluation, and tracking
slices into the `train-smoke` / `evaluate-smoke` commands that back
`make train` / `make evaluate`, closing the Phase 0 local lifecycle gate
(train and evaluate share one MLflow run).

Unlike the prior slices there is no "logic-only" boundary — wiring everything
end-to-end is the point. This slice populates the deferrals from earlier
slices: `metadata["mlflow_run_id"]`, `train_run_id.txt`, and the MLflow run
linkage.

The Phase 0 doc is the spec of record. This design pins the orchestration
structure and micro-decisions; it does not reopen any locked decision.

## Scope

In scope:
- `src/ml_lab/cli.py` — `train_smoke`, `evaluate_smoke` (plain orchestration
  functions) + a Typer `app` with `train-smoke` / `evaluate-smoke` commands.
- `tests/test_lifecycle.py`.

Out of scope:
- GPU automation (`scripts/do_gpu.py`, cloud-init, benchmark) — the remaining
  Phase 0 item, its own slice.
- Any change to the data/model/evaluation/tracking modules. The CLI composes
  them; it sets `metadata["mlflow_run_id"]` by mutating the dict before save,
  so the model module needs no change.

## Structure (decided)

Plain orchestration functions with config-default arguments, wrapped by thin
Typer commands. Tests call the plain functions with `tmp_path` directories, so
they exercise the exact code the CLI runs without touching the real repo dirs.

## Module design — `src/ml_lab/cli.py`

Imports: `typer`; `pandas`; `accuracy_score`, `f1_score` from
`sklearn.metrics`; `pathlib.Path`; from `ml_lab.config`:
`DATA_PROCESSED_SMOKE`, `MODELS_SMOKE_LATEST`, `REPORTS_SMOKE_LATEST`,
`MLRUNS_DIR`, `EXPERIMENT_NAME`, `SMOKE_SEED`, `FEATURE_NAMES`, `TARGET_NAME`;
`write_smoke_data` from `ml_lab.data.synthetic`; `train_smoke_model` from
`ml_lab.models.logistic_smoke`; `evaluate_smoke_model` from
`ml_lab.evaluation.logistic_eval`; `start_training_run`,
`log_evaluation_metrics` from `ml_lab.tracking.mlflow_utils`.

### `train_smoke`

```python
def train_smoke(
    data_dir=DATA_PROCESSED_SMOKE,
    model_dir=MODELS_SMOKE_LATEST,
    reports_dir=REPORTS_SMOKE_LATEST,
    tracking_uri=MLRUNS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    ...
```

Behavior:
1. `manifest = write_smoke_data(dest=data_dir)` — writes `train.csv`,
   `test.csv`, `split_manifest.json`.
2. `train_df = pd.read_csv(Path(data_dir) / "train.csv")`.
3. `model = train_smoke_model(train_df)`.
4. Train metrics on the train split:
   `features = train_df[FEATURE_NAMES]`; `y_true = train_df[TARGET_NAME]`;
   `y_pred = model.predict(features)`;
   `train_metrics = {"train_accuracy": float(accuracy_score(y_true, y_pred)),
   "train_f1": float(f1_score(y_true, y_pred))}`.
5. Params:
   ```python
   params = {
       "model_type": model.metadata["model_type"],
       "random_seed": SMOKE_SEED,
       "train_rows": manifest["train_rows"],
       "test_rows": manifest["test_rows"],
       "feature_count": len(FEATURE_NAMES),
       "feature_names": list(FEATURE_NAMES),
       "target_name": TARGET_NAME,
   }
   ```
6. `run_id = start_training_run(params, train_metrics,
   tracking_uri=tracking_uri, experiment_name=experiment_name)`.
7. `model.metadata["mlflow_run_id"] = run_id`; `model.save(dest=model_dir)`.
8. `reports_dir = Path(reports_dir)`; `reports_dir.mkdir(parents=True,
   exist_ok=True)`; write `reports_dir / "train_run_id.txt"` =
   `run_id + "\n"` (encoding utf-8).
9. Return `run_id`.

### `evaluate_smoke`

```python
def evaluate_smoke(
    data_dir=DATA_PROCESSED_SMOKE,
    model_dir=MODELS_SMOKE_LATEST,
    reports_dir=REPORTS_SMOKE_LATEST,
    tracking_uri=MLRUNS_DIR,
) -> dict:
    ...
```

(Parameter order mirrors `train_smoke` — `data_dir` first — for CLI-module
consistency; the internal `evaluate_smoke_model` call is by keyword.)

Behavior:
1. `run_id_path = Path(reports_dir) / "train_run_id.txt"`. If it does not
   exist, raise `FileNotFoundError` (evaluate requires a prior train).
2. `run_id = run_id_path.read_text(encoding="utf-8").strip()`.
3. `metrics = evaluate_smoke_model(model_dir=model_dir, data_dir=data_dir,
   reports_dir=reports_dir)` — writes `metrics.json` (now carrying the real
   `mlflow_run_id` from model metadata) and `predictions.csv`.
4. `eval_metrics = {k: metrics[k] for k in ("accuracy", "precision", "recall",
   "f1")}`.
5. `log_evaluation_metrics(run_id, eval_metrics, tracking_uri=tracking_uri)` —
   appends evaluation metrics to the same run the training created.
6. Return `metrics`.

### Typer commands

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

`pyproject.toml` already defines `ml-lab = "ml_lab.cli:app"`, and the `Makefile`
`train`/`evaluate` targets already call `uv run ml-lab train-smoke` /
`evaluate-smoke`, so `make train` / `make evaluate` work once this module
exists.

## Data flow (one MLflow run)

`train-smoke` creates the run and logs training params + metrics, records the
run id in both `metadata.json` and `train_run_id.txt`. `evaluate-smoke` reads
`train_run_id.txt`, computes + writes evaluation artifacts, and reopens that
run to append evaluation metrics. Net result: one MLflow run holds training
params, training metrics, and evaluation metrics — the Phase 0 lifecycle gate.

## Determinism

Deterministic data + model → identical run contents (params/metrics) and
artifacts on every run; accuracy/F1 clear the 0.95 gate with margin.

## Tests — `tests/test_lifecycle.py`

Tests call the plain functions with `tmp_path` directories and a per-test
tracking store (`tmp_path / "mlruns"`), reading MLflow back via
`mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))`.

- **`train_smoke` writes all artifacts:** returns a non-empty `run_id`;
  `train.csv`, `test.csv`, `split_manifest.json` exist under `data_dir`;
  `model.joblib`, `metadata.json`, `schema.json` exist under `model_dir`;
  `metadata.json`'s `mlflow_run_id` equals the returned `run_id`;
  `reports_dir/train_run_id.txt` exists and its stripped contents equal
  `run_id`.
- **`train_smoke` logs to MLflow:** the run (read via client) contains the
  params (string form, e.g. `model_type == "LogisticRegression"`,
  `feature_count == "4"`) and metrics `train_accuracy`, `train_f1`.
- **End-to-end lifecycle gate:** `train_smoke(...)` then
  `evaluate_smoke(...)` on the same dirs → `metrics.json` and
  `predictions.csv` exist under `reports_dir`; returned `metrics["accuracy"]`
  and `metrics["f1"]` are ≥ 0.95; the single run id holds `model_type` param,
  `train_accuracy` metric, **and** `accuracy` metric (train + eval together in
  one run).
- **Evaluate without train fails:** calling `evaluate_smoke(...)` on a fresh
  `reports_dir` with no `train_run_id.txt` raises `FileNotFoundError`.
- **CLI commands are wired:** `typer.testing.CliRunner().invoke(app,
  ["--help"])` exits 0 and its output names both `train-smoke` and
  `evaluate-smoke` (verifies the Makefile/pyproject entrypoints resolve,
  without executing the pipeline against real repo dirs).

## Non-goals / deferred

- No GPU automation (separate slice).
- No changes to data/model/evaluation/tracking modules.
- No extra CLI options/flags beyond the two commands (YAGNI); orchestration
  parameters exist only as function defaults for testability, not as CLI flags.
