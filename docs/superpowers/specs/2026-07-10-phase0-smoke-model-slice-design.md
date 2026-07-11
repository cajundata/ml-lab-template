# Phase 0 — Smoke Model Slice (Design)

Date: 2026-07-10
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00`, section
"Smoke model training" and the "Model tests" assertion list). Builds on the
completed synthetic data slice
(`2026-07-10-phase0-synthetic-data-slice-design.md`).

**Slice boundary (decided):** model logic only. This slice trains the
deterministic logistic smoke model, wraps it with schema validation, and
persists the three model artifacts. MLflow tracking, `train_run_id.txt`, the
`cli.py` `train-smoke` command, and evaluation/metrics are **deferred to their
own later slices**.

The Phase 0 doc is the spec of record. This design pins down the
implementation shape (approach A: a `SmokeModel` wrapper class) and the
micro-decisions the doc leaves implicit; it does not reopen any locked
decision.

## Scope

In scope:
- `src/ml_lab/models/logistic_smoke.py` — `SmokeModel` wrapper,
  `train_smoke_model`, `SchemaValidationError`.
- `models/smoke/latest/{model.joblib, metadata.json, schema.json}` outputs
  (gitignored).
- `tests/test_logistic_smoke_model.py`.

Out of scope (later slices):
- MLflow logging, experiment setup, `reports/smoke/latest/train_run_id.txt`.
- `predict_proba` / prediction scores (evaluation slice).
- `src/ml_lab/cli.py` and the `train-smoke` command.
- Evaluation, metrics, `evaluation/logistic_eval.py`.

## Module design — `src/ml_lab/models/logistic_smoke.py`

Imports config constants: `MODELS_SMOKE_LATEST`, `FEATURE_NAMES`,
`TARGET_NAME`, `SMOKE_SEED`. Do not modify `config.py`.

Module-level constants (model-specific, defined in this module):

```python
MODEL_NAME = "logistic_smoke"
MODEL_VERSION = "0.1.0"
```

### `SchemaValidationError(ValueError)`

Raised by `SmokeModel.predict` when the input feature frame does not match the
schema. Subclasses `ValueError` so callers (and the later evaluation slice)
can catch it precisely.

### `SmokeModel`

A wrapper holding three attributes:
- `estimator` — the fitted `sklearn.linear_model.LogisticRegression`.
- `schema` — dict (see schema.json below).
- `metadata` — dict (see metadata.json below).

Methods:

1. `predict(self, df: pd.DataFrame) -> np.ndarray`
   - Validate `df` against `self.schema` (see Validation semantics). Raise
     `SchemaValidationError` on any mismatch.
   - Return `self.estimator.predict(df[schema["feature_names"]])`.

2. `save(self, dest: Path = MODELS_SMOKE_LATEST) -> None`
   - `dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)`.
   - `joblib.dump(self.estimator, dest / "model.joblib")`.
   - Write `dest / "schema.json"` = `json.dumps(self.schema, indent=2) + "\n"`.
   - Write `dest / "metadata.json"` =
     `json.dumps(self.metadata, indent=2) + "\n"` (encoding="utf-8").

3. `@classmethod load(cls, dest: Path = MODELS_SMOKE_LATEST) -> "SmokeModel"`
   - Load `model.joblib` via `joblib.load`.
   - Load `schema.json` and `metadata.json` via `json.loads`.
   - Return `cls(estimator=..., schema=..., metadata=...)`.

### `train_smoke_model(train_df: pd.DataFrame) -> SmokeModel`

- Split `train_df` into `X = train_df[FEATURE_NAMES]` and
  `y = train_df[TARGET_NAME]`.
- Fit `LogisticRegression(random_state=SMOKE_SEED)` on `(X, y)`.
- Build `schema` and `metadata` from `train_df` (feature dtypes read from
  `X.dtypes`, target labels from the sorted unique values of `y`).
- Return `SmokeModel(estimator, schema, metadata)`.

Takes a DataFrame, not a path — decoupled from disk. The later CLI slice
composes `write_smoke_data()` → read `train.csv` → `train_smoke_model(train_df)`
→ `model.save()`.

## Validation semantics

`SmokeModel.predict` validates in this order, raising `SchemaValidationError`
with a clear message on the first failure:

1. **Column set + order:** `list(df.columns) != schema["feature_names"]`
   fails. This single exact-equality check rejects a missing column, an extra
   column, and wrong feature order simultaneously. (The input frame must
   contain exactly the feature columns, in schema order — the target column is
   not expected at predict time.)
2. **Numeric dtype:** any feature column that is not numeric
   (`pandas.api.types.is_numeric_dtype` is False) fails.

These two checks satisfy the four required rejection cases: missing column,
extra column, wrong order, non-numeric values.

## Artifact schemas

`schema.json`:

```json
{
  "feature_names": ["x0", "x1", "x2", "x3"],
  "dtypes": {"x0": "float64", "x1": "float64", "x2": "float64", "x3": "float64"},
  "target_name": "target",
  "target_labels": [0, 1]
}
```

- `feature_names` captures required names and order.
- `dtypes` maps each feature to its pandas dtype string (from the training
  frame).
- `target_labels` are the sorted unique target values as plain ints.

`metadata.json`:

```json
{
  "model_name": "logistic_smoke",
  "model_type": "LogisticRegression",
  "version": "0.1.0",
  "features": ["x0", "x1", "x2", "x3"],
  "target": "target",
  "created_at": "<ISO-8601 UTC timestamp>",
  "mlflow_run_id": null
}
```

- `model_type` = `type(estimator).__name__`.
- `created_at` = `datetime.now(timezone.utc).isoformat()` (a real timestamp;
  not deterministic, not checksummed).
- `mlflow_run_id` = `null` in this slice; the tracking slice will populate it.

## Determinism

- The training data is already deterministic (synthetic data slice).
- `LogisticRegression(random_state=SMOKE_SEED)` makes fitting reproducible;
  two independent trains on the same data produce identical `coef_`.
- `created_at` in metadata is intentionally non-deterministic and is therefore
  only asserted for presence/format in tests, never for an exact value.

## Tests — `tests/test_logistic_smoke_model.py`

Tests build data in-memory (`generate_smoke_data` → `split_smoke_data`) and
save into `tmp_path`. Covering the Phase 0 "Model tests" list plus reasonable
additions:

- `train_smoke_model` returns a `SmokeModel` whose `estimator` is a fitted
  `LogisticRegression` (has `coef_`).
- After `save`: `model.joblib`, `metadata.json`, `schema.json` all exist.
- `SmokeModel.load` round-trips: reloaded model's `predict` equals the
  pre-save model's `predict` on the same features.
- Prediction count equals input row count.
- Predictions are binary (subset of `{0, 1}`).
- `schema.json` contents: `feature_names` in order `[x0..x3]`, `dtypes` has an
  entry per feature, `target_name == "target"`, `target_labels == [0, 1]`.
- `metadata.json` has the required keys with expected static values
  (`model_name`, `model_type`, `version`, `features`, `target`,
  `mlflow_run_id is None`) and a present `created_at`.
- Deterministic training: two `train_smoke_model` calls on the same train
  split produce equal `estimator.coef_`.
- Rejection tests, each expecting `SchemaValidationError`:
  - missing feature column,
  - extra feature column,
  - wrong feature order,
  - non-numeric feature values.

## Non-goals / deferred

- No MLflow, no `train_run_id.txt`, no CLI, no evaluation/metrics.
- No `predict_proba`/scores yet — added in the evaluation slice, where the
  wrapper is the natural home.
- `make train` remains non-functional until the CLI slice lands (expected).
