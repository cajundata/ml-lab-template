# Phase 0 — Evaluation Slice (Design)

Date: 2026-07-11
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00`, section
"What `make evaluate` does" and the "Evaluation tests" assertion list). Builds
on the synthetic data slice and the smoke model slice
(`2026-07-10-phase0-smoke-model-slice-design.md`).

**Slice boundary (decided):** evaluation logic only. This slice loads the saved
model + test split, computes metrics, and writes `metrics.json` +
`predictions.csv`. MLflow run reopen/logging, `train_run_id.txt`, and the
`cli.py` `evaluate-smoke` command are **deferred to their own later slices**.

The Phase 0 doc is the spec of record. This design pins the implementation
shape and the micro-decisions the doc leaves implicit; it does not reopen any
locked decision.

## Scope

In scope:
- `src/ml_lab/models/logistic_smoke.py` — add `SmokeModel.predict_proba`.
- `src/ml_lab/evaluation/logistic_eval.py` — new `evaluate_smoke_model`.
- `reports/smoke/latest/{metrics.json, predictions.csv}` outputs (gitignored).
- `tests/test_logistic_evaluation.py`.

Out of scope (later slices):
- MLflow run reopen, metric logging, `reports/smoke/latest/train_run_id.txt`.
- `src/ml_lab/cli.py` and the `evaluate-smoke` command; `make evaluate`.
- Segment/error analysis, calibration curves, ROC-AUC (Phase 1 concerns).

## Part 1 — `SmokeModel.predict_proba`

Add to `SmokeModel` (in `src/ml_lab/models/logistic_smoke.py`):

```python
    def predict_proba(self, df: pd.DataFrame):
        self._validate_features(df)
        return self.estimator.predict_proba(
            df[self.schema["feature_names"]]
        )[:, 1]
```

- Validates features via the existing `_validate_features` (same
  missing/extra/wrong-order/non-numeric rejection as `predict`).
- Returns a 1-D array of positive-class probabilities `P(y = 1)`. The estimator
  is trained on labels `[0, 1]`, so `predict_proba(...)[:, 1]` is the
  probability of class `1`.

## Part 2 — `evaluate_smoke_model`

New module `src/ml_lab/evaluation/logistic_eval.py`.

```python
def evaluate_smoke_model(
    model_dir=MODELS_SMOKE_LATEST,
    data_dir=DATA_PROCESSED_SMOKE,
    reports_dir=REPORTS_SMOKE_LATEST,
) -> dict:
    ...
```

Imports: `SmokeModel`, `SchemaValidationError` from
`ml_lab.models.logistic_smoke`; `MODELS_SMOKE_LATEST`, `DATA_PROCESSED_SMOKE`,
`REPORTS_SMOKE_LATEST`, `FEATURE_NAMES`, `TARGET_NAME` from `ml_lab.config`;
`pandas`, `json`, `pathlib.Path`, and sklearn metric functions.

Behavior, in order:

1. **Guard — test split present:** `test_path = Path(data_dir) / "test.csv"`.
   If it does not exist, raise `FileNotFoundError` with a clear message.
   (Evaluation never regenerates data.)
2. **Load model:** `model = SmokeModel.load(model_dir)`.
3. **Read test data:** `test_df = pd.read_csv(test_path)`.
4. **Guard — schema match:** if any column in `FEATURE_NAMES` is absent from
   `test_df.columns`, or `TARGET_NAME` is absent, raise
   `SchemaValidationError`. (Feature-dtype mismatches are additionally caught
   by `model.predict` / `model.predict_proba` in the next step.)
5. **Predict:** `features = test_df[FEATURE_NAMES]`;
   `y_true = test_df[TARGET_NAME]`;
   `y_pred = model.predict(features)`;
   `y_score = model.predict_proba(features)`.
6. **Compute metrics** with sklearn:
   - `accuracy = accuracy_score(y_true, y_pred)`
   - `precision = precision_score(y_true, y_pred)` (binary, `pos_label=1`)
   - `recall = recall_score(y_true, y_pred)`
   - `f1 = f1_score(y_true, y_pred)`
   - `cm = confusion_matrix(y_true, y_pred)` (2×2)
   - `evaluated_row_count = len(test_df)`
7. **Write artifacts** (create `reports_dir` with
   `mkdir(parents=True, exist_ok=True)`):
   - `metrics.json` = `json.dumps(metrics, indent=2) + "\n"`, encoding utf-8,
     where `metrics` =
     ```json
     {
       "accuracy": <float>,
       "precision": <float>,
       "recall": <float>,
       "f1": <float>,
       "confusion_matrix": [[<int>, <int>], [<int>, <int>]],
       "evaluated_row_count": <int>,
       "mlflow_run_id": null
     }
     ```
     Metric floats cast via `float(...)`; confusion matrix cast to nested
     Python ints; `mlflow_run_id` copied from `model.metadata["mlflow_run_id"]`
     (`null` this slice).
   - `predictions.csv` with columns `y_true, y_pred, y_score` (via a pandas
     DataFrame, `index=False`).
8. **Return** the `metrics` dict.

## Determinism

Deterministic data (fixed seed) + deterministic model
(`random_state=SMOKE_SEED`) means metrics and predictions are identical on
every run. On this linearly separable dataset the logistic model is expected to
score well above the 0.95 accuracy/F1 gate.

## Tests — `tests/test_logistic_evaluation.py`

Setup per test (in `tmp_path`): `write_smoke_data(dest=data_dir)` to produce
`train.csv`/`test.csv`; read `train.csv`; `train_smoke_model(train_df)` then
`.save(dest=model_dir)`; call
`evaluate_smoke_model(model_dir, data_dir, reports_dir)`.

`predict_proba` tests (on `SmokeModel`):
- Returns a 1-D array of length == input row count.
- All values in `[0.0, 1.0]`.
- Rejects a bad feature frame (e.g., missing column) with
  `SchemaValidationError`.

`evaluate_smoke_model` tests:
- `metrics.json` exists after evaluation.
- `predictions.csv` exists after evaluation.
- `accuracy >= 0.95` and `f1 >= 0.95`.
- `precision` and `recall` are numeric and within `[0.0, 1.0]`.
- `confusion_matrix` is 2×2 (list of two rows, each length 2).
- `evaluated_row_count == 60`.
- `predictions.csv` has exactly 60 data rows and columns
  `["y_true", "y_pred", "y_score"]`.
- Raises `FileNotFoundError` when `test.csv` is absent (evaluate against a
  data_dir with no test split).
- Raises `SchemaValidationError` when a feature column is dropped from the
  written `test.csv` before evaluation.

## Non-goals / deferred

- No MLflow, no `train_run_id.txt`, no CLI, no `make evaluate`.
- `metrics.json` carries `mlflow_run_id: null`; the tracking slice populates it
  and adds run reopen/logging.
- No ROC-AUC, calibration, or segment error analysis (Phase 1).
