# Phase 0 — Synthetic Smoke Data Slice (Design)

Date: 2026-07-10
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00`, section
"Synthetic smoke data" and the "Data tests" assertion list). This slice
delivers **only** deterministic synthetic data generation, splitting,
persistence, and its tests. It is the first incremental checkpoint toward the
local lifecycle gate.

The Phase 0 doc is the spec of record. This design only pins down the
micro-decisions that doc leaves implicit; it does not reopen any locked
decision.

## Scope

In scope:
- `src/ml_lab/data/synthetic.py` — generation, split, persistence.
- `data/processed/smoke/{train.csv, test.csv, split_manifest.json}` outputs.
- `tests/test_synthetic_data.py` — the Data-tests assertions.

Out of scope (later slices/plans):
- `src/ml_lab/cli.py` and the `train-smoke` command.
- Smoke model (`models/logistic_smoke.py`), evaluation, tracking, MLflow.
- All GPU automation.

## Module design — `src/ml_lab/data/synthetic.py`

Imports the constants already defined in `src/ml_lab/config.py`
(`SMOKE_SEED`, `SMOKE_ROWS`, `SMOKE_TRAIN_ROWS`, `SMOKE_TEST_ROWS`,
`FEATURE_NAMES`, `TARGET_NAME`, `DATA_PROCESSED_SMOKE`). The current
uncommitted version references these names without importing them and is
therefore broken; fixing that is part of this slice.

Three functions:

1. `generate_smoke_data(seed=SMOKE_SEED, rows=SMOKE_ROWS) -> pd.DataFrame`
   - `rng = np.random.default_rng(seed)`
   - Features: `rng.normal(size=(rows, 4))` as columns `FEATURE_NAMES`
     (`x0, x1, x2, x3`).
   - Deterministic label rule (from the spec):
     `target = 1 if (2.0*x0 - 1.25*x1 + 0.75*x2 + 0.25*x3) > 0 else 0`.
   - Returns a DataFrame with columns `[x0, x1, x2, x3, target]`.
   - The linear rule makes the dataset exactly separable by its own
     hyperplane, which is what the downstream smoke model needs.

2. `split_smoke_data(df) -> tuple[pd.DataFrame, pd.DataFrame]`
   - First `SMOKE_TRAIN_ROWS` (180) rows → train.
   - Last `SMOKE_TEST_ROWS` (60) rows → test.
   - No shuffle. Row order is preserved.

3. `write_smoke_data(dest=DATA_PROCESSED_SMOKE) -> dict`
   - Generate, then split.
   - Create `dest` if missing.
   - Write `train.csv` and `test.csv` with `index=False`.
   - Compute the manifest (below), write `split_manifest.json`, and return it.
   - Later, `cli.train-smoke` will call this. For this slice it is exercised
     directly by the tests.

## Manifest schema — `split_manifest.json`

```json
{
  "seed": 20260706,
  "rows": 240,
  "train_rows": 180,
  "test_rows": 60,
  "feature_names": ["x0", "x1", "x2", "x3"],
  "target_name": "target",
  "train_sha256": "<hex>",
  "test_sha256": "<hex>"
}
```

- `train_sha256` / `test_sha256` are the SHA-256 hex digests of the written
  CSV files' bytes. This makes "checksums are stable for the fixed seed"
  mechanically testable: regenerating with the same seed must reproduce the
  same digests.
- Checksums are computed from the on-disk CSV bytes (after writing), so the
  digest covers exactly what a later reader loads.

## Determinism guarantees

- Fixed seed `20260706` via `np.random.default_rng`.
- `to_csv(index=False)` with pandas' default float formatting is byte-stable
  for identical float values, so repeated runs produce identical files and
  identical digests.
- No shuffling anywhere in this slice.

## Tests — `tests/test_synthetic_data.py`

Covers the Phase 0 "Data tests" list:

- Generation is deterministic for seed `20260706` (two calls are equal).
- Train split has 180 rows; test split has 60 rows.
- Feature columns are exactly `x0, x1, x2, x3`; target column is exactly
  `target`.
- Target is binary (`{0, 1}`).
- No required feature column contains nulls.
- `split_manifest.json` exists after `write_smoke_data`.
- Manifest `train_rows`/`test_rows` match the persisted CSV row counts.
- Manifest `feature_names` order matches the persisted CSV feature columns.
- Checksums are stable: writing twice (into isolated temp dirs) yields equal
  `train_sha256` and `test_sha256`.

Tests write into a temporary directory (not the repo's real
`data/processed/smoke/`) so they are isolated and repeatable.

## Non-goals / deferred

- No CLI wiring yet (`make train` will fail until `cli.py` lands in a later
  slice — that is expected and acceptable for this checkpoint).
- No model, evaluation, MLflow, or GPU work.
