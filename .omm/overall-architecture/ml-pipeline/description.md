The deterministic local train/evaluate lifecycle — three modules under `src/ml_lab/`, orchestrated by `cli.py`, touching no network.

- `data/synthetic.py` — generates 240 rows of seeded, linearly separable binary-classification data (4 gaussian features, a fixed linear score, threshold at 0). Splits 180/60 with **no shuffle**, writes `train.csv` / `test.csv`, and emits `split_manifest.json` carrying the seed, row counts, and **SHA-256 checksums of both CSVs**.
- `models/logistic_smoke.py` — fits `LogisticRegression` and wraps it in a `SmokeModel` that carries its own `schema` and `metadata`. Predictions are **schema-validated**: column order and numeric dtypes must match, else `SchemaValidationError`. `save()`/`load()` persist `model.joblib` + `schema.json` + `metadata.json` as a unit.
- `evaluation/logistic_eval.py` — loads the saved model and the saved test split, computes accuracy/precision/recall/F1 + confusion matrix, and writes `metrics.json` and a per-row `predictions.csv` (y_true, y_pred, y_score).

The whole pipeline is reproducible byte-for-byte from `SMOKE_SEED = 20260706`. That determinism is the actual deliverable; the model itself is a placeholder for Phase 1.
