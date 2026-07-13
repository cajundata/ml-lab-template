**Stage 1** — `src/ml_lab/data/synthetic.py`. No download, no fixture files, no network.

`generate_smoke_data()` draws 240 rows x 4 gaussian features from `np.random.default_rng(20260706)`, computes a fixed linear score (`2.0*x0 - 1.25*x1 + 0.75*x2 + 0.25*x3`), and thresholds at zero. **Linearly separable by construction** — the model is supposed to nail it.

`split_smoke_data()` takes the first 180 rows as train, the remaining 60 as test, **with no shuffle**. Seed + no shuffle = a split reproducible byte-for-byte.

`write_smoke_data()` persists `train.csv` / `test.csv` and writes `split_manifest.json`: the seed, row counts, feature/target names, and **SHA-256 checksums of both CSVs**. It returns that manifest, which `train_smoke()` folds directly into the MLflow params — so every run permanently records exactly which bytes it trained on.

**Outputs:** `data/processed/smoke/{train.csv, test.csv, split_manifest.json}`.
