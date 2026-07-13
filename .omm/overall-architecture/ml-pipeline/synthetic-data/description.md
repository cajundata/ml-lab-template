`src/ml_lab/data/synthetic.py` — deterministic data generation. No network, no download, no fixture files.

`generate_smoke_data()` draws 240 rows x 4 gaussian features from `np.random.default_rng(SMOKE_SEED)`, computes a fixed linear score (`2.0*x0 - 1.25*x1 + 0.75*x2 + 0.25*x3`), and thresholds at zero. The data is linearly separable by construction — the model is *supposed* to nail it, because the point is to prove the lifecycle, not to be a hard problem.

`split_smoke_data()` takes the first 180 rows as train and the rest as test, **with no shuffle**. Combined with the fixed seed, that makes the split reproducible byte-for-byte.

`write_smoke_data()` persists `train.csv` / `test.csv` and writes `split_manifest.json` carrying the seed, row counts, feature/target names, and **SHA-256 checksums of both CSVs**. It returns the manifest, which `train_smoke()` folds straight into the MLflow params — so every run records exactly which bytes it trained on.
