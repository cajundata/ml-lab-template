Not a module — the set of directories the two pipelines read and write. Worth naming explicitly because it is where the halves quietly meet, and because none of it is source.

- `data/processed/smoke/` — `train.csv`, `test.csv`, `split_manifest.json` (with SHA-256 checksums).
- `models/smoke/latest/` — `model.joblib`, `schema.json`, `metadata.json` saved as a unit.
- `reports/smoke/latest/` — `metrics.json`, `predictions.csv`, and `train_run_id.txt` (the handoff file that lets `evaluate-smoke` find the MLflow run that `train-smoke` created).
- `mlruns/` — MLflow's local file store.
- `artifacts/<run-id>/` — the GPU benchmark bundle pulled down from the droplet: the probe JSONs, `benchmark.json`, `benchmark.log`, `nvidia-smi.txt`, plus `bootstrap.log` (the droplet's cloud-init output, pulled separately as forensic evidence).

`make clean` reaps `data/processed`, `models/smoke`, and `reports/smoke` — but **not** `mlruns/` and **not** `artifacts/`. GPU runs cost real money, so their evidence is never automatically deleted.
