`make train` -> `ml-lab train-smoke`. Runs the entire local forward pass in one shot:

1. Generate + split + persist the synthetic data (seeded, no shuffle) and get back the manifest with its SHA-256 checksums.
2. Fit `LogisticRegression` on `train.csv`.
3. Score it on its own training split (`train_accuracy`, `train_f1`).
4. Open an **MLflow run**, logging params (model type, seed, row counts, feature/target names) and the training metrics — and get back the **run id**.
5. Stamp that run id into the model's `metadata.json`, then `save()` the model, schema, and metadata as a unit.
6. Write the run id to `reports/smoke/latest/train_run_id.txt`.

That last step is the handoff to `evaluate-smoke`. It echoes `Training run: <run_id>`.
