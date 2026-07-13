**The reproducibility chain, end to end.** It is worth tracing once, because it is the thing Phase 0 actually built:

`SMOKE_SEED` -> `generate_smoke_data()` -> a no-shuffle 180/60 split -> `split_manifest.json` with **SHA-256 checksums of both CSVs** -> those row counts and the seed logged as **MLflow params** -> the run id stamped into the model's **`metadata.json`** -> that same run reopened by `evaluate-smoke` to carry the **held-out metrics**.

Follow that chain in either direction: from a saved `model.joblib` you can find the MLflow run that produced it; from an MLflow run you can find the exact bytes it trained on. That round trip is the deliverable.

**`make mlflow-ui`** serves the local file store at 127.0.0.1:5000 to see it.

(One footnote in `mlflow_utils.py` worth knowing: MLflow 3.14 put the local file store into maintenance mode and raises without an opt-out, so the module sets `MLFLOW_ALLOW_FILE_STORE` at import time. Phase 0 uses the file store deliberately.)
