# Models directory

This directory holds generated model artifacts. Nothing here is committed except this README and the `.gitkeep` placeholder. Every artifact is reproducible by rerunning training, so committing binaries would only bloat the repo and invite version drift between the model and the code that produced it.

## Layout

In Phase 0, `make train` writes the smoke model to `models/smoke/latest/`:

- `model.joblib` is the serialized logistic smoke model.
- `metadata.json` records the model name, model type, version, feature list, target, creation timestamp, and the MLflow run id that produced it.
- `schema.json` records the required feature names, feature order, dtypes, and target labels. Evaluation refuses to run if incoming data does not match this schema.

`make clean` removes the contents of `models/smoke/`. Regenerate with `make train`.