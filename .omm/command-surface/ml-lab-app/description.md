`src/ml_lab/cli.py` — the `ml-lab` Typer app, installed as a console script via `[project.scripts]`. Free, deterministic, offline.

Two commands, `train-smoke` and `evaluate-smoke`, and they must run in that order: training writes the MLflow run id to `reports/smoke/latest/train_run_id.txt`, and evaluation reads it back to append its metrics to the *same* run.

Worth knowing when reading the file: the Typer commands are three-line wrappers. The real logic is in the module-level `train_smoke()` / `evaluate_smoke()` functions above them, whose every path and knob is a keyword argument defaulting to a value from `config.py` — which is what makes the whole lifecycle testable without touching the real workspace.
