`src/ml_lab/cli.py` — the `ml-lab` Typer app, installed as a console script (`[project.scripts]`). Two commands, but the file is more than a CLI: the module-level `train_smoke()` and `evaluate_smoke()` functions *are* the local lifecycle orchestration, and the Typer commands are three-line wrappers around them.

`train_smoke()` runs the whole forward pass: generate + split + persist the data, fit the model, score it on its own training split, open an MLflow run with the params and training metrics, stamp the returned run id into the model's metadata, save the model, and write `train_run_id.txt`.

`evaluate_smoke()` picks that run id back up off disk, evaluates the saved model against the saved test split, and appends the eval metrics to the *same* MLflow run. It refuses to run if `train_run_id.txt` is absent, with a message that says to train first.

Every path and knob is a keyword argument defaulting to a value from `config.py`, which is what makes both functions directly testable without touching the real workspace.
