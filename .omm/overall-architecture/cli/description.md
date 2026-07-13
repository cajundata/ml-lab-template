The operator's entry surface. Two independent Typer apps, both normally reached through `Makefile` targets rather than invoked directly.

- `src/ml_lab/cli.py` — the `ml-lab` app (installed as a console script via `[project.scripts]`). Two commands: `train-smoke` and `evaluate-smoke`. It is also where the local lifecycle is *orchestrated*: the `train_smoke()` / `evaluate_smoke()` functions above the Typer commands are the real logic, and the commands are thin wrappers that echo the result.
- `src/ml_lab/gpu/cli.py` — the GPU app: `run`, `up`, `down`, `audit`. Reached via `scripts/do_gpu.py`, a four-line entry point that just imports and calls this `app`.
- `Makefile` — the memorable surface: `make train`, `make evaluate`, `make gpu-run`, `make gpu-up`, `make gpu-down DROPLET_ID=<id>`, `make gpu-audit`, `make test`, `make mlflow-ui`, `make clean`.

Note the asymmetry: `ml-lab` is an installed entry point, while the GPU app is invoked as a script path. That is deliberate — the GPU commands spend real money, so they are not on `$PATH` by accident.
