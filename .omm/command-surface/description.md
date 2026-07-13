Everything a human can type. Two Typer apps behind one `Makefile`, and the asymmetry between them is deliberate.

**`ml-lab`** is a proper console script, installed via `[project.scripts]` and on `$PATH` after `uv sync`. It is free to run, deterministic, and offline.

**`do_gpu`** is *not* on `$PATH`. It is reached as a script path (`uv run python scripts/do_gpu.py ...`), through a four-line entry point that imports the Typer app and calls it. **These commands spend real money**, and they are kept off `$PATH` so nobody invokes one by tab-completion accident.

In practice everyone types the `Makefile` target rather than either app directly:

| Free | Costs money |
| --- | --- |
| `make test` (193 tests) | `make gpu-run` |
| `make train` / `make evaluate` | `make gpu-up` |
| `make mlflow-ui` | `make gpu-down DROPLET_ID=<id>` |
| `make clean` / `make setup` | `make gpu-audit` (free, but tells you what is costing) |
