`Makefile` — nine targets, and the surface humans actually type.

**Free:** `setup` (`uv sync --dev`), `test` (`uv run pytest` — 193 tests), `train`, `evaluate`, `mlflow-ui` (serves the local file store at 127.0.0.1:5000), `clean` (reaps `data/processed`, `models/smoke`, `reports/smoke` — but deliberately **not** `mlruns/` or `artifacts/`).

**GPU:** `gpu-run`, `gpu-up`, `gpu-down`, `gpu-audit` — each a thin wrapper around `uv run python scripts/do_gpu.py <cmd>`.

The one piece of real logic in the whole file is `gpu-down`'s guard:

```make
@test -n "$(DROPLET_ID)" || (echo "DROPLET_ID is required: make gpu-down DROPLET_ID=<id>" && exit 1)
```

It refuses to run without an explicit droplet id rather than guessing. Destroying the wrong droplet, or destroying nothing while believing you destroyed something, is precisely the failure mode this repo exists to prevent — so the destroy command does not get to be clever.
