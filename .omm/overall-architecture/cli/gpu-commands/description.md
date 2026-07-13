`src/ml_lab/gpu/cli.py` — the GPU Typer app, reached through `scripts/do_gpu.py` (a four-line entry point that imports this `app` and calls it). Four commands:

- **`audit`** — print any billable lab resources; **exit nonzero if the account is dirty**. That exit code is what makes it usable as a CI/pre-flight check, not just a human report.
- **`down --droplet-id <id>`** — the always-safe manual reap. Routes through `destroy_and_verify` like everything else.
- **`up [--ttl-seconds N]`** — create, verify, hand off, and *leave the droplet alive* for debugging. Passing `--ttl-seconds` also flips `enforce_budget=False`, which is the deliberate escape hatch for short-fuse self-destruct testing (a 5-minute TTL cannot fit the benchmark budget, and that is the point).
- **`run`** — the full lifecycle. Note it does `raise typer.Exit(code=gpu_run())`: the benchmark's exit code becomes the process's exit code, so a failed benchmark is visible to whatever called it.
