`src/ml_lab/gpu/cli.py`, reached through `scripts/do_gpu.py` — a four-line entry point that imports this Typer `app` and calls it.

**Deliberately not installed on `$PATH`.** Unlike `ml-lab`, these commands provision hardware that bills by the hour, so they are invoked as an explicit script path. Nobody should reach `run` by tab-completion.

Four commands: `audit` (free — is anything billing?), `down` (the always-safe reap), `up` (create and hand off, leaving the droplet **alive**), `run` (the full lifecycle, **always** destroying).

Two of them have exit-code contracts that matter: `audit` exits nonzero when the account is dirty, so it works as a preflight/CI check; and `run` propagates the benchmark's own exit code, so a failed GPU run is visible to whatever invoked it.
