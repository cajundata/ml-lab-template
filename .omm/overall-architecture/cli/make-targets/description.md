`Makefile` — the memorable surface, and the one humans actually type. Nine targets:

`setup` (`uv sync --dev`), `test` (`uv run pytest` — 193 tests), `train`, `evaluate`, `mlflow-ui` (serves the local file store at 127.0.0.1:5000), `clean`, and the four GPU targets: `gpu-run`, `gpu-up`, `gpu-down`, `gpu-audit`.

One target has a guard worth noting: `gpu-down` hard-requires `DROPLET_ID` and fails with a usage message rather than doing anything clever if it is missing. Destroying the wrong droplet — or nothing at all while believing you destroyed something — is exactly the failure this repo exists to prevent.
