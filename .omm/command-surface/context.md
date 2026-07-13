`make setup` (`uv sync --dev`) is enough for the entire local half — train, evaluate, test, MLflow UI. No credentials, no network, no accounts.

The GPU half additionally needs **machine-local state that does not travel with a `git clone`** (full checklist in `README.md` -> "GPU prerequisites"):

1. `doctl auth init` — the DO CLI auth is per-machine.
2. An SSH key **registered with DigitalOcean**, with `DO_SSH_KEY_IDS` set to its **numeric ID** (`doctl compute ssh-key list`) — *not* its name — and `DO_SSH_KEY_PATH` pointing at the **private** key (*not* the `.pub`). Both of these were real, time-costing mistakes at S4 live.
3. `cp .env.example .env`, then fill in the destroy-scoped DO token and the Spaces keys.
4. Sanity check: **`make gpu-audit` should print clean.**
