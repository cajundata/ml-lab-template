Every one of these seams was **mocked** through slices S1-S3. S4 was the first live execution against real infrastructure, and most of what is written here is what S4 taught.

The DigitalOcean integration is deliberately split across **two transports**, which looks redundant until you see why: doctl is convenient and handles auth, but the destroy-token probe needs to test a *specific credential's* scope, which means going to the REST API directly with that token in hand. Master plan §3 specifies this split.

Setting up a new machine means rebuilding all the machine-local credential state (`README.md` -> "GPU prerequisites"):
1. `doctl auth init`
2. An SSH key **registered with DO** — `DO_SSH_KEY_IDS` must be the **numeric ID** from `doctl compute ssh-key list`, not the name; `DO_SSH_KEY_PATH` must be the **private** key, not the `.pub`. Both were real, time-costing mistakes at S4.
3. `.env` with the destroy token and Spaces keys.
4. `make gpu-audit` should print clean.

The full live Results log and all 10 findings are in `docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md`.
