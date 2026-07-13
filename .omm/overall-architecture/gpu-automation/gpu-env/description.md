`src/ml_lab/gpu/gpu_env.py` — loads `.env` and validates the operator's machine-local configuration, **failing loud and naming every missing variable at once** rather than one per run.

Two frozen dataclasses, validated **separately** and for a reason:
- **`load_gpu_env()` -> `GpuEnv`** — `DO_DROPLET_DESTROY_TOKEN`, `DO_SSH_KEY_IDS` (comma-split), `DO_SSH_KEY_PATH`.
- **`load_spaces_env()` -> `SpacesEnv`** — `SPACES_ACCESS_KEY_ID`, `SPACES_SECRET_ACCESS_KEY`, `SPACES_BUCKET`.

They are split because only `gpu_run` uploads artifacts. `gpu_up` must not fail merely because Spaces credentials are unset — but `gpu_run` must fail *before creating a droplet* if they are, which is why both loads happen in preflight.

**The two setup mistakes this catches** (both cost real time at S4 live, both documented in `CLAUDE.md`): `DO_SSH_KEY_IDS` must be the **numeric ID** from `doctl compute ssh-key list`, not the key's name; and `DO_SSH_KEY_PATH` must be the **private** key, not the `.pub`.
