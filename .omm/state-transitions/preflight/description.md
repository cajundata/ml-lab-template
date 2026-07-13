**Free. Nothing exists yet, so nothing can be owed.** Four gates, all inside `create_lab_droplet` (`create.py:140`) so both `gpu_up` and `gpu_run` inherit them by construction:

1. **No lab droplet already exists** — `list_lab_droplets()` must be empty, else `LabDropletExistsError`. Refusing to create a second droplet means one stranded droplet blocks all runs until reaped. That is a feature.
2. **`validate_constants()`** — every SKU in `ACCEPTABLE_SIZE_SLUGS` and the pinned image must exist in the account. Note it deliberately does **not** gate on capacity: capacity is resolved live at create time, because it flaps by the minute.
3. **`probe_destroy_token()`** — `DELETE /v2/droplets/1`; a 404 proves the token authenticates and carries delete scope. **Never provision what you cannot destroy.**
4. **`validate_timeout_budget()`** — SSH + bootstrap + benchmark must fit *under* the TTL, so the work can never outlive the safety net.

Env loading (`load_gpu_env`, and for `gpu_run` also `load_spaces_env`) happens even earlier, in `lifecycle.py`, and fails loud naming *all* missing vars at once.

**Exit:** all gates pass -> `provisioning`. Any gate fails -> nothing was created; the run dies here for free.
