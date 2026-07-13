`src/ml_lab/gpu/create.py` — every safety gate lives inside `create_lab_droplet()`, so both `gpu_run` and `gpu_up` inherit them by construction rather than by discipline.

**The preflight gates, in order** (all before any droplet exists, so a failure strands nothing):
1. `list_lab_droplets()` must be empty — refuse to create a second lab droplet (`LabDropletExistsError`).
2. `validate_constants()` — every SKU in `ACCEPTABLE_SIZE_SLUGS` and the pinned image must exist in the account.
3. `probe_destroy_token()` — never provision what we cannot destroy.
4. `validate_timeout_budget()` — the work must fit under the TTL.

**The capacity retry** (`_create_with_region_retry`) is the file's real complexity, and it is earned. Single-GPU capacity on DO flaps across **both** regions and SKUs by the minute, so: try each interchangeable SKU in order (Hopper first), resolve the region DO *currently* reports for it, and attempt the create. Capacity can vanish between the resolve and the create POST — that surfaces as a DO 422 "not available in this region", which excludes that exact `(SKU, region)` pair and moves on. When nothing has capacity, re-probe on a bounded poll (windows reopen within minutes) until `CREATE_CAPACITY_WAIT_SECONDS`. Any non-capacity error re-raises immediately.

**Every 422 in that loop is pre-billing.** Nothing was created, so nothing is owed.

`resolve_region()` treats `DO_REGION` as a *preference*, not a pin: use it when DO reports the size available there, else take the first region DO reports, and when DO reports no regions at all (common for GPU SKUs) fall back to `DO_REGION` and let the create call's 422 be the authority. Tags are applied **atomically** via `--tag-names` at create time — never create-then-tag, which would leave a window where a droplet exists that the audit cannot see.
