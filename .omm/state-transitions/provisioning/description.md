The capacity hunt (`_create_with_region_retry`, `create.py:90`). **Still free — every 422 here is pre-billing.**

Single-GPU capacity on this account flaps across **both** regions and SKUs by the minute, so there is no static answer to "where do I create this". The loop:

- Walk `ACCEPTABLE_SIZE_SLUGS` in order (H100 -> H200 -> L40S; Hopper first, proven to boot the pinned image).
- For each, `resolve_region()` asks DO which regions *currently* report the size. `DO_REGION` (nyc2) is used when available, else the first region DO names, else `DO_REGION` as a fallback with the create call's 422 as the authority.
- Attempt the create. Capacity can vanish **between** the resolve and the POST — that is the DO **422 "not available in this region"**. Exclude that exact `(SKU, region)` pair, print which one was skipped, move on.
- When *nothing* has capacity, clear the exclusions and re-probe on a bounded poll (windows reopen within minutes) up to `CREATE_CAPACITY_WAIT_SECONDS` (180s), then give up with a `ConstantsError` naming everything tried.
- **Any non-capacity error re-raises immediately** — this loop is for capacity, not for swallowing real failures.

Tags are applied **atomically** via `--tag-names` in the create call itself. Never create-then-tag: that would leave a window in which a billable droplet exists that `make gpu-audit` cannot see.

**Exit:** DO returns a droplet id -> `booting`. **Billing starts on that line.**
