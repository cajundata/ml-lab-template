**Single-GPU capacity on this account flaps across BOTH regions and SKUs, by the minute.** A statically pinned `(size, region)` pair does not work — this was the dominant surprise of S4 live.

Consequences to respect when editing this file:
- `DO_REGION` is a **preferred hint**, not a pin. Do not "simplify" `resolve_region` back into a constant.
- `ACCEPTABLE_SIZE_SLUGS` is an **interchangeable** list (H100, H200, L40S — Hopper first). The run lands on whichever has capacity; the caller must not assume which.
- Capacity can vanish **between** the region resolve and the create POST. That surfaces as a DO **422 "not available in this region"**, which is handled by excluding that exact `(SKU, region)` pair and retrying. **All 422s here are pre-billing** — nothing was created, nothing is owed.
- `LabDropletExistsError` is a safety feature, but it means **one stranded droplet blocks every subsequent run** until it is reaped. That is intentional: if you are hitting it, run `make gpu-audit` — do not work around it.
