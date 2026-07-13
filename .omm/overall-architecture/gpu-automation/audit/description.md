`src/ml_lab/gpu/audit.py` — answers one question: *is this DigitalOcean account billing-clean?*

`collect_audit()` gathers lab droplets **plus** related billable resources — volumes, snapshots, reserved IPs, load balancers — into an `AuditReport`. `is_clean()` is true only if every one of those lists is empty. `format_report()` renders it for a human, and for a dirty account prints each droplet's id, name, status, region, size, image, public IP, age, TTL expiry, whether it is **OVERDUE** (past its `ttl-expiry-<epoch>` tag), and the exact `make gpu-down DROPLET_ID=<id>` command to reap it.

Matching is two-pronged for droplets — the `ml-lab` tag **or** the `ml-lab-gpu-` name prefix — so a droplet whose tagging somehow failed still gets caught by its name. Related resources match on the lab tag set only; Phase 0 never creates them, so any lab-tagged hit is already a failure worth reporting.

It is used in two places, which is why it matters more than a reporting module usually would: as the human-facing `make gpu-audit` (exiting nonzero when dirty), and as the **final assertion inside `destroy_and_verify`**. Teardown is not complete until this says clean.
