**The terminal, safe state — and the only definition of "done" this repo accepts.**

Reached exclusively through `destroy_and_verify` (`teardown.py`). Success means **the droplet does not exist AND the account audit is clean**. Not "the DELETE returned 2xx" — an accepted DELETE is a promise, not a fact.

What it actually does:
1. **Masks SIGINT** (`SIG_IGN`, restored in a `finally`) and prints *"Teardown in progress — do not interrupt."* A second Ctrl-C does nothing.
2. Calls `destroy_droplet`, then **ignores its return value entirely** — a 404, an accepted 2xx, and a transient error all just mean "poll anyway".
3. **`_poll_until_absent`** re-checks `get_droplet` until it returns `None`, up to 600s. A transient `DOClientError` is treated as "unknown, keep polling", never as "probably fine".
4. **`collect_audit()` + `is_clean()`** — droplets, volumes, snapshots, reserved IPs, load balancers must *all* be empty. A dirty audit raises `TeardownError` with the full report.

And if teardown itself fails, its `TeardownError` **overrides any error already in flight** (the original chained as `__context__`). A failed benchmark is a bad day. A stranded billable droplet is worse — so it gets to be the loudest alarm in the system.

Also reachable for free from `preflight` (a gate failed; nothing was ever created) and manually from anywhere via `make gpu-down DROPLET_ID=<id>`.
