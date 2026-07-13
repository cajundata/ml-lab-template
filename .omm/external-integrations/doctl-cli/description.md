The DigitalOcean CLI, shelled out to as a **subprocess** — not a library. Wrapped by `src/ml_lab/gpu/do_client.py`, which appends `-o json` to every call and parses stdout.

Handles droplet **list / get / create / delete**, plus the catalog reads (`size list`, `image list`, `region list`) and the audit sweeps (volumes, snapshots, reserved IPs, load balancers).

Three details that matter:
- **`create_droplet` applies tags atomically** via `--tag-names` in the create call itself. Never create-then-tag — that would leave a window in which a billable droplet exists that `make gpu-audit` cannot see.
- **It deliberately does NOT pass `--wait`**, so the droplet id returns immediately and can be printed before anything else can go wrong.
- **`get_droplet` distinguishes "gone" from "could not tell"** — `None` on a clean 404, `DOClientError` on anything else. `_poll_until_absent` is built entirely on that distinction, and confusing the two would mean reporting a droplet destroyed when it is still billing.

Authenticated via `doctl auth init`, which is **machine-local** and does not travel with a `git clone`.
