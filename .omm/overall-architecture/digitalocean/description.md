External. The cloud provider, reached two different ways on purpose.

**`doctl` (subprocess)** — list, get, create, and delete droplets; list sizes, regions, images, volumes, snapshots, reserved IPs, load balancers. Authenticated machine-locally via `doctl auth init`; that auth does **not** travel with a `git clone`.

**The DO REST API (`api.digitalocean.com/v2`, via `requests`)** — used for exactly one thing locally: `probe_destroy_token()`, which sends `DELETE /v2/droplets/1` against a known-nonexistent droplet. A **404 means good** (the token authenticates and carries delete scope); 401/403 means the token is rejected. This runs as a preflight gate *before any droplet is created*, so we never provision something we lack the credentials to destroy.

The same REST endpoint is called a third way — from *inside* the droplet, by the self-destruct timer, using the destroy-scoped token planted by cloud-init.

**The landmine:** `doctl -o json` writes errors, **including 404**, to STDOUT rather than stderr. Every error path in `do_client.py` concatenates both streams before inspecting them. Getting this wrong produced a false `TeardownError` at S4 live that would have broken every run.
