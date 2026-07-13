`https://api.digitalocean.com/v2` — reached directly (not via doctl) in exactly two places, and both are about **destruction**.

**1. `probe_destroy_token()` (local, `do_client.py`, via `requests`).** Sends `DELETE /v2/droplets/1` against a known-nonexistent droplet id. **A 404 means good**: the token authenticates *and* carries delete scope. A 401/403 means the token is rejected. Anything else is unexpected and raises.

This runs as a **preflight gate, before any droplet is created** — the principle being *never provision what you cannot destroy*. It is also why the REST API is used at all rather than just doctl: the probe has to test a **specific credential's scope**, which means presenting that token directly. Master plan §3 specifies the split.

**2. The self-destruct script (remote, `/opt/ml-lab/self_destruct.sh`).** The droplet DELETEs *itself* against this same endpoint, using the destroy-scoped token planted by cloud-init at mode 0600 — retrying every 300s until it succeeds.

The **destroy-scoped token is a separate credential** from whatever doctl authenticates with. The droplet is given exactly the authority to delete itself, and nothing more.
