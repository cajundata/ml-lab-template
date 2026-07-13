`make gpu-up` — the **debug path**. Creates a GPU droplet, waits for its public IP, waits for SSH, waits for the bootstrap marker attesting the self-destruct timer is armed — and then **leaves the droplet alive** so you can log into it.

It prints a handoff block: id, name, region, size, public IP, TTL, self-destruct retry interval, and the exact `make gpu-down DROPLET_ID=<id>` line — closing with *"Self-destruct timer has been verified active on the droplet. Do not power off this droplet. Destroy it."*

**You are now responsible for that droplet.** The on-droplet timer will reap it at TTL (2h) if you walk away, but until then it bills.

On **any** post-create failure — including `KeyboardInterrupt` — it destroys and re-raises. A preflight failure creates nothing.

`--ttl-seconds N` also flips `enforce_budget=False`. That is the deliberate escape hatch for **short-fuse self-destruct testing**: a 5-minute TTL cannot possibly fit the SSH + bootstrap + benchmark budget, and the whole point of passing it is to watch the droplet reap itself quickly. Do not use it for real runs.
