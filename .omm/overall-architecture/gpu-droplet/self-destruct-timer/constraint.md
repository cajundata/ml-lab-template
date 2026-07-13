**Never power off a lab droplet.** A stopped GPU droplet still bills — *and* a stopped droplet is not running systemd, which means this timer is not running either. Powering off disarms the exact safety net that would otherwise have saved you. Destroy it: `make gpu-down DROPLET_ID=<id>`.

Other invariants:
- The timer must be **armed and verified active before** anything slow or failure-prone runs in `runcmd`, and **re-verified** before the readiness marker is written. Do not reorder cloud-init so the expensive vLLM install happens before the timer is up.
- The destroy token must be a **separate, destroy-scoped** DO token — never the token doctl authenticates with — and it lives only in `/etc/ml-lab/run.env` at mode **0600**.
- `OnUnitActiveSec` retry is what makes this robust to a transient API failure. A one-shot timer would be a safety net with a hole in it.
