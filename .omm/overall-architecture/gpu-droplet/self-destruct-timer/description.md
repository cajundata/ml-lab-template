`ml-lab-self-destruct.timer` + `.service` + `/opt/ml-lab/self_destruct.sh` — installed on the droplet by cloud-init. **This is the layer that survives everything else failing.**

The script is six lines: source `/etc/ml-lab/run.env` (mode 0600, carrying the destroy-scoped token), ask the DO metadata service at `169.254.169.254` for its *own* droplet id, and `DELETE /v2/droplets/<id>` against the DO API. The droplet deletes itself. It needs no help from the machine that created it.

The timer is `OnBootSec=<TTL>` (7200s default) with `OnUnitActiveSec=<SELF_DESTRUCT_RETRY_SECONDS>` (300s) — so it does not fire once and give up. If the DELETE fails (network blip, API hiccup), it **retries every 5 minutes, forever**.

Two properties make this trustworthy rather than aspirational:
- **Bootstrap asserts the timer is active twice** — immediately after enabling it, and again at the end of `runcmd` — and only *then* writes the readiness marker. The local side refuses to use a droplet whose marker does not attest `self_destruct_timer_active: true`.
- **It was proven live at S4:** a `kill -9`'d run, with zero local involvement, was reaped by this timer.

This is why the repo's first law is "`down` always means destroy, never power off". A **stopped** GPU droplet still bills — and a stopped droplet is not running systemd, so the self-destruct timer is not running either. Powering off disarms the very safety net that would have saved you.
