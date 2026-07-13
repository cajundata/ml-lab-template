**Defense in depth: three independent layers, and they must stay independent.**

1. **Local `try`/`finally`** — `gpu_run` destroys in a `finally`, so a normal exception, a failed benchmark, or a clean exit all reap the droplet.
2. **SIGINT masking** — once `destroy_and_verify` is armed, Ctrl-C is ignored and restored only in a `finally`. A second Ctrl-C mid-teardown does nothing. Operator impatience cannot strand a droplet.
3. **Remote systemd timer** — the droplet DELETEs *itself* at TTL, using its own destroy-scoped token and the metadata service to learn its own id. This layer needs **nothing** from the local machine, which is exactly why it survives `kill -9`, a laptop lid closing, or a network partition.

Layer 3 is the one that actually saves you, and it is the one easiest to accidentally disable. Two ways to do that, both forbidden:
- **Powering off the droplet.** A stopped GPU droplet still bills, and a stopped droplet is not running systemd — so the timer that would have reaped it is not running either. **`down` always means destroy.**
- **Reordering cloud-init** so the slow, failure-prone install happens before the timer is armed and verified.

**Billing starts at `provisioning -> booting`**, when DO returns the droplet id — *not* when the droplet becomes useful. Everything before that line (all four preflight gates, every capacity 422) is free.
