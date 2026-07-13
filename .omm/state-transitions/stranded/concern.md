**The self-destruct timer bounds the damage. It does not eliminate it.**

A droplet stranded at minute 1 of a 2-hour TTL burns ~2 hours of GPU time before it reaps itself. On single-GPU H100/H200 hourly rates, that is real money for zero work.

- **`gpu_up` puts you here on purpose**, and nothing watches you afterwards. Run it, get distracted, close the laptop — the TTL is your only backstop.
- **Nothing runs `make gpu-audit` on a schedule.** A stranded droplet is noticed when a human looks, or when the TTL fires. A cron/CI audit would close that window.
- **`LabDropletExistsError` is the accidental tripwire** — the next `gpu-run` refuses to start because a droplet already exists. That is often *how you find out*. When it happens: run `make gpu-audit`, then `make gpu-down`. Never work around it.

And the trap that turns a bounded problem into an unbounded one: **powering the droplet off does not stop billing, but it does stop systemd — which stops the self-destruct timer.** A powered-off droplet bills forever and has no way to reap itself. `down` always means destroy.
