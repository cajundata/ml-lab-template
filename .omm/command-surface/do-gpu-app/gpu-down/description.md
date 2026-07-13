`make gpu-down DROPLET_ID=<id>` — the always-safe manual reap. Routes through `destroy_and_verify` exactly like every automatic teardown does; there is no separate, lesser destroy path.

Requires the droplet id explicitly (the `Makefile` refuses without it). It does not guess, does not destroy "the one it found", and does not accept a name.

What it guarantees: SIGINT is ignored while teardown is armed (a second Ctrl-C does nothing), the droplet is **polled to absence** rather than trusted to a 2xx, and the account audit must come back clean afterwards. Anything less raises `TeardownError`.

**This is what `down` means. It destroys.** It does not power off, stop, or shut down — because a stopped GPU droplet still bills, and a stopped droplet is not running the systemd timer that would otherwise have reaped it.
