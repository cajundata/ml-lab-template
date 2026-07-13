SSH is reachable and the droplet has attested, in JSON, that it is `ready` **and** that its `self_destruct_timer_active` is true. This is the **only** state from which the code is willing to do real work on the machine.

**This is where the two entry paths split**, and the difference is the whole distinction between them:

- **`gpu_run`** proceeds straight to `benchmarking`. The droplet is a means to an end and will be destroyed in a `finally` no matter what happens next.
- **`gpu_up` stops here — deliberately leaving the droplet alive.** It is the debug path: you asked for a GPU box you can SSH into. It prints a handoff block with the id, name, region, size, public IP, TTL, self-destruct retry interval, and the exact `make gpu-down DROPLET_ID=<id>` command — closing with *"Self-destruct timer has been verified active on the droplet. Do not power off this droplet. Destroy it."*

So `gpu_up` transitions you into `stranded` **on purpose**. That is safe only because the timer was verified before the handoff — the droplet cannot outlive its TTL even if you walk away.

From here, **any** exception — including `KeyboardInterrupt` — destroys and re-raises. And once `destroy_and_verify` is armed, SIGINT is ignored: a second Ctrl-C does nothing. Proven live at S4.
