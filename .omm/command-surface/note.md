**The command you should know before any other: `make gpu-audit`.** It is free, it is safe, and it answers the only question that ever really matters here — *is anything billing right now?* If it prints clean, you owe nothing. If it prints dirty, it hands you the exact `make gpu-down DROPLET_ID=<id>` line for each droplet it found.

Run it after any GPU session, after any crash, and any time you are unsure. It is also the first sanity check when setting up a new machine.

**`make gpu-up` vs `make gpu-run`**, since the names are close and the consequences are not:
- **`gpu-up`** creates a droplet, verifies it, and **leaves it alive** so you can SSH in. *You* are now responsible for reaping it. It says so, in its handoff block.
- **`gpu-run`** creates a droplet, runs the benchmark, pulls and uploads the artifacts, and **always destroys it**.

If you only ever type one of them, type `gpu-run`.
