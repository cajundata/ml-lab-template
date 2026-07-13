**A billable droplet nobody is watching. This is the failure state the entire repo is built to prevent.**

Two ways in, and the difference matters:

- **Deliberately, via `gpu_up`.** This is the debug path. You asked for a live GPU box to SSH into, and you got one — after the code first *proved* its self-destruct timer was armed. The handoff block tells you exactly how to reap it. This is a controlled stay in a dangerous state.
- **Accidentally, via `kill -9`** (or a hard crash, or a laptop that loses power). No `finally` runs. No SIGINT handler fires. The local process is simply gone, and it takes every local safety layer with it.

The second case is why layer 3 exists. **The remote systemd self-destruct timer needs nothing from the local machine.** It sources its own destroy-scoped token from `/etc/ml-lab/run.env`, asks the DO metadata service at `169.254.169.254` for its own droplet id, and DELETEs itself — retrying every 300s until it succeeds. It was verified live at S4 by actually `kill -9`-ing a run and watching the droplet reap itself.

**But the timer bounds the damage; it does not eliminate it.** You can still burn up to the full 2h TTL. So:

- **`make gpu-audit`** is the way out — it reports every billable lab resource, exits nonzero when dirty, and prints the exact destroy command for each droplet it finds.
- **`make gpu-down DROPLET_ID=<id>`** is the always-safe reap.
- **Never power the droplet off.** A stopped GPU droplet still bills — and a stopped droplet is not running systemd, so the self-destruct timer is not running either. Powering off *disarms the very thing that would have saved you.*
