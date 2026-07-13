`src/ml_lab/gpu/remote.py` — the SSH seam. Mirrors `do_client`'s subprocess pattern: shell out to the real `ssh` / `scp` binaries, isolate it behind one module, mock it in every test.

`SSH_OPTS` is hardened and non-interactive by design — `BatchMode=yes` (never prompt), `IdentitiesOnly=yes` (use only the `-i` key, not whatever the agent happens to hold), `StrictHostKeyChecking=accept-new`, a bounded `ConnectTimeout`, and `UserKnownHostsFile=/dev/null` so ephemeral droplets never pollute `known_hosts`.

- **`wait_for_ssh`** polls `ssh ... true` until it exits 0.
- **`wait_for_bootstrap`** polls `/opt/ml-lab/bootstrap-ready.json` and **parses it as JSON** — not a string match — requiring `ready` **and** `self_destruct_timer_active` to both be exactly `True`. A droplet that cannot arm its own self-destruct never satisfies this, and so never gets used.
- **`scp_up` / `scp_down`** are one-shot: any nonzero exit or timeout raises `RemoteError`.

Both waiters take injectable `now` and `sleep`, so a test for the 600-second deadline runs instantly instead of taking ten minutes.
