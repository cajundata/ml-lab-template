The real `ssh` and `scp` binaries, shelled out to via `src/ml_lab/gpu/remote.py`. Mirrors `do_client`'s subprocess pattern: one external dependency, one module, mocked in every test.

`SSH_OPTS` is hardened and **non-interactive by construction**:
- `BatchMode=yes` — never prompt. A hung password prompt in an automated GPU run is a stranded droplet.
- `IdentitiesOnly=yes` — use *only* the `-i` key, not whatever the agent happens to be holding.
- `StrictHostKeyChecking=accept-new` + `UserKnownHostsFile=/dev/null` — ephemeral droplets never pollute `known_hosts`.
- `ConnectTimeout=15` — bound every connect.

Used for four things: `wait_for_ssh` (poll `ssh ... true` until it exits 0), `wait_for_bootstrap` (poll and **JSON-parse** the readiness marker), `scp_up` (deliver the benchmark), and `scp_down` (pull the bundle and the cloud-init log).

Every failure — nonzero exit or timeout — becomes a `RemoteError`. The waiters take injectable `now` and `sleep`, so a test of the 600-second deadline runs instantly instead of taking ten minutes.

**Setup gotcha:** `DO_SSH_KEY_PATH` must point at the **private** key, not the `.pub`, and `DO_SSH_KEY_IDS` must be the key's **numeric DO ID**, not its name. Both were real, time-costing mistakes at S4 live.
