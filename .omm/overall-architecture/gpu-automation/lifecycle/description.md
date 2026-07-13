`src/ml_lab/gpu/lifecycle.py` — the spine. Read this file first; its docstring states the safety contract more precisely than any diagram can.

Both entry points share `create -> wait_for_public_ip -> wait_for_ssh -> wait_for_bootstrap`. They diverge only at the end:

**`gpu_up`** (debug path) leaves a fully verified droplet **alive** and prints a handoff block: id, name, region, size, public IP, TTL, and the exact `make gpu-down` command to reap it — ending with "Do not power off this droplet. Destroy it." On *any* post-create exception, including `KeyboardInterrupt`, it destroys and re-raises. A preflight failure creates no droplet, so there is nothing to destroy.

**`gpu_run`** (full path) runs the benchmark, pulls the bundle, uploads it to Spaces, and **always destroys in a `finally`**. A benchmark that completes but fails still gets its partial bundle pulled and uploaded before the destroy. A benchmark that *raises* (timeout, transport failure) destroys immediately — destroy beats artifact preservation.

One subtlety worth internalizing: if `destroy_and_verify` itself fails, its `TeardownError` **overrides** whatever error was in flight, with the original chained as `__context__`. A failed benchmark is a bad day; a stranded billable droplet is a worse one, so it gets to be the loudest alarm.

`wait_for_public_ip` also lives here, and it treats a transient `DOClientError` as "no IP yet" rather than a failure — the poll, not any single call, is the truth.
