**`down` always means destroy. Never power off a lab droplet — a stopped GPU droplet still bills.**

- Every teardown path — success, exception, `KeyboardInterrupt`, benchmark failure, manual `make gpu-down` — must route through `destroy_and_verify`. Do not add a code path that calls `do_client.destroy_droplet` directly.
- Success is **droplet absent AND audit clean**. An accepted DELETE is a promise, not a fact; poll to absence.
- SIGINT stays ignored for the whole teardown window. Do not "improve" this by making it interruptible.
- If teardown fails, its `TeardownError` **must** win over any in-flight error (the original is chained as `__context__`). A stranded billable droplet is the loudest alarm in the system.
