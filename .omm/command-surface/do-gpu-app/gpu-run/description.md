`make gpu-run` — the full lifecycle, and **the one GPU command that always cleans up after itself.** If you only ever type one, type this one.

Create -> wait for IP -> wait for SSH -> wait for bootstrap -> scp the benchmark up -> run it over SSH -> pull the bundle -> upload it to DO Spaces -> **destroy, in a `finally`**.

Preflight (both the GPU env *and* the Spaces credentials) runs before any droplet exists, so a misconfiguration strands nothing.

The failure semantics are the interesting part:
- A benchmark that **completes but fails** returns a nonzero exit code. The partial bundle is **still pulled and still uploaded** before the destroy — a failed GPU run is exactly when you most want the evidence, and that evidence dies with the droplet.
- A benchmark that **times out**, or any transport failure, raises. No pull. Destroy immediately. **Destroy beats artifact preservation.**
- If teardown itself fails, its `TeardownError` **overrides** whatever error was in flight. A stranded billable droplet is the loudest alarm in the system.

The benchmark's exit code becomes the process's exit code (`raise typer.Exit(code=gpu_run())`).
