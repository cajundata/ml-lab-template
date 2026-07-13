`gpu_run` only. The payload runs, and the state machine's most subtle distinction lives here: **"the benchmark failed" and "the benchmark could not be run" are different events with different consequences.**

- **The benchmark runs to completion but fails** (a required probe returns `ok: false`) -> nonzero exit code, **not** an exception. `gpu_run` still pulls the bundle and uploads it to Spaces before the `finally` destroys. A failed GPU run is precisely when you most want the evidence, and that evidence dies with the droplet.
- **The benchmark times out** (1800s) **or transport fails** (scp/ssh) -> `RemoteError`. No pull. The `finally` destroys immediately. **Destroy beats artifact preservation.**

Either way the droplet is destroyed. The only question is whether you get the bundle first.

The pull is two scp's: the remote bundle at `/opt/ml-lab/artifacts/<run-id>/` recursively, plus `/var/log/cloud-init-output.log` as `bootstrap.log`. Then `upload_bundle` pushes everything to `s3://<bucket>/ml-pathway/phase0/<run-id>/`, so the evidence outlives the machine.

**Exit:** always -> `destroyed`. The one escape is a **`kill -9` of the local process** — no `finally` executes, nothing tears anything down, and you land in `stranded`. That is the exact scenario the remote self-destruct timer exists for, and it was verified live.
