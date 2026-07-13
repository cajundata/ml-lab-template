`src/ml_lab/gpu/benchmark.py` — the *local* driver for the remote benchmark. It never touches a live droplet itself; it composes the `remote` seam, which is mocked in every test.

**`deliver_and_run_benchmark()`** scp's `scripts/gpu_benchmark.py` up to `/opt/ml-lab/gpu_benchmark.py`, runs it over SSH against the droplet's venv Python, and **returns its exit code**. The distinction it draws is the important part:
- A benchmark that **runs to completion but fails** returns a nonzero code. `gpu_run` still pulls and uploads the partial bundle — a failed GPU run is exactly when you most want the evidence.
- A benchmark that **times out**, or a transport failure, raises `RemoteError`. `gpu_run`'s `finally` then destroys without pulling. Destroy beats artifact preservation.

**`pull_artifacts()`** recursively scp's the remote bundle down to `artifacts/<run-id>/`, then separately pulls `/var/log/cloud-init-output.log` as `bootstrap.log`. That second pull is deliberate forensics: when a droplet misbehaves, the cloud-init log is usually the only place the reason is recorded, and it dies with the droplet.

Local paths are module-anchored via `Path(__file__).resolve().parents[3]`, so nothing depends on the working directory.
