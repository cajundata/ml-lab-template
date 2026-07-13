The GPU benchmark's output, and **the only thing that survives the droplet**.

Written on the droplet at `/opt/ml-lab/artifacts/<run-id>/`, then pulled down by `benchmark.pull_artifacts()` to `artifacts/<run-id>/` and archived by `spaces.upload_bundle()` to `s3://<bucket>/ml-pathway/phase0/<run-id>/`.

Contents:
- `benchmark.json` — the roll-up: run id, per-probe pass/fail, and `required_ok`.
- `benchmark.log` — the same, human-readable.
- `torch_cuda.json`, `transformers_smoke.json`, `vllm_smoke.json`, `system.json` — per-probe results. **A probe that raised has its full traceback written into its own file.**
- `nvidia-smi.txt` — raw, including stderr, so a failing call is still diagnosable.
- `bootstrap.log` — the droplet's `/var/log/cloud-init-output.log`, pulled separately.

That last one is deliberate forensics. When a droplet misbehaves, the cloud-init log is usually the only place the reason is recorded — and it dies with the machine.

**A partial bundle is always written**, even when probes fail, and a completed-but-failed benchmark still gets pulled and uploaded before teardown. A failed GPU run is exactly when you most want the evidence.

`make clean` never touches `artifacts/`. GPU runs cost real money; their evidence is not automatically deleted.
