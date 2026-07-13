**The design rule to carry away:** every failure mode degrades toward `destroyed`. There is no state from which "we could not tell what happened" resolves to "leave it running". `_poll_until_absent` treats an ambiguous API error as "keep polling" rather than "assume it worked". `wait_for_bootstrap` treats an unparseable marker as "not ready" rather than "probably fine". `destroy_and_verify` treats a dirty audit as failure even after a successful DELETE.

**Where to look when a run misbehaves**, by state:
- Stuck in `provisioning` -> capacity. The retry loop prints which `(SKU, region)` pairs it tried and skipped.
- Stuck in `booting` -> almost certainly `pip install vllm`. Check `artifacts/<run-id>/bootstrap.log` (the pulled cloud-init log).
- Reached `verified` but the benchmark failed -> the bundle was still pulled and uploaded. Read `benchmark.json` and the per-probe JSONs; a probe that raised has its traceback written into its own artifact.
- Anything at all, at any time -> **`make gpu-audit`**. It exits nonzero if the account is dirty and prints the exact `make gpu-down` command for each droplet it finds.
