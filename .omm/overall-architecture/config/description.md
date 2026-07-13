Two pinned-constants modules. Neither has logic worth speaking of; both are load-bearing.

**`src/ml_lab/config.py`** — the local ML half. Anchors every path to `REPO_ROOT` (computed from `__file__`, so nothing depends on the working directory), and pins the determinism knobs: `SMOKE_SEED = 20260706`, 240 rows split 180/60, the 4 feature names, the target name, and the MLflow experiment name.

**`src/ml_lab/gpu/constants.py`** — the GPU half, and the more interesting of the two. It pins:
- `ACCEPTABLE_SIZE_SLUGS` — an *interchangeable* list of single-GPU SKUs (H100, H200, L40S), Hopper first. Create lands whichever has capacity.
- `DO_REGION = "nyc2"` — a **preferred hint only**, not a pin. The live region is resolved per SKU at create time.
- `DO_IMAGE_SLUG = "gpu-h100x1-base"` — proven to boot cleanly on Hopper.
- `SPACES_REGION = "nyc3"` — nyc2 has no Spaces; nyc3 is the nearest.
- The full timeout budget (TTL 7200s, SSH 600s, bootstrap 1800s, benchmark 1800s, destroy-poll 600s, self-destruct retry 300s).
- `validate_timeout_budget()` — asserts SSH + bootstrap + benchmark fits *under* the TTL, so the work can never outlive the safety net.

The header says "no fallback" and means it: changing a slug here is a deliberate decision, not something the code papers over at runtime.
