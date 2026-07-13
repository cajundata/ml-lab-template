Informational. The "what am I actually running on?" record — which matters more than usual here, because **the run does not know in advance which GPU it landed on.**

Capacity flaps across SKUs, so `create` takes whichever of H100 / H200 / L40S has capacity at that moment. This probe is how the artifact bundle records the answer: `torch.cuda.get_device_name(0)` and `get_device_properties(0).total_memory`, plus OS, Python version, and CUDA visibility.

Best-effort by design — it reports `ok: true` even if the torch import fails, stashing the traceback in `error` rather than failing the run. It is context, not a gate.
