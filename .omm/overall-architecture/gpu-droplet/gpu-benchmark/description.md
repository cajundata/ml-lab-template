`scripts/gpu_benchmark.py` — the payload. The one thing the droplet exists to run.

**Self-contained by design**: top-level imports are stdlib only, and it never imports `ml_lab`. torch / transformers / vllm are imported *lazily inside each probe*. That is what lets the file be unit-tested in CI on a machine with no GPU and no CUDA, while only ever *executing* on the droplet (scp'd up, run against `/opt/ml-lab/venv/bin/python`).

Five probes, of which **two are required**:
- **`torch_cuda`** (required) — CUDA is visible, and a small matmul actually runs on the device.
- **`transformers_smoke`** (required) — load `facebook/opt-125m` with transformers and generate 8 tokens on the GPU. **This is Phase 0's GPU-workload proof.**
- `vllm_smoke` (**informational only**) — vLLM does not yet serve on this stack; engine-core init fails. It is deliberately *not* required. Do not "fix" the benchmark by promoting it.
- `system`, `nvidia_smi` (informational) — GPU name, memory, driver, raw `nvidia-smi`.

`run_benchmark()` runs every probe **defensively**: a probe that raises has its traceback written into its own artifact and is marked failed, and the remaining probes still run. **A partial bundle is always written.** The exit code is 0 iff both *required* probes pass — which is what `benchmark-driver` reads and what ultimately becomes `make gpu-run`'s exit code.

Probes are injected as a `name -> callable` mapping, which is what makes the whole defensive-write path testable without a GPU.
