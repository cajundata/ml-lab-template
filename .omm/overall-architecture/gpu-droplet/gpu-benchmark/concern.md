**vLLM does not serve on this stack.** `probe_vllm_smoke` fails at engine-core init. It is marked informational precisely so this does not block Phase 0, and the required GPU-workload proof was moved to `transformers_smoke` (which exercises the same torch/CUDA path without vLLM's serving machinery).

Do **not** resolve this by promoting `vllm_smoke` back to required, and do not treat its failure as a regression. vLLM belongs to the **Phase-5 cloud-serving track**.

When someone does pick it up: **capture the engine *subprocess* stderr first.** vLLM forks an engine-core process, and the parent's traceback is not where the real error lives — that is the single fact most likely to save a day of debugging.
