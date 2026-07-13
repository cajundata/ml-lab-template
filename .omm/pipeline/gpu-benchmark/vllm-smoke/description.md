**Informational only — and it currently FAILS. That is expected. Do not treat it as a regression.**

Imports vLLM, loads `facebook/opt-125m`, generates 8 tokens. On this stack it never gets that far: **engine-core init fails.**

It is deliberately excluded from `required_ok`, which is why `make gpu-run` still exits 0 on a healthy droplet. Phase 0's required GPU-workload proof was moved to `transformers_smoke`, which exercises the same torch/CUDA path without vLLM's serving machinery.

**Do not "fix" the benchmark by promoting this back to required.** vLLM is the **Phase-5 cloud-serving** engine, and its engine-core tuning is that phase's problem, not this one's.

When someone does pick it up, one fact will save a day of debugging: **capture the engine *subprocess* stderr first.** vLLM forks an engine-core process, and the parent's traceback — which is all this probe currently records — is not where the real error lives.

(Ironically, `pip install vllm` in cloud-init is still what provisions torch + CUDA on the droplet, and it dominates bootstrap time. So vLLM earns its place on the machine even though it cannot serve.)
