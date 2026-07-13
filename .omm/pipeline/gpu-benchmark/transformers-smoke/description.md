**REQUIRED. This is Phase 0's GPU-workload proof** — the probe that says a *real model* runs on this hardware, not just a matmul.

Loads `facebook/opt-125m` with transformers (`AutoTokenizer` + `AutoModelForCausalLM`), moves it to `cuda`, and generates 8 tokens greedily (`do_sample=False`, so it is deterministic). `ok` requires **both** `load_ok` and `generate_ok` (the latter meaning it actually produced new tokens, counted by differencing the output and input sequence lengths).

Records `device`, `load_ok`, `generate_ok`, `token_count`, `elapsed_s`, and — on failure — the full traceback.

**It exists because vLLM does not work on this stack.** `vllm_smoke` was originally the required proof; when its engine-core init turned out to fail, the requirement moved here. This probe deliberately exercises **the same torch/CUDA path** vLLM would, but without vLLM's serving machinery — so Phase 0 still gets a genuine end-to-end "a language model generated tokens on this GPU" proof, and vLLM's problems stay in Phase 5 where they belong.

Note it pulls the model from Hugging Face at run time, so the droplet needs outbound network. **Proven live at S4** on Hopper.
