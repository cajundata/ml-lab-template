The Hugging Face Hub — called **from the droplet**, never from this machine.

`probe_transformers_smoke` downloads `facebook/opt-125m` (the pinned `SMOKE_MODEL_ID`) at benchmark time via `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained`, moves it to CUDA, and generates 8 tokens greedily. That is Phase 0's **required GPU-workload proof**: a real language model generating real tokens on real hardware.

125M parameters is chosen to be small enough that the download and load are not the interesting part — the GPU execution is.

**This is the one required probe with an external dependency outside DigitalOcean**, and it is worth being clear-eyed about: the droplet needs outbound network, and a Hub outage, a rate limit, or a network-restricted droplet fails a *required* probe on an otherwise perfectly healthy GPU. There is no local cache and no mirror. Baking the model into the image would remove that coupling.
