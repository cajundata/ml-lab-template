The GPU droplet's lifecycle as a state machine — the single most important thing to understand in this repo.

Read it with one question in mind: **from every state, what happens if this goes wrong?** The answer is always the same, and that is the design. Every edge that is not the happy path points at `destroyed`.

The states:
- **preflight** — gates run *before any droplet exists*. A failure here creates nothing, so there is nothing to owe.
- **provisioning** — the multi-SKU / multi-region capacity retry. Self-loops on a DO 422, which is **pre-billing**.
- **booting** — cloud-init arms the self-destruct timer, verifies it, installs the stack, verifies it again, and only then writes the readiness marker. **Billing starts the moment the droplet id is returned, not when it is ready.**
- **verified** — SSH reachable, marker attests `ready` *and* `self_destruct_timer_active`.
- **benchmarking** — the payload runs. `gpu_run` only.
- **destroyed** — droplet absent **and** account audit clean. The terminal, safe state.
- **stranded** — the state the whole repo exists to prevent: a billable droplet nobody is watching.

Note the two edges *into* `stranded`, and that both have an exit. `gpu_up` puts you there **deliberately** (it is the debug path — it leaves a verified droplet alive so you can log into it). A `kill -9` puts you there **accidentally**. In both cases the remote self-destruct timer reaps the droplet at TTL with **zero local involvement** — which was proven live at S4, by actually `kill -9`-ing a run and watching the droplet delete itself.
