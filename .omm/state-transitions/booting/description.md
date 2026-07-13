**The meter is running.** The droplet id exists, so DigitalOcean is billing — regardless of whether the machine is useful yet. This is the longest and least observable state in the machine.

Three sequential waits on the local side, all with injectable clocks so their deadline tests run instantly:
- **`wait_for_public_ip`** (600s) — polls `get_droplet` for a public IPv4. A transient `DOClientError` counts as "no IP yet", not as failure: the poll is the truth, not any single call.
- **`wait_for_ssh`** (600s) — polls `ssh ... true` until it exits 0.
- **`wait_for_bootstrap`** (**1800s**) — polls `/opt/ml-lab/bootstrap-ready.json` and parses it as JSON, requiring `ready` **and** `self_destruct_timer_active` to both be exactly `True`.

Meanwhile, on the droplet, cloud-init: arms the self-destruct timer -> **asserts it is active** -> captures `nvidia-smi` -> builds a venv and `pip install vllm` (which drags in torch + CUDA, and is the bulk of the wall-clock) -> **asserts the timer is active again** -> writes the readiness marker.

The ordering is the safety property: **the timer is armed and verified before the slow, failure-prone work runs, and re-verified before the machine may declare itself ready.**

**Exit:** marker attests both flags -> `verified`. Any of the three waits times out -> `destroyed` (torn down; never used). A droplet that cannot arm its own self-destruct never reports ready, and so is never benchmarked on.

**When a run is mysteriously slow, it is here** — and it is almost certainly the vLLM install. `pull_artifacts` fetches the cloud-init log as `bootstrap.log` precisely because this evidence otherwise dies with the droplet.
