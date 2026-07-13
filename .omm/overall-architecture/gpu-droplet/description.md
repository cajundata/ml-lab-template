The remote half — everything that exists *on* the DigitalOcean GPU droplet. It is not a service we run; it is a machine that boots, proves itself, does one job, and deletes itself.

Provisioned from the `gpu-h100x1-base` image (NVIDIA AI/ML Ready) with `scripts/cloud-init-gpu.yaml.tmpl` as `user_data`. Boot does five things, in order:

1. Writes `/etc/ml-lab/run.env` at **mode 0600**, carrying the destroy-scoped DO token.
2. Installs and starts a **systemd self-destruct timer** (`OnBootSec=TTL`, retrying every `SELF_DESTRUCT_RETRY_SECONDS`) — then *asserts it is active* before continuing.
3. Captures `nvidia-smi` output.
4. Builds a venv at `/opt/ml-lab/venv` and `pip install vllm` (which pulls torch + CUDA).
5. Re-asserts the timer is active, then writes `/opt/ml-lab/bootstrap-ready.json`.

That marker file is the handshake: the local side polls it and requires **both** `ready` and `self_destruct_timer_active` to be exactly `true`, parsed as JSON rather than string-matched. A droplet that cannot arm its own self-destruct never reports ready — so the local side tears it down instead of using it.

`scripts/gpu_benchmark.py` is then scp'd up and run against the venv's Python. It is self-contained on purpose (stdlib-only top-level imports, no `ml_lab` import) so it can be reasoned about and unit-tested locally while only ever *executing* remotely.
