`scripts/cloud-init-gpu.yaml.tmpl` — the droplet's entire boot sequence, rendered by `cloud_init.py` and handed to DO as `user_data` at create time. There is no configuration management, no Ansible, no second SSH-in-and-provision step: the machine either brings itself up correctly or it never reports ready.

**`write_files`** plants four things: `/etc/ml-lab/run.env` (**mode 0600**, carrying the destroy-scoped DO token, the run id, and the TTL), the `self_destruct.sh` script (mode 0700), and the systemd `.service` + `.timer` units.

**`runcmd`**, in order:
1. `daemon-reload`, then `enable --now` the self-destruct timer.
2. **`systemctl is-active --quiet` on the timer** — the boot *fails here* if it did not arm.
3. `mkdir` the artifacts dir; capture `nvidia-smi`.
4. `apt-get install python3-venv python3-pip ffmpeg`; build `/opt/ml-lab/venv`; `pip install vllm` (which drags in torch + CUDA — the slowest step by far).
5. **Assert the timer is active again.**
6. Write `/opt/ml-lab/bootstrap-ready.json`.

The ordering is the safety property: **the timer is armed and verified before anything slow or failure-prone happens**, and re-verified before the machine is allowed to declare itself ready. The expensive, flaky work is bracketed by the thing that guarantees the machine cannot outlive its TTL.
