`src/ml_lab/gpu/` — the billing-safe DigitalOcean GPU automation, and the heart of this repo. Ten modules in three layers:

**Orchestration** — `lifecycle.py` (the spine: `gpu_up` for debug, `gpu_run` for the full benchmark), `create.py` (preflight gates + the multi-SKU capacity retry), `teardown.py` (the single function every destroy routes through), `audit.py` (is the account billing-clean?).

**Seams** (one external dependency each, mocked in every test) — `do_client.py` (doctl subprocess + the DO REST destroy-token probe), `remote.py` (hardened non-interactive ssh/scp), `spaces.py` (boto3 upload to DO Spaces).

**Support** — `cloud_init.py` (renders the self-destruct cloud-init template), `gpu_env.py` (loads and validates `.env`, failing loud with *all* missing vars at once), `constants.py` (the pinned slugs, tags, and timeout budget).

The safety contract, stated plainly: once a droplet exists, **every** exit path — success, exception, `KeyboardInterrupt`, benchmark failure — goes through `destroy_and_verify`. `gpu_run` puts it in a `finally`. `gpu_up` is the one exception, and only on a *fully verified* success: it deliberately leaves the droplet alive for debugging, having first proven the on-droplet self-destruct timer is armed.
