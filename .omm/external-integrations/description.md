Everything outside this codebase that it talks to — and the pattern it uses to talk to them.

**The seam pattern.** Each external dependency is isolated behind exactly one module, exposing intent-level functions, and **mocked in every test**. There are three:
- `do_client.py` — doctl (subprocess) + the DO REST API (`requests`)
- `remote.py` — `ssh` / `scp` (subprocess)
- `spaces.py` — DO Spaces (boto3, lazily imported, injectable client)

Nothing else in the codebase reaches for the network. The orchestration layer (`lifecycle.py`, `create.py`, `teardown.py`) composes these seams and is therefore fully testable — all 193 tests run offline, with no GPU and no DigitalOcean account. **If you add an external dependency, add it as a fourth seam in this shape.**

Clocks and sleeps are injectable too (`now=`, `sleep=`), so a test for a 600-second deadline runs instantly rather than actually waiting ten minutes.

**Two integrations are not reached from this machine at all.** The Hugging Face Hub is called *from the droplet* (the benchmark downloads `facebook/opt-125m` at run time), and the DO metadata service is called *by the droplet* to learn its own id so it can delete itself. They are drawn here because they are load-bearing — a droplet with no outbound network fails the required probe, and a droplet that cannot reach `169.254.169.254` cannot self-destruct.

The odd one out is `mlflow-store`: it is a "backend" with no network at all — the local `./mlruns` file store.
