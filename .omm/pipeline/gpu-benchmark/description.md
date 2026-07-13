The hardware proof — `scripts/gpu_benchmark.py`, run **on the droplet**. Not an ML pipeline: it answers "does this machine actually have a working GPU that can run a real model?", and its output is evidence, not a model.

**Self-contained by design.** Top-level imports are stdlib only, and it never imports `ml_lab`. torch / transformers / vllm are imported *lazily inside each probe*. That is what lets it be unit-tested in CI on a machine with no GPU while only ever *executing* remotely (scp'd up, run against `/opt/ml-lab/venv/bin/python`).

**Every probe runs defensively.** A probe that raises has its traceback written into its own artifact and is marked failed — and the remaining probes still run. **A partial bundle is always written.** The exit code is 0 **iff both required probes pass**, and that code propagates all the way out to `make gpu-run`'s exit status.

Probes are injected as a `name -> callable` mapping, which is what makes the whole defensive-write path testable without a GPU.
