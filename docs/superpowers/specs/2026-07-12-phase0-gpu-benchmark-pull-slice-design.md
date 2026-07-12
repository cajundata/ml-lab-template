# Phase 0 — GPU Benchmark + Local Drive/Pull Slice (S3c-1) (Design)

Date: 2026-07-12
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` §3, "Benchmark
execution", "vLLM smoke check", "Benchmark artifact bundle", and the `gpu-run`
lifecycle steps 10–13). The GPU-automation work is decomposed into four slices; S1
(safety spine), S2 (create path), S3a (cloud-init renderer), and S3b (remote seam +
`gpu-up`) shipped. S3c (benchmark + artifacts + Spaces + `gpu-run`) is large, so —
per the one-plan-per-slice preference — it is cut into two sub-slices, and this
design covers the first:

- **S3c-1 — scp seam + benchmark script + local drive/pull (this slice):** the scp
  primitives deferred from S3b, the self-contained `scripts/gpu_benchmark.py` that
  runs on the droplet, and the local driver that delivers the script, runs it over
  SSH, and pulls the artifact bundle to disk. A library slice: nothing wires these
  into a full lifecycle yet, and no test touches a real droplet, SSH connection, or
  GPU.
- **S3c-2 — Spaces upload + `gpu_run` lifecycle + `run` command:** boto3 DO Spaces
  upload, the full create → benchmark → pull → upload → always-destroy `gpu_run`
  lifecycle, and the `run` Typer command. S3c-2 is the first caller of this slice's
  driver functions.

The master plan is the spec of record and locks the benchmark scope (boot, GPU
visibility, `nvidia-smi`, torch-sees-CUDA, a small CUDA op, vLLM load + generate),
the artifact bundle contents, the scp-after-SSH-readiness delivery, and the
local-drives / droplet-self-destructs ownership split. This design pins the module
layout, the benchmark script's testable-core structure, the exit-code contract, and
the test seams; it does not reopen locked decisions.

## Scope

In scope:
- `src/ml_lab/gpu/remote.py` (extend) — `_run_scp` plus `scp_up` / `scp_down`
  primitives on the existing SSH seam.
- `scripts/gpu_benchmark.py` (new) — the self-contained on-droplet benchmark script:
  a testable orchestration core plus lazy-importing GPU probes.
- `src/ml_lab/gpu/benchmark.py` (new) — the local driver: `deliver_and_run_benchmark`
  and `pull_artifacts`.
- `tests/test_gpu_remote.py` (extend), `tests/test_gpu_benchmark.py` (new),
  `tests/test_gpu_benchmark_driver.py` (new) — all offline, every seam mocked.

Out of scope (later slices / gates):
- boto3 DO Spaces upload, the `gpu_run` lifecycle, and the `run` command — S3c-2.
- Any live DigitalOcean call, live SSH, live scp, or actual `torch` / `vllm`
  execution — S4 (the first live run also confirms the image boots on RTX 4000 Ada).
- No new `make` target. `Makefile`'s `gpu-run` already points at `do_gpu.py run`; it
  stays non-functional until S3c-2 wires `gpu_run`.

## Module layout (decided)

The scp *primitives* live in `remote.py` beside `_run_ssh` — same seam, same
`SSH_OPTS`, same subprocess-mock test pattern. The *local orchestration* (run-id →
artifact paths, deliver-run-pull) lives in a **new** `ml_lab/gpu/benchmark.py`, so
`remote.py` stays a pure SSH/scp seam and does not accrue benchmark-specific logic.
This mirrors the existing `create.py` / `remote.py` / `lifecycle.py` separation. The
S3b reviewer's suggested `_deadline_poll` helper is **not** introduced here: scp and
the benchmark run are one-shot subprocess calls with a hard timeout, not poll loops,
so no fourth deadline-shaped function appears.

## `remote.py` — scp primitives

```python
def _run_scp(argv, *, key_path, timeout):
    """Run `scp -i <key> <SSH_OPTS...> <argv...>`; return the CompletedProcess."""

def scp_up(host, local_path, remote_path, *, key_path, timeout): ...
def scp_down(host, remote_path, local_path, *, key_path, timeout, recursive=False): ...
```

- `_run_scp` mirrors `_run_ssh`: `["scp", "-i", key_path, *SSH_OPTS, *argv]`,
  `capture_output=True`, `text=True`, bounded by `timeout`. Reusing `SSH_OPTS` keeps
  the same hardening (`BatchMode`, `IdentitiesOnly`, `accept-new`,
  `UserKnownHostsFile=/dev/null`).
- `scp_up` sends `local_path` → `root@<host>:<remote_path>`. `scp_down` pulls
  `root@<host>:<remote_path>` → `local_path`; `recursive=True` adds `-r` for the
  bundle directory.
- **One-shot semantics (distinct from the poll-waiters):** `wait_for_ssh` /
  `wait_for_bootstrap` tolerate transient failure until a deadline; scp is a single
  required transfer, so a nonzero exit or `subprocess.TimeoutExpired` raises
  `RemoteError` immediately. No new exception type.

## `scripts/gpu_benchmark.py` — self-contained on-droplet script

The droplet venv has `vllm` (and transitively `torch`) but **not** `ml_lab`, so this
is a standalone single file delivered by scp and run with the droplet's
`venv/bin/python`. It imports nothing from `ml_lab`.

Structure:

1. **Module top: stdlib only** — `json`, `sys`, `subprocess`, `platform`, `argparse`,
   `traceback`, `time`, `pathlib`. No `torch` / `vllm` at import time, so the module
   is importable in CI (that importability is the point of the design).
2. **Probe functions**, each lazy-importing its GPU dependency *inside* the function
   body, each returning a JSON-serializable result (or raw text):

   | Probe | Returns | Gating |
   | --- | --- | --- |
   | `probe_torch_cuda()` | `{cuda_available, matmul_ok, elapsed_s}` (torch from vLLM; small matrix multiply) | **required** |
   | `probe_vllm_smoke(model_id)` | `{model_id, load_ok, generate_ok, token_count, elapsed_s, error}` (import → load pinned model → generate a few tokens from a fixed prompt) | **required** |
   | `probe_system()` | OS, Python, GPU name/memory, driver, CUDA visibility | informational |
   | `probe_nvidia_smi()` | raw `nvidia-smi` text | best-effort |

3. **Testable core:**

   ```python
   def run_benchmark(run_id, artifacts_root, probes=DEFAULT_PROBES) -> int:
   ```

   Creates `artifacts_root/<run-id>/`, then for each probe runs it wrapped in
   try/except: on success writes its artifact; on exception records
   `{"ok": false, "error": <traceback text>}` (or the failure captured inside the
   probe's own result dict) and **still writes every remaining artifact** — a partial
   bundle is always produced. Writes the six bundle files (`benchmark.json`,
   `system.json`, `torch_cuda.json`, `vllm_smoke.json`, `nvidia-smi.txt`,
   `benchmark.log`); `benchmark.json` aggregates which probes ran and their `ok`
   flags. Returns `0` iff **both required probes succeeded**, else `1`. `probes` is an
   injectable name→callable mapping so tests supply fakes (including failing ones)
   with no GPU present.
4. `main(argv)` — argparse `--run-id`, `--smoke-model-id` (default
   `facebook/opt-125m`), `--artifacts-root` (default `/opt/ml-lab/artifacts`); calls
   `sys.exit(run_benchmark(...))`.

The script defaults `--smoke-model-id` to the pinned `facebook/opt-125m` so it is
runnable standalone, but the local driver passes the value explicitly (below) to keep
a single source of truth.

## `ml_lab/gpu/benchmark.py` — local driver

Module-anchored paths (the `parents[3]` trick from `cloud_init.py`, so the driver
works regardless of CWD):

```python
BENCHMARK_SCRIPT_PATH  = <repo>/scripts/gpu_benchmark.py   # module-anchored
LOCAL_ARTIFACTS_ROOT   = <repo>/artifacts                  # module-anchored
REMOTE_BENCHMARK_PATH  = "/opt/ml-lab/gpu_benchmark.py"
REMOTE_ARTIFACTS_ROOT  = "/opt/ml-lab/artifacts"
REMOTE_VENV_PYTHON     = "/opt/ml-lab/venv/bin/python"
CLOUD_INIT_LOG         = "/var/log/cloud-init-output.log"

def deliver_and_run_benchmark(
    host, run_id, *, key_path,
    smoke_model_id=SMOKE_MODEL_ID,
    timeout=BENCHMARK_TIMEOUT_SECONDS,
) -> int: ...

def pull_artifacts(
    host, run_id, *, key_path,
    dest_root=LOCAL_ARTIFACTS_ROOT,
    timeout=SSH_TIMEOUT_SECONDS,
) -> Path: ...
```

- `deliver_and_run_benchmark` — `scp_up` the script to `REMOTE_BENCHMARK_PATH`, then
  run `REMOTE_VENV_PYTHON REMOTE_BENCHMARK_PATH --run-id <id> --smoke-model-id <m>`
  over SSH bounded by `BENCHMARK_TIMEOUT_SECONDS`. **Returns the benchmark process
  exit code** (see the error-handling contract below).
- `pull_artifacts` — `scp_down(recursive=True)` the remote `<run-id>/` bundle to
  `dest_root/<run-id>/`, then `scp_down` `CLOUD_INIT_LOG` to
  `dest_root/<run-id>/bootstrap.log`. Returns the local dest `Path`. Raises
  `RemoteError` on any scp failure.

## Data flow

```
local:  scripts/gpu_benchmark.py  --scp_up-->  droplet:/opt/ml-lab/gpu_benchmark.py
local:  --ssh--> venv/bin/python gpu_benchmark.py --run-id R --smoke-model-id M
                 (timeout = BENCHMARK_TIMEOUT_SECONDS = 1800s)  → exit code
drop :  writes /opt/ml-lab/artifacts/R/{benchmark,system,torch_cuda,vllm_smoke}.json,
                                        nvidia-smi.txt, benchmark.log
local:  --scp_down -r--> artifacts/R/
local:  --scp_down----> /var/log/cloud-init-output.log → artifacts/R/bootstrap.log
```

In S3c-2 the `gpu_run` lifecycle calls `deliver_and_run_benchmark` then
`pull_artifacts` between the S3b `wait_for_bootstrap` and the always-run `finally`
teardown, and uploads the pulled bundle to Spaces.

## Error handling — the key distinction

Two failure classes are deliberately represented differently, so S3c-2's `gpu_run`
can preserve partial artifacts without confusing a hang with a completed-but-failed
benchmark:

- **Transport failure or benchmark timeout** — scp nonzero exit, or the SSH run
  raising `subprocess.TimeoutExpired` — raises `RemoteError`. In S3c-2 this reaches
  `gpu_run`'s `finally` → `destroy_and_verify`, with **no pull**. Destroy beats a
  partial pull after a hang, matching the master plan's "benchmark timeout handler
  destroys."
- **Benchmark ran to completion but a required probe failed** —
  `deliver_and_run_benchmark` *returns* a nonzero exit code (it does **not** raise).
  S3c-2 can then still `pull_artifacts` the partial bundle, upload it, and surface the
  nonzero result *after* teardown. This is what makes "partial artifacts first"
  actually reach local disk.

Within `gpu_benchmark.py`, each probe is individually defensive: a probe exception is
captured (into its result dict or an `{"ok": false, "error": …}` record) and the
remaining artifacts are still written, so the bundle is always complete-enough to
diagnose. Only the two **required** probes (`torch_cuda`, `vllm_smoke`) gate the exit
code; `system` and `nvidia-smi` are informational and never fail the run (`nvidia-smi`
is already `|| true` in cloud-init).

## Safety invariants

- No S3c-1 code path reaches the network, a droplet, an SSH connection, or a GPU;
  every seam (`remote._run_ssh`, `remote._run_scp`, subprocess) is mocked in tests.
- No new teardown path: S3c-1 adds no destroy logic. `destroy_and_verify` remains the
  single teardown chokepoint, called only by S3c-2's `gpu_run` (and existing callers).
- The scp seam reuses hardened `SSH_OPTS`; no key material or destroy token appears in
  any scp/ssh argv this slice constructs (the driver passes only `--run-id` and
  `--smoke-model-id`).
- `gpu_benchmark.py` imports nothing from `ml_lab` and no GPU library at module top,
  so it stays a self-contained scp'able file and remains importable in CI.

## Testing strategy

All offline; `remote` seam and subprocess mocked; no test reaches the network, a
droplet, SSH, scp, or a GPU.

`tests/test_gpu_remote.py` (extend):
- `scp_up` / `scp_down` build the correct argv — `-i key`, `SSH_OPTS`, correct
  `root@host:path` orientation, `-r` present only when `recursive=True`.
- A nonzero scp exit → `RemoteError`; `subprocess.TimeoutExpired` → `RemoteError`.

`tests/test_gpu_benchmark.py` (new):
- The module imports with **no `torch` / `vllm` installed** (the core is import-safe).
- All-pass injected probes → exit `0`, all six artifacts written under
  `tmp_path/<run-id>/`, `benchmark.json` aggregate reflects all `ok`.
- A **required** probe that raises → its artifact records the error text, the other
  artifacts are still written, exit `1`.
- A failing **optional** probe (`nvidia-smi`) → still exit `0`.
- `run_benchmark` creates `artifacts_root/<run-id>/` when absent.

`tests/test_gpu_benchmark_driver.py` (new):
- `deliver_and_run_benchmark` `scp_up`s the script to `REMOTE_BENCHMARK_PATH` and runs
  the correct remote command (`REMOTE_VENV_PYTHON … --run-id … --smoke-model-id …`)
  with `timeout=BENCHMARK_TIMEOUT_SECONDS`; a nonzero benchmark exit is **returned**,
  not raised; an SSH `TimeoutExpired` → `RemoteError`.
- `pull_artifacts` does `scp_down(recursive=True)` for the bundle and a second
  `scp_down` mapping the cloud-init log to `bootstrap.log`, returns the local dest
  `Path`, and raises `RemoteError` on an scp failure.

## Non-goals / deferred

- No Spaces/boto3 upload, no `gpu_run` lifecycle, no `run` command — S3c-2.
- No new `make` target — S3c-2 activates `make gpu-run`.
- No live DigitalOcean call, live SSH/scp, actual `torch`/`vllm` execution, or
  failure-path-gate run — S4.
- No `_deadline_poll` refactor — scp and the benchmark run are one-shot, not polls, so
  nothing triggers it.
