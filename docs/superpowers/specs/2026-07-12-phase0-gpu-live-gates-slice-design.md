# Phase 0 — GPU Live Failure-Path Gates Slice (S4) (Design)

Date: 2026-07-12
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` — the "Binding
rule" and "Safety contract" at the top, the "Interactive debug path" /
short-TTL self-destruct procedure, the "Shared destroy-and-verify path", and the
"Recurring self-destruct semantics" and cloud-init sections). The GPU-automation
work was decomposed into S1 (safety spine), S2 (create path), S3a (cloud-init
renderer), S3b (remote seam + `gpu-up`), S3c-1 (scp seam + benchmark + local
drive/pull), and S3c-2 (Spaces upload + `gpu_run` lifecycle + `run` command). All of
those shipped as **library slices with every external seam mocked**: across 179
tests, no code has ever touched a real DigitalOcean droplet, a real SSH connection,
real boto3/Spaces, or a real GPU.

**S4 is the inverse of every prior slice.** It is the live validation slice. Its job
is to prove — against real droplets and real money — that the safety net the master
plan's binding rule demands actually works, and to make first contact with the
real GPU image. It produces **no production code and no new automated tests**. Its
deliverable is an executable runbook (this document) plus the recorded evidence of
running it.

The master plan is the spec of record. It already locks the safety contract ("no
matter which path runs — happy path, Ctrl-C, SIGKILL, terminal loss, timeout, manual
down, recurring self-destruct, duplicate destroy, or already-destroyed recovery —
success means the droplet does not exist and audit is clean"), the short-TTL
self-destruct test procedure, and the "destroy beats artifact preservation"
ordering. This design turns those locked rules into a concrete, ordered live
checklist and defines the pass/fail and contingency outcomes for each gate.

### Account-reality facts (verified via doctl 2026-07-11; pinned in `constants.py`)

- GPU region is **nyc2** (atl1 has no GPU capacity — the master plan's atl1 was wrong).
- RTX 4000 Ada single-GPU size slug is **gpu-4000adax1-20gb** (type `nvidia_rtx4000_ada`).
- Base image is **gpu-h100x1-base** — the only single-GPU generic "NVIDIA AI/ML Ready
  Image". The slug is **legacy-named after H100** and has **never been booted on RTX
  4000 Ada**. Gate 0 is where we find out whether it boots and whether the GPU stack
  (`nvidia-smi` / torch-CUDA / vLLM) works on real hardware. The S3c-1 probes have
  only ever run against fakes.

## Scope

In scope:
- This runbook document: preconditions, four ordered gates, safety discipline,
  contingencies, and a results log filled in during live execution.
- Live execution of the runbook with the user, recording evidence.
- Fixing only what live execution breaks, within the contingency rules below.
- Memory updates capturing the account-reality findings and S4 completion.

Out of scope:
- New production code, new automated tests, or refactors. (If a live gate exposes a
  bug, the fix is a follow-up — see Contingencies.)
- Debugging GPU images live. If `gpu-h100x1-base` does not work on RTX 4000 Ada, S4
  stops and spins off an image-selection slice (S5); it does not iterate on images.
- Any arbitrary dollar/spend ceiling. The safety net + audit discipline + always-
  available manual reap are the controls; there is no fixed cost cap.

## Design decisions

These were settled during brainstorming and are not reopened by the plan:

1. **Deliverable = runbook + live execution.** No production code. The tooling for
   all three failure paths already exists (`gpu-run`'s `finally`-teardown, `gpu-up
   --ttl-seconds` short-fuse with immediate id/name/PID print, the SIGINT-hardened
   `destroy_and_verify`, `make gpu-audit`). S4 proves it, it does not build it.

2. **Risk-first ordering: happy path first, then deliberate failures.** The biggest
   unknown (does the image boot? do the real probes pass?) is confronted first, in
   the run whose teardown is the *safest* (`finally` destroys on success). Only after
   the box is known to work do we deliberately break things.

3. **Four gates, three droplet lifetimes + a final sweep.** The master plan's three
   failure-path gates map to Gate 0 (happy path), Gate 1 (Ctrl-C mid-run), Gate 2
   (`kill -9` self-destruct). Gate 3 is a fresh-shell closing audit that proves
   nothing leaked across all three lifetimes.

4. **Audit is the spine.** `make gpu-audit` must be clean *before* and *after* every
   gate; a dirty audit is a hard stop. The slice passes only when a fresh shell (no
   in-memory state) audits clean at the very end.

5. **TTLs: realistic where safe, short only where the gate requires it.** Gate 0 uses
   the default TTL (7200s) because `finally` reaps it in minutes. Gate 2 inherently
   uses the short fuse (`--ttl-seconds 900`) — the short fuse *is* the test. There is
   no blanket short-TTL rule.

6. **Image failure → stop, record, spin off.** Decouples the safety proof (S4's real
   job, which must pass regardless of what boots) from "the GPU workload runs"
   (deferred to the image slice if the image is bad).

## Preconditions

Verify all of these before creating any droplet. Any failure is a hard stop.

1. **Env / credentials present.** `load_gpu_env` inputs — `DO_DROPLET_DESTROY_TOKEN`,
   `DO_SSH_KEY_IDS`, `DO_SSH_KEY_PATH` — and `load_spaces_env` inputs — Spaces
   access key, secret key, bucket — are all set. (`gpu-up` does not need Spaces, but
   Gate 0's `gpu-run` does.)
2. **Destroy token valid.** The token passes `probe_destroy_token` (create-path
   preflight already calls it, but confirm out-of-band so a stale token is a
   pre-spend stop, not a mid-run surprise).
3. **Constants confirmed.** Region `nyc2`, size `gpu-4000adax1-20gb`, image
   `gpu-h100x1-base`, Spaces region `nyc3` — as pinned in `constants.py`.
4. **Clean starting audit.** `make gpu-audit` prints clean (no matching droplet by
   `ml-lab` tag or `ml-lab-gpu-` name prefix; no related tagged volumes/snapshots/
   reserved-IPs/load-balancers).
5. **`doctl` authenticated.** A raw `doctl` call succeeds (so the first real call is
   not an auth failure).

## The four gates

Run in order. `make gpu-audit` must be clean before starting each gate and
re-confirmed clean after it (respecting Gate 2's TTL wait). Every run prints its
exact `make gpu-down DROPLET_ID=<id>` line — that is the always-safe manual abort if
anything hangs or misbehaves.

### Gate 0 — Happy path (`make gpu-run`)

The canonical command; full lifecycle at the realistic default TTL (7200s).

Command:

```text
make gpu-run
```

Watch, in order:
- droplet id / name / local PID printed immediately;
- public IPv4 acquired;
- SSH ready;
- `bootstrap-ready.json` verified (`ready` **and** `self_destruct_timer_active` both
  `true`);
- benchmark script scp'd up and run over SSH;
- artifacts pulled to `artifacts/<run-id>/` (6 files + `bootstrap.log`);
- Spaces upload URI (`s3://<bucket>/ml-pathway/phase0/<run-id>/...`) printed;
- `destroy_and_verify` runs in the `finally` block.

**PASS iff:** exit code 0; all 6 benchmark artifacts present locally; both **required**
probes (`torch_cuda`, `vllm_smoke`) report `ok: true`; a Spaces URI is returned; and
afterward the droplet is gone and `make gpu-audit` is clean. (`system` and
`nvidia_smi` probes are informational and do not gate the exit code, but their
output is recorded.)

**If the image does not boot, or boots but a required probe fails:** this is the
image contingency, not a safety failure. Stop after this gate. Record the evidence
(what failed, `nvidia-smi`/torch/vLLM output, `bootstrap.log`), confirm the droplet
was destroyed and audit is clean, update the account-reality memory, and open a
follow-up **S5 image-selection slice**. The safety net having destroyed the droplet
is itself a partial S4 pass and must be verified.

### Gate 1 — Ctrl-C mid-run (`make gpu-run`, interrupted)

Prove that interrupting the canonical command destroys cleanly, including the
operator-impatience double-interrupt path.

Procedure:
1. `make gpu-run` again.
2. Wait until it is past bootstrap verification and into the benchmark run.
3. Press **Ctrl-C** once.
4. Press **Ctrl-C a second time** while teardown is polling.

**PASS iff:** the second Ctrl-C does **not** abort teardown polling and does **not**
skip the post-destroy audit; the process exits nonzero; `destroy_and_verify`
completes; the droplet is gone and `make gpu-audit` is clean. This exercises the
`except BaseException` → shared-teardown path and the SIGINT-hardening from S1, plus
the "artifact-failure handler when a droplet still exists" branch.

### Gate 2 — `kill -9` self-destruct (`gpu-up --ttl-seconds 900`)

Prove the **remote** systemd self-destruct timer reaps the droplet with **zero** local
involvement. This is the only gate that validates the safety net independent of the
local process.

Procedure (from the master plan, followed exactly):
1. `uv run python scripts/do_gpu.py up --ttl-seconds 900` (short fuse; `--ttl-seconds`
   sets `enforce_budget=False`). Note: the `make gpu-up` target passes no TTL, so the
   short-fuse mode uses the direct script invocation.
2. Wait **only** until id / name / local PID print.
3. `kill -9 <pid>` (or close the terminal/window). Do **not** use Ctrl-C — Ctrl-C
   would run local cleanup and test the local watchdog instead of the remote timer.
4. Do **not** run `gpu-down`.
5. Do **not** wait for bootstrap in that process.
6. After the TTL should have lapsed (900s + up to one 300s self-destruct retry),
   open a **fresh shell**.
7. `make gpu-audit`.

**PASS iff:** the fresh-shell audit shows no matching lab GPU droplet and is clean.

**If the droplet still exists after TTL + one retry window:** this is a **real
self-destruct failure** — the most serious finding S4 can produce (the remote timer
or the in-droplet destroy-token path is broken). Manually `make gpu-down
DROPLET_ID=<id>`, confirm audit clean, and record it as a **blocking bug** with a
follow-up slice; S4 does not pass until the self-destruct path is fixed and re-run.

### Gate 3 — Fresh-shell final audit

Prove nothing leaked across all three droplet lifetimes.

In a **brand-new terminal** (no in-memory state from the prior gates):

```text
make gpu-audit
```

**PASS iff:** clean.

## Safety discipline

- **Audit brackets every gate.** Clean before, clean after. A dirty starting audit is
  a hard stop — never create on top of an existing droplet (the create path refuses
  anyway; we do not rely on that).
- **Manual reap is always available.** Every run prints `make gpu-down
  DROPLET_ID=<id>`. When in doubt — hang, weird output, wanting out — run it, then
  audit. This is the always-safe abort in place of a spend ceiling.
- **Gate 2 timing.** Its after-audit must wait out the full TTL (900s) plus at least
  one 300s self-destruct retry before concluding pass/fail.
- **One gate at a time.** Never have two lab droplets alive at once; the ordering and
  between-gate audits guarantee it.

## Contingencies

| Situation | Prescribed response |
|---|---|
| Image doesn't boot / required probe fails (Gate 0) | Stop; record evidence + `bootstrap.log`; verify auto-destroy happened + audit clean; update account-reality memory; open S5 image-selection slice. |
| Bootstrap never readies (SSH/bootstrap timeout, Gate 0/1) | Tool's own timeout should fire `destroy_and_verify`; verify it did and audit is clean. If not, manual `gpu-down` + audit; record as a bug. |
| Gate 1 second Ctrl-C aborts polling or skips audit | Safety-hardening regression — manual `gpu-down` + audit; record as a blocking bug against the SIGINT path. |
| Gate 2 droplet survives TTL + retry | Real self-destruct failure (most serious). Manual `gpu-down` + audit; blocking bug + follow-up slice; S4 does not pass. |
| Teardown itself fails (`TeardownError`) | Droplet may survive → manual `gpu-down` + audit; record the chained cause (`__context__`). |
| Any dirty audit at any point | Hard stop; manual `gpu-down` of whatever is found; do not proceed until clean. |

## Results log

Filled in during live execution and committed as the evidence S4 ran. One row per
gate:

| Gate | Command | Droplet id | Run id | Outcome (PASS/FAIL) | Duration | Notes / probe results |
|---|---|---|---|---|---|---|
| 0 — happy path | `make gpu-run` | | | | | |
| 1 — Ctrl-C mid-run | `make gpu-run` + 2× Ctrl-C | | | | | |
| 2 — kill -9 self-destruct | `do_gpu.py up --ttl-seconds 900` + `kill -9` | | | | | |
| 3 — fresh-shell audit | `make gpu-audit` | — | — | | — | |

Also record, in memory:
- **Account-reality finding:** does `gpu-h100x1-base` boot on `gpu-4000adax1-20gb`? Do
  `nvidia-smi` / torch-CUDA / vLLM work? (Confirms or refutes the pinned assumption.)
- **S4 completion:** which gates passed, any spun-off follow-up slices (S5 image, or
  a self-destruct bug), and the closing clean audit.

## Success criteria

S4 passes when **all** hold:
- Gate 1 and Gate 2 pass — the two deliberate failure paths both leave no droplet and
  a clean audit (the safety contract is proven live).
- Gate 3 (fresh-shell final audit) is clean.
- Gate 0 either fully passes, **or** its image contingency was executed cleanly (image
  finding recorded, S5 spun off) **and** its droplet was still safely destroyed. A
  bad image does not fail S4's safety proof; a surviving droplet does.
- Results log committed and memory updated.

This closes GPU automation and Phase 0.
```
