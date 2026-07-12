# GPU Live Failure-Path Gates (S4) Implementation Plan

> **For agentic workers:** This is a **live-execution runbook**, not a code slice. It has no failing-test → implement cycle because it produces no production code and no automated tests. Steps use checkbox (`- [ ]`) syntax for tracking. **Execution model: inline / interactive only** — it spends real money and requires the operator at the keyboard to press Ctrl-C and `kill -9`. Do NOT dispatch this to an unattended subagent.

**Goal:** Prove, against real DigitalOcean droplets, that the GPU safety net works on every failure path (happy-path auto-destroy, Ctrl-C mid-run, `kill -9` remote self-destruct) and make first contact with the real GPU image — closing Phase 0.

**Architecture:** Four ordered gates, each a separate real droplet, bracketed by `make gpu-audit`. Risk-first: happy path first (confirm the box works, safest teardown), then deliberate failures, then a fresh-shell closing audit. No new code; fixes to bugs surfaced live are follow-up slices per the spec's contingency table.

**Tech Stack:** `scripts/do_gpu.py` (Typer: `run` / `up --ttl-seconds` / `down --droplet-id` / `audit`), `make` targets, DigitalOcean (`nyc2`, `gpu-4000adax1-20gb`, `gpu-h100x1-base`), DO Spaces (`nyc3`).

**Reference spec:** `docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md` (spec of record: `.master_plan/ml_pathway_phase_00`).

---

## Files

- Modify: `docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md` — fill the Results log table during execution (this is the committed evidence).
- Update (memory): `~/.claude/projects/-Users-weldon-projects-ml-lab-template/memory/phase0-progress.md` and the account-reality memory — record the image finding and S4 completion.

No source files are created or modified by this plan. Any code change is out of scope and becomes a follow-up slice.

---

## Global rules (apply to every task)

- **Audit brackets every gate.** `make gpu-audit` must be clean *before* starting a gate and re-confirmed clean *after* it. A dirty audit is a hard stop.
- **One droplet alive at a time.** Never start a gate until the previous droplet is destroyed and audit is clean.
- **Manual reap is always available.** Every run prints `make gpu-down DROPLET_ID=<id>`. If anything hangs or looks wrong: run it, then audit. This is the abort in place of a spend ceiling.
- **Record as you go.** After each gate, write its row into the spec's Results log before starting the next gate.

---

### Task 0: Preconditions

**Files:** none (verification only).

- [ ] **Step 1: Confirm env / credentials are present**

Run:
```bash
uv run python -c "from ml_lab.gpu.gpu_env import load_gpu_env, load_spaces_env; load_gpu_env(); load_spaces_env(); print('env OK')"
```
Expected: prints `env OK`. If it raises `GpuEnvError`, it names every missing variable at once — set them in `.env` (`DO_DROPLET_DESTROY_TOKEN`, `DO_SSH_KEY_IDS`, `DO_SSH_KEY_PATH`, and the Spaces access key / secret key / bucket) and re-run. Hard stop until it prints `env OK`.

- [ ] **Step 2: Confirm `doctl` is authenticated**

Run:
```bash
doctl account get
```
Expected: prints account info (email, status). If it errors with an auth message, run `doctl auth init` yourself (type `! doctl auth init` in the prompt) and re-run. Hard stop until authenticated.

- [ ] **Step 3: Confirm the destroy token is valid (pre-spend)**

Run:
```bash
uv run python -c "from ml_lab.gpu.do_client import probe_destroy_token; from ml_lab.gpu.gpu_env import load_gpu_env; probe_destroy_token(load_gpu_env().destroy_token); print('token OK')"
```
Expected: prints `token OK`. If it raises `DOClientError`, the token is stale or underscoped — regenerate it in the DO console, update `.env`, re-run. Hard stop until `token OK`. (This turns a mid-run silent-billing failure into a pre-spend stop.)

- [ ] **Step 4: Confirm pinned constants**

Run:
```bash
uv run python -c "from ml_lab.gpu import constants as c; print(c.DO_REGION_SLUG, c.DO_SIZE_SLUG, c.DO_IMAGE_SLUG, c.SPACES_REGION)"
```
Expected: `nyc2 gpu-4000adax1-20gb gpu-h100x1-base nyc3`. If any differ, stop and reconcile against the spec's account-reality facts before spending.

- [ ] **Step 5: Confirm a clean starting audit**

Run:
```bash
make gpu-audit
```
Expected: clean report — no matching droplet (by `ml-lab` tag or `ml-lab-gpu-` name prefix), no related tagged volumes/snapshots/reserved-IPs/load-balancers; exit 0.
If NOT clean: destroy whatever is reported (`make gpu-down DROPLET_ID=<id>`), re-audit, and do not proceed until clean.

- [ ] **Step 6: Commit the pre-flight state (checkpoint)**

No file changes yet, so nothing to commit here — proceed. (The first commit is after Gate 0 records its result.)

---

### Task 1: Gate 0 — Happy path (`make gpu-run`)

**Files:** spec Results log (row: Gate 0).

Confirms the image boots on RTX 4000 Ada, the real GPU probes pass, artifacts pull, Spaces upload works, and `finally` auto-destroys. This is the riskiest unknown and the safest teardown.

- [ ] **Step 1: Confirm clean audit (gate precondition)**

Run: `make gpu-audit` → Expected: clean, exit 0. Hard stop if dirty.

- [ ] **Step 2: Start the happy-path run**

Run:
```bash
make gpu-run
```
Note the printed droplet id / name / local PID immediately (record them). Realistic default TTL (7200s); `finally` will reap it in minutes on success.

- [ ] **Step 3: Observe the lifecycle, in order**

Watch for, and note each milestone:
- public IPv4 acquired;
- SSH ready;
- `bootstrap-ready.json` verified (`ready` **and** `self_destruct_timer_active` both `true`);
- benchmark scp'd up and run over SSH;
- artifacts pulled to `artifacts/<run-id>/` (expect 6 benchmark files + `bootstrap.log`);
- Spaces URI printed (`s3://<bucket>/ml-pathway/phase0/<run-id>/...`);
- `destroy_and_verify` runs in `finally`.

- [ ] **Step 4: Evaluate PASS-iff**

Run:
```bash
ls artifacts/<run-id>/ && cat artifacts/<run-id>/torch_cuda.json artifacts/<run-id>/vllm_smoke.json
```
(substitute the real `<run-id>`).
**PASS iff:** `make gpu-run` exited 0; all 6 benchmark artifacts present; both required probes (`torch_cuda`, `vllm_smoke`) show `"ok": true`; a Spaces URI was printed; droplet gone afterward.

- [ ] **Step 5: Confirm post-gate audit is clean**

Run: `make gpu-audit` → Expected: clean, exit 0. If a droplet survives, `make gpu-down DROPLET_ID=<id>` then re-audit and record it as a teardown bug.

- [ ] **Step 6: If the image failed (contingency branch)**

If the image did not boot, or a required probe reported `"ok": false`:
- Do NOT debug images live. Capture evidence: `make gpu-run` output, `artifacts/<run-id>/nvidia-smi.txt`, `artifacts/<run-id>/benchmark.log`, `artifacts/<run-id>/bootstrap.log`.
- Verify the droplet was still destroyed and audit is clean (this part of the safety net must hold regardless).
- This is an image finding, not a safety failure. Proceed to Gate 1 (the safety gates must still be proven), and record that an **S5 image-selection slice** is required.

- [ ] **Step 7: Record Gate 0 in the Results log and commit**

Fill the Gate 0 row (droplet id, run id, PASS/FAIL, duration, notes incl. probe results / image finding) in the spec file, then:
```bash
git add docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md
git commit -m "docs(s4): record Gate 0 happy-path live result"
```

---

### Task 2: Gate 1 — Ctrl-C mid-run (`make gpu-run`, interrupted)

**Files:** spec Results log (row: Gate 1).

Proves interrupting the canonical command destroys cleanly, including the operator-impatience double-interrupt path (SIGINT hardening from S1).

- [ ] **Step 1: Confirm clean audit (gate precondition)**

Run: `make gpu-audit` → Expected: clean, exit 0. Hard stop if dirty.

- [ ] **Step 2: Start the run and let it reach the benchmark**

Run:
```bash
make gpu-run
```
Record the droplet id / name / PID. Wait until it is **past** bootstrap verification and **into** the benchmark run (you'll see the bootstrap-ready confirmation, then benchmark delivery/execution).

- [ ] **Step 3: Interrupt once**

Press **Ctrl-C** once. Teardown (`except BaseException` → `destroy_and_verify`) should begin polling.

- [ ] **Step 4: Interrupt a second time during teardown**

While teardown is polling, press **Ctrl-C** again.
**Expected:** the second Ctrl-C does NOT abort polling and does NOT skip the post-destroy audit. Teardown completes; process exits nonzero.

- [ ] **Step 5: Evaluate PASS-iff and confirm audit**

Run: `make gpu-audit` → Expected: clean, exit 0.
**PASS iff:** second Ctrl-C didn't abort teardown/audit; process exited nonzero; droplet gone; audit clean.
If the second Ctrl-C aborted teardown or left the droplet: `make gpu-down DROPLET_ID=<id>`, audit, and record a **blocking bug** against the SIGINT-hardening path (follow-up slice; S4 does not pass).

- [ ] **Step 6: Record Gate 1 in the Results log and commit**

Fill the Gate 1 row, then:
```bash
git add docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md
git commit -m "docs(s4): record Gate 1 Ctrl-C live result"
```

---

### Task 3: Gate 2 — `kill -9` self-destruct (`up --ttl-seconds 900`)

**Files:** spec Results log (row: Gate 2).

Proves the **remote** systemd self-destruct timer reaps the droplet with zero local involvement. This is the only gate that validates the safety net independent of the local process. Follow the master-plan procedure exactly.

- [ ] **Step 1: Confirm clean audit (gate precondition)**

Run: `make gpu-audit` → Expected: clean, exit 0. Hard stop if dirty.

- [ ] **Step 2: Start the short-fuse droplet**

Run:
```bash
uv run python scripts/do_gpu.py up --ttl-seconds 900
```
(The `make gpu-up` target passes no TTL, so use this direct invocation for the short fuse; `--ttl-seconds` sets `enforce_budget=False`.)

- [ ] **Step 3: Wait ONLY for id / name / PID, then hard-kill**

As soon as the droplet id / name / local PID print (record them), immediately:
```bash
kill -9 <pid>
```
(or close the terminal/window).
- Do **NOT** use Ctrl-C (it would run local cleanup and test the local watchdog, not the remote timer).
- Do **NOT** run `gpu-down`.
- Do **NOT** wait for bootstrap.

- [ ] **Step 4: Wait out the TTL + one retry window**

Wait at least **900s (TTL) + 300s (one self-destruct retry) ≈ 20 minutes** before concluding. The remote timer must delete the droplet on its own.

- [ ] **Step 5: Evaluate PASS-iff from a FRESH shell**

Open a **brand-new terminal** and run:
```bash
make gpu-audit
```
**PASS iff:** no matching lab GPU droplet remains; audit clean.
If the droplet still exists: this is a **real self-destruct failure** (most serious finding). `make gpu-down DROPLET_ID=<id>`, confirm audit clean, and record a **blocking bug** + follow-up slice. S4 does not pass until the self-destruct path is fixed and this gate re-run.

- [ ] **Step 6: Record Gate 2 in the Results log and commit**

Fill the Gate 2 row, then:
```bash
git add docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md
git commit -m "docs(s4): record Gate 2 kill-9 self-destruct live result"
```

---

### Task 4: Gate 3 — Fresh-shell final audit

**Files:** spec Results log (row: Gate 3).

Proves nothing leaked across all three droplet lifetimes.

- [ ] **Step 1: Audit from a brand-new terminal**

In a **fresh shell** (no in-memory state from the prior gates):
```bash
make gpu-audit
```
**PASS iff:** clean, exit 0.
If not clean: destroy what's reported, re-audit, and investigate which gate leaked (record it).

- [ ] **Step 2: Record Gate 3 in the Results log and commit**

Fill the Gate 3 row, then:
```bash
git add docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md
git commit -m "docs(s4): record Gate 3 fresh-shell final audit"
```

---

### Task 5: Record findings and close S4

**Files:** memory (`phase0-progress.md`, account-reality memory), spec Results log (final state).

- [ ] **Step 1: Write the account-reality finding to memory**

Update the account-reality memory with whether `gpu-h100x1-base` boots on `gpu-4000adax1-20gb` and whether `nvidia-smi` / torch-CUDA / vLLM work on the real GPU — confirming or refuting the pinned assumption.

- [ ] **Step 2: Update the phase-0 progress memory**

Record S4 outcome: which gates passed, any follow-up slices spun off (S5 image slice, or a self-destruct/SIGINT bug), and the closing clean audit. If all safety gates passed and Gate 3 is clean, note **Phase 0 complete**.

- [ ] **Step 3: Verify success criteria before declaring done**

Confirm all hold (per spec §Success criteria):
- Gate 1 and Gate 2 passed (both deliberate failure paths left no droplet + clean audit);
- Gate 3 clean;
- Gate 0 fully passed OR its image contingency executed cleanly (finding recorded, S5 spun off) AND its droplet was destroyed;
- Results log committed and memory updated.

- [ ] **Step 4: Final commit / push**

```bash
git add -A && git commit -m "docs(s4): close GPU live failure-path gates — Phase 0 complete" || echo "nothing to commit"
git push origin prod
```
(Only if success criteria hold. If a blocking bug was found, commit the evidence and leave S4 open pending the fix.)

---

## Self-review notes

- **Spec coverage:** Preconditions → Task 0; Gate 0/1/2/3 → Tasks 1–4; safety discipline (audit brackets, manual reap, one-at-a-time) → global rules + per-gate audit steps; contingencies → Step 6 (Gate 0), Step 5 (Gates 1 & 2), Task 4 Step 1; Results log + memory → Task 5; success criteria → Task 5 Step 3. All spec sections mapped.
- **No placeholders:** every command is concrete; `<run-id>` / `<pid>` / `<id>` are runtime values the operator substitutes from printed output, not plan gaps.
- **Command consistency:** `make gpu-run` / `make gpu-audit` / `make gpu-down DROPLET_ID=<id>` and `uv run python scripts/do_gpu.py up --ttl-seconds 900` verified against `Makefile` and `src/ml_lab/gpu/cli.py`.
