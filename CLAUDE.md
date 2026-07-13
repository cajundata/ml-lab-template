# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo. Concise on purpose —
the specs under `docs/superpowers/` and the master plan are the fuller record.

## What this is

`ml-lab-template` — a reusable ML lab skeleton for the Cajun Data ML Pathway. A
deterministic train/evaluate lifecycle with local MLflow tracking, plus **billing-safe
DigitalOcean GPU automation** (one command up, always self-destructs). Python for model
development; Go is the durable service layer in later phases. See `README.md` for usage.

## Working agreement (important)

- **Use `superpowers:*` skills, NOT `gsd-*` skills** in this repo. Always.
- Workflow for any non-trivial change: **brainstorm → spec (`docs/superpowers/specs/`)
  → plan (`docs/superpowers/plans/`) → subagent-driven execution**, TDD throughout.
- **Small, incremental slices — one plan per slice.** The GPU work was cut into
  S1…S4 (and sub-slices) exactly this way.
- Work happens on the **`prod`** branch (not `main`); commit + push there.
- Co-author trailer on commits: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Where things live

- **`.master_plan/`** — the governing document (pathway direction + `ml_pathway_phase_00`
  detail). Cited by every spec as the "spec of record". Now tracked in git.
- **`docs/superpowers/specs/`, `docs/superpowers/plans/`** — per-slice design + runbooks.
  The S4 spec (`2026-07-12-phase0-gpu-live-gates-slice-design.md`) has the live Results
  log + the 10 live findings.
- **`src/ml_lab/`** — `data/`, `models/`, `evaluation/`, `tracking/`, `gpu/`, `cli.py`.
- **`.omm/`** — architecture docs as Mermaid diagrams + prose, 5 perspectives recursed to
  leaf elements. **Read this to orient fast.** `state-transitions` is the one that matters
  most (droplet lifecycle as a state machine; the `stranded` billing state and the three
  defense layers out of it). Browse with `omm view`; regenerate with the `omm-scan` skill
  after structural changes. Generated FROM the code — the code wins in any disagreement.
- **Project memory** lives OUTSIDE the repo at
  `~/.claude/projects/-Users-weldon-projects-ml-lab-template/memory/` — see "Portability".

## Status

**Phase 0 is COMPLETE (2026-07-13).** Local train/evaluate lifecycle + GPU automation
both proven end-to-end live. `make test` = **193 tests**. Next is Phase 1 (tabular).

## GPU automation — hard-won facts (from S4 live)

Every GPU seam was mocked through S3; live execution (S4) surfaced these. Don't relearn
them the hard way:

- **`down` always means destroy. Never power off a lab droplet — a stopped GPU droplet
  still bills.** `make gpu-audit` reports anything billable; `make gpu-down DROPLET_ID=<id>`
  is the always-safe manual reap.
- **Single-GPU capacity flaps across BOTH regions and SKUs by the minute.** `constants.py`
  pins an interchangeable list `ACCEPTABLE_SIZE_SLUGS = [gpu-h100x1-80gb, gpu-h200x1-141gb,
  gpu-l40sx1-48gb]` (Hopper first); create resolves the live region per SKU and lands
  whichever `(SKU, region)` has capacity. `DO_REGION` is only a preferred hint.
- **The `gpu-h100x1-base` image boots cleanly on Hopper (H100/H200)**: `nvidia-smi`, torch+CUDA,
  and a `transformers` `opt-125m` generate all run on-GPU.
- **vLLM does not yet serve** (engine-core init fails on this stack). `vllm_smoke` is an
  *informational* probe; the required GPU-workload proof is `transformers_smoke`. vLLM
  belongs to the Phase-5 cloud-serving track — capture the engine *subprocess* stderr first.
- **doctl `-o json` writes errors (incl. 404) to STDOUT, not stderr.** All error handling
  reads both. (This caused a false `TeardownError` that would have broken every run.)
- **The self-destruct safety net is proven live**: Ctrl-C mid-run tears down cleanly
  (2nd Ctrl-C ignored), and a `kill -9`'d run is reaped by the remote systemd timer with
  zero local involvement.

## New-machine setup (GPU work)

`git clone` gives you all code + specs + `.master_plan/`. Then, for GPU work you must
reestablish machine-local state (`README.md` → "GPU prerequisites" has the full checklist):

1. `uv sync --dev`
2. `doctl auth init` (the DO CLI auth is machine-local)
3. Create/copy an SSH key, **register it with DO**, and in `.env` set `DO_SSH_KEY_IDS`
   to its **numeric ID** (`doctl compute ssh-key list`) — NOT its name — and
   `DO_SSH_KEY_PATH` to the **private** key path (NOT the `.pub`).
4. `cp .env.example .env` and fill the DO destroy token + Spaces keys.
5. Sanity: `make gpu-audit` should print clean.

## Portability (switching machines)

The **live** project memory is at `~/.claude/projects/<repo-path-with-slashes-as-dashes>/memory/`
and does NOT travel with a `git clone`. Two mechanisms keep you covered:

- This `CLAUDE.md` + the committed specs + `.master_plan/` are the durable record — enough
  to rebuild full context on a new machine.
- A snapshot of the live memory is committed under `docs/claude-memory/`. Refresh it before
  switching machines with `scripts/sync-claude-memory.sh`; re-seed it on the new machine
  (after cloning) with `scripts/restore-claude-memory.sh`.
