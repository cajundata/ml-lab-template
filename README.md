# ml-lab-template

A reusable ML lab skeleton for the Cajun Data ML Pathway. It gives you two things:

1. A **deterministic train/evaluate lifecycle** with local MLflow tracking — seeded data,
   a smoke model, an evaluation pass, all logged to one MLflow run.
2. **Billing-safe DigitalOcean GPU automation** — one command up, one command down, and
   `down` *always* means destroy. A run that crashes, is Ctrl-C'd, or is `kill -9`'d still
   ends with the droplet gone.

Python drives model development; Go is planned as the durable service layer in later phases.

**Status:** Phase 0 is complete and proven end-to-end live (local lifecycle + GPU automation).
`make test` runs **193 tests**, all passing, with every external seam mocked — the full suite
needs no cloud credentials and costs nothing.

---

## Quickstart (local — no cloud account needed)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone <your-fork-or-this-repo> ml-lab-template
cd ml-lab-template
make setup          # uv sync --dev
make test           # 193 tests, no credentials required
```

Then run the lifecycle:

```bash
make train          # generate seeded data, train the smoke model, open an MLflow run
make evaluate       # score the saved model on the saved test split, append metrics to that same run
make mlflow-ui      # browse runs at http://127.0.0.1:5000
```

`train` and `evaluate` log to the **same** MLflow run: `train` writes the run id to
`reports/smoke/latest/train_run_id.txt`, and `evaluate` reads it back and appends its metrics.
Run `train` before `evaluate` — evaluating without it fails loudly rather than inventing a run.

Everything is seeded (`SMOKE_SEED` in `src/ml_lab/config.py`), so two clean runs on two machines
produce identical data, an identical model, and identical metrics.

`make clean` removes generated data, models, and reports (it leaves `mlruns/` alone).

---

## Repo layout

```
src/ml_lab/
  config.py          paths + seed + experiment name — the knobs for the local lifecycle
  data/              deterministic synthetic data generation
  models/            the logistic smoke model (train + save/load)
  evaluation/        scoring against the saved test split
  tracking/          MLflow run helpers
  serving/           (placeholder — later phases)
  cli.py             `ml-lab` Typer app: train-smoke, evaluate-smoke
  gpu/               the GPU automation package (see below)
scripts/
  do_gpu.py          entrypoint behind the `make gpu-*` targets
  gpu_benchmark.py   the benchmark that runs ON the droplet
  cloud-init-gpu.yaml.tmpl   droplet bootstrap + the remote self-destruct timer
tests/               193 tests; every cloud seam mocked
data/ models/ reports/ artifacts/   generated output (gitignored)
mlruns/              local MLflow store (gitignored)
```

The GPU package is split one-concern-per-module: `do_client` (doctl subprocess seam),
`remote` (SSH/SCP seam), `spaces` (boto3 seam), `create`, `teardown`, `audit`, `lifecycle`,
`cloud_init`, `benchmark`, `gpu_env`, `constants`.

## Make targets

| Target | What it does |
|---|---|
| `make setup` | `uv sync --dev` |
| `make test` | full suite (no credentials, no cost) |
| `make train` | generate data + train + log to MLflow |
| `make evaluate` | score the saved model, append metrics to the training run |
| `make mlflow-ui` | MLflow UI on `127.0.0.1:5000` |
| `make clean` | delete generated data/models/reports |
| `make gpu-audit` | report anything billable on the DO account |
| `make gpu-run` | full GPU lifecycle — **always destroys** |
| `make gpu-up` | create a droplet and **leave it alive** (debug path — you must reap it) |
| `make gpu-down DROPLET_ID=<id>` | destroy a droplet and verify it's gone |

---

## GPU automation

### The safety model — read this first

GPU droplets are disposable and expensive. This repo is built around one rule:

> **`down` always means destroy. There is no stop or power-off workflow.**
> **A stopped GPU droplet still bills. Never power off a lab droplet — destroy it.**

Three independent layers keep you from paying for a forgotten GPU:

1. **`gpu-run` destroys in a `finally`.** Success, benchmark failure, SSH timeout, exception,
   Ctrl-C — the droplet is destroyed and its absence verified. Destroy beats artifact
   preservation; a second Ctrl-C during teardown is ignored.
2. **A remote self-destruct timer**, installed by cloud-init at boot and verified active before
   the droplet is handed to you. If your laptop dies or the process is `kill -9`'d, the droplet
   deletes *itself* with zero local involvement (TTL default 2h, with a retry).
3. **`make gpu-audit`** — the manual backstop. It lists every billable lab resource (droplets,
   volumes, snapshots, reserved IPs, load balancers) and exits nonzero if the account is dirty.

Layers 1 and 2 are both proven live, not just in tests. If you are ever unsure, run `make gpu-audit`.

On top of those, `create` refuses to launch a second lab droplet while one is already alive — so a
mistaken double `gpu-run` cannot quietly double your bill.

### Prerequisites (once per machine)

A `git clone` brings the code, specs, and plans — it does **not** bring any of the machine-local
state below. Local train/evaluate needs none of it; GPU work needs all of it.

1. **Install and authenticate `doctl`** (the DO CLI auth lives outside `.env`):

   ```bash
   brew install doctl        # or see DigitalOcean's install docs
   doctl auth init           # paste a DO API token
   doctl account get         # verify
   ```

2. **Register an SSH key with DigitalOcean** — the droplet is created with your public key so
   the tooling can connect. Add one in the DO control panel (or `doctl compute ssh-key import`),
   then get its **numeric ID**:

   ```bash
   doctl compute ssh-key list    # note the ID column, e.g. 57744215
   ```

3. **Create a DO Spaces bucket** in the **`nyc3`** region (pinned in `gpu/constants.py`) and
   generate a Spaces access key pair. Only `gpu-run` uploads artifacts; `gpu-up` does not, and
   won't fail on missing Spaces credentials.

4. **Fill in `.env`:**

   ```bash
   cp .env.example .env
   ```

   | Variable | Required for | Notes |
   |---|---|---|
   | `DO_DROPLET_DESTROY_TOKEN` | all GPU commands | DO API token with droplet **delete** scope. Powers both local teardown and the in-droplet self-destruct — it is the only secret placed on the droplet. |
   | `DO_SSH_KEY_IDS` | all GPU commands | Comma-separated **numeric IDs or fingerprints** from step 2. **Not** the key *name* — doctl rejects names. |
   | `DO_SSH_KEY_PATH` | all GPU commands | Path to the matching **private** key, e.g. `~/.ssh/id_ed25519`. **Not** the `.pub`. |
   | `SPACES_ACCESS_KEY_ID` | `gpu-run` | Spaces key pair from step 3. |
   | `SPACES_SECRET_ACCESS_KEY` | `gpu-run` | |
   | `SPACES_BUCKET` | `gpu-run` | Bucket name (must live in `nyc3`). |

   Missing variables fail fast and name *all* of them at once, before any droplet is created.
   `.env.example` also carries two slots marked **reserved** (`MLFLOW_TRACKING_URI`,
   `DO_API_TOKEN_LOCAL`) that no code reads today — leave them blank.

5. **Sanity check** — this should print a clean report before you create anything:

   ```bash
   make gpu-audit
   ```

### Running a GPU job

```bash
make gpu-run
```

Creates a droplet, waits for SSH and for cloud-init bootstrap to finish, verifies the
self-destruct timer is armed, runs the benchmark on the GPU, pulls the artifact bundle back,
uploads it to Spaces — and destroys the droplet, always.

The bundle lands locally in `artifacts/<run-id>/` and durably at
`s3://<your-bucket>/ml-pathway/phase0/<run-id>/`. It contains `benchmark.json`, `system.json`,
`nvidia-smi.txt`, `torch_cuda.json`, `vllm_smoke.json`, `benchmark.log`, and `bootstrap.log`
(the droplet's cloud-init output, pulled *before* teardown so you can debug a dead droplet).

**To keep a droplet alive for interactive debugging:**

```bash
make gpu-up     # prints id, name, region, size, public IP, TTL, and the exact down command
```

`gpu-up` verifies bootstrap and hands the droplet to you **alive**. The self-destruct timer is
your only safety net at that point — reap it yourself as soon as you're done:

```bash
make gpu-down DROPLET_ID=<id>
```

(If bootstrap or SSH *fails*, `gpu-up` destroys the droplet before raising. Only a fully verified
droplet is left running.)

### Capacity, SKUs, and regions

Single-GPU capacity on DigitalOcean flaps across **both** regions and SKUs by the minute, so the
lab does not pin one machine. `gpu/constants.py` holds an interchangeable list —
`ACCEPTABLE_SIZE_SLUGS = ["gpu-h100x1-80gb", "gpu-h200x1-141gb", "gpu-l40sx1-48gb"]` (Hopper
first) — and create resolves the live region per SKU and lands whichever `(SKU, region)` pair has
capacity. `DO_REGION` is a *preferred hint*, not a constraint. If nothing has capacity, create
polls for a bounded window and then gives up cleanly without leaving anything billable behind.

### Known limits

- **vLLM does not currently serve** on this image/stack (engine-core init fails). `vllm_smoke.json`
  is an **informational** probe, not a gate. The required proof that the GPU actually did work is
  `transformers_smoke` inside `benchmark.json` — an `opt-125m` generate on-GPU. Cloud serving is
  Phase 5 work.
- The `gpu-h100x1-base` image boots cleanly on Hopper (H100/H200): `nvidia-smi`, torch+CUDA, and a
  `transformers` generate all run on-GPU.

---

## Making it your own

This is a *template*. When you fork it for your own lab, these are the things worth changing:

| Where | What |
|---|---|
| `src/ml_lab/gpu/constants.py` | `BASE_TAGS` contains `owner-weldon` — **change the owner tag to yours.** Audit and teardown match droplets by tag/name prefix, so this is how your resources are identified. Also here: `DROPLET_NAME_PREFIX`, `SPACES_KEY_PREFIX`, `DEFAULT_TTL_SECONDS` (2h), the SKU list, and `DO_REGION`. |
| `src/ml_lab/config.py` | `EXPERIMENT_NAME`, `SMOKE_SEED`, row counts, feature/target names. |
| `pyproject.toml` | Project `name`, and the `ml-lab` console-script name if you rename the package. |
| `src/ml_lab/models/`, `data/`, `evaluation/` | Swap the smoke model for your real one — the CLI, tracking, and GPU layers don't care what the model is. |

Keep the safety spine (`teardown`, `audit`, cloud-init self-destruct) intact. It is the part that
stops a forgotten droplet from quietly billing you for a week.

---

## Documentation

- **`.omm/`** — architecture as Mermaid diagrams + prose, five perspectives down to leaf elements.
  **The fastest way to orient.** Start with `state-transitions` (the droplet lifecycle as a state
  machine, including the `stranded` billing state and the three ways out of it). Browse with
  `omm view`. Generated *from* the code — if they ever disagree, the code wins.
- **`docs/superpowers/specs/` and `docs/superpowers/plans/`** — per-slice design docs and runbooks.
  The GPU work was built as slices S1–S4; the S4 spec carries the live results log and the ten
  findings that came out of real execution.
- **`.master_plan/`** — the governing pathway document, cited by every spec.
- **`CLAUDE.md`** — the working agreement for AI-assisted development in this repo.

## Development

Work happens on the `prod` branch. The workflow is brainstorm → spec → plan → subagent-driven
execution, TDD throughout, one plan per small slice. Every external seam (doctl, SSH, boto3) is
isolated behind a single module with an injectable client, which is why the whole suite runs
offline and free.
