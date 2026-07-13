# ml-lab-template

Reusable ML lab skeleton for the Cajun Data ML Pathway. It runs a deterministic train/evaluate lifecycle with local MLflow tracking and provides billing-safe DigitalOcean GPU automation.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
uv sync --dev
```

Copy `.env.example` to `.env` and fill in credentials before any GPU work. Local train/evaluate does not require credentials.

## Local lifecycle

Run the test suite:

```bash
make test
```

Generate the deterministic smoke data and train the smoke model:

```bash
make train
```

Evaluate the saved model against the saved test split:

```bash
make evaluate
```

Both commands log to the same MLflow run. To inspect runs:

```bash
make mlflow-ui
```

Then open http://127.0.0.1:5000.

## GPU prerequisites

GPU work needs machine-local setup beyond `uv sync` — a `git clone` does **not** bring
any of this. Do it once per machine:

1. **DigitalOcean CLI auth** (machine-local, not in `.env`):

   ```bash
   doctl auth init          # paste a DO API token; verify with: doctl account get
   ```

2. **SSH key registered with DigitalOcean.** The droplet is created with your SSH key so
   the tool can connect. Register a key (or reuse one), then list its numeric id:

   ```bash
   doctl compute ssh-key list      # note the numeric ID column (e.g. 57744215)
   ```

3. **`.env`** — `cp .env.example .env` and fill in:
   - `DO_DROPLET_DESTROY_TOKEN` — a DO API token with droplet **delete** scope (powers
     both the in-droplet self-destruct and local teardown).
   - `DO_SSH_KEY_IDS` — comma-separated **numeric IDs or fingerprints** from step 2
     (NOT the key *name* — doctl rejects names).
   - `DO_SSH_KEY_PATH` — path to the matching **private** key, e.g. `~/.ssh/id_ed25519`
     (NOT the `.pub`).
   - `SPACES_ACCESS_KEY_ID` / `SPACES_SECRET_ACCESS_KEY` / `SPACES_BUCKET` — DO Spaces
     credentials for artifact upload (bucket in the `nyc3` region).

4. **Sanity check** — this should print clean before you create anything:

   ```bash
   make gpu-audit
   ```

Local train/evaluate needs none of this.

## GPU work

GPU droplets are disposable. `down` always means destroy. There is no stop or power-off workflow.

The lab is robust to DigitalOcean's fluctuating GPU capacity: it tries an interchangeable
list of single-GPU SKUs (H100 / H200 / L40S) across regions and lands whichever has
capacity, so you don't have to pick one by hand.

**Warning: a stopped GPU droplet still bills. Never power off a lab droplet. Destroy it.** If you are unsure whether anything is still running, run:

```bash
make gpu-audit
```

Full lifecycle (create, benchmark, pull artifacts, upload, destroy, audit):

```bash
make gpu-run
```

Manual destroy:

```bash
make gpu-down DROPLET_ID=<id>
```