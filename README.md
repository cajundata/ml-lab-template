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

## GPU work

GPU droplets are disposable. `down` always means destroy. There is no stop or power-off workflow.

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