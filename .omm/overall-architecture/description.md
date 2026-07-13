`ml-lab-template` is a reusable ML lab skeleton for the Cajun Data ML Pathway. It has two halves that share a CLI surface, a pinned-constants layer, and a local workspace, but otherwise do not touch each other:

1. **A deterministic local train/evaluate lifecycle** (`data/` -> `models/` -> `evaluation/` -> `tracking/`). Seeded synthetic data, a logistic-regression smoke model, and a single MLflow run that carries both training and evaluation metrics. Everything runs on the local file store; no network.

2. **Billing-safe DigitalOcean GPU automation** (`src/ml_lab/gpu/`). One command creates a GPU droplet, verifies it booted with an armed self-destruct timer, runs a benchmark on it, pulls the artifact bundle, uploads it to DO Spaces, and *always* destroys the droplet. Every external seam (doctl, ssh/scp, boto3) is an isolated module, mocked in every test.

The dominant architectural force is **cost safety**, not model quality. A GPU droplet bills by the hour whether or not it is doing work, so the design treats "a droplet exists that we did not destroy" as the primary failure mode. Phase 0 is complete: `make test` is 193 tests, and both halves have been proven end-to-end (the GPU half live, on real hardware).
