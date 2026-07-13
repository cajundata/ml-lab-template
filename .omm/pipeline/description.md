Two independent pipelines. They share no code and never run together — but both exist to prove the same thing: that a lifecycle *completes deterministically and leaves an auditable trail*.

**The local smoke lifecycle** (`make train` -> `make evaluate`): generate seeded synthetic data -> fit -> save -> evaluate the saved artifact against the saved held-out split -> report.

The subtlety worth internalizing is that **training and evaluation write to ONE MLflow run, in two passes.** `start_training_run` creates the run and returns its id; `log_evaluation_metrics` reopens that same run by id and appends to it. The id travels between two separate CLI invocations through a file on disk (`reports/smoke/latest/train_run_id.txt`) and is also stamped into the model's `metadata.json`. So one MLflow row tells the whole story of a model — what data, what params, how it scored in training, and how it scored on held-out data — and any saved model can be traced back to it.

**The GPU benchmark pipeline** runs *on the droplet*: five probes, of which two are required, writing a bundle that is pulled down to `artifacts/<run-id>/` and archived to DO Spaces before the droplet is destroyed.

Neither pipeline's *model* is the point. The determinism (seed `20260706`, no shuffle, SHA-256-checksummed splits) and the evidence trail are the deliverable. Phase 1 swaps the estimator; the rails stay.
