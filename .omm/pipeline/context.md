This is Phase 0: a *skeleton*. The 4-feature logistic regression on synthetic data has exactly one job — prove the train/evaluate/track lifecycle works end-to-end, deterministically, with an auditable trail. It is scaffolding, and it is meant to be thrown away.

**Phase 1 (tabular) is the next slice**, and the seams it will use are already in place:
- `SmokeModel` wraps the estimator with its schema and metadata. Swap the estimator; keep the wrapper.
- Every path and knob in `train_smoke()` / `evaluate_smoke()` is a keyword argument defaulting to a `config.py` constant, so the pipeline is already parameterizable — today's commands just take no arguments.
- The MLflow two-pass pattern (one run, training then evaluation) is the durable part.

The GPU benchmark pipeline is not an ML pipeline at all — it is a **hardware proof**. It answers "does this droplet actually have a working GPU that can run a real model?", and its output is evidence, not a model.
