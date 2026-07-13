This is Phase 0 of a multi-phase ML pathway. It is deliberately a *skeleton*: the model is a 4-feature logistic regression on synthetic data whose only job is to prove the lifecycle end-to-end. Do not mistake the smoke model for the point of the repo — the point is the rails around it.

Governing documents:
- `.master_plan/` is the spec of record (pathway direction + `ml_pathway_phase_00` detail), tracked in git.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` hold per-slice design + runbooks. The S4 spec (`2026-07-12-phase0-gpu-live-gates-slice-design.md`) carries the live Results log and the 10 findings from first contact with real hardware.

The GPU work was cut into slices S1..S4. Every GPU seam was *mocked* through S3; S4 was the first live execution, and it is where the hard-won facts in `CLAUDE.md` came from. Python is the model-development language; Go is planned as the durable service layer in later phases (`src/ml_lab/serving/` is an empty placeholder for that).

Next up is Phase 1 (tabular). vLLM serving is deferred to the Phase-5 cloud-serving track.
