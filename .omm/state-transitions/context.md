Every GPU seam in this repo was **mocked** through slices S1-S3. S4 was the first live execution against real DigitalOcean hardware, and this state machine is largely a record of what S4 taught.

The transitions that exist *because* of live findings:
- The `provisioning` **self-loop** — capacity flapping across both regions and SKUs was not anticipated; the original design pinned one `(size, region)` pair.
- The `preflight -> destroyed` edge being *free* — the four gates were consolidated inside `create_lab_droplet` so that a misconfiguration can never strand anything.
- The `stranded -> self-destruct -> destroyed` path — **proven live** by `kill -9`-ing a run and confirming the remote timer reaped the droplet with zero local involvement.
- The `verified -> destroyed` Ctrl-C edge — **proven live**, including that a second Ctrl-C during teardown is correctly ignored.

The full live Results log and all 10 findings are in `docs/superpowers/specs/2026-07-12-phase0-gpu-live-gates-slice-design.md`.
