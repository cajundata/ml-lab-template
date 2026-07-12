# Phase 0 — GPU Cloud-init Renderer Slice (S3a) (Design)

Date: 2026-07-12
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` §3, "Cloud-init
design", "Recurring self-destruct semantics", "Droplet identity", "Bootstrap
readiness"). The GPU-automation work is decomposed into four slices; S1 (safety
spine) and S2 (create path) shipped. S3 (provision + benchmark) is large, so — per
the one-plan-per-slice preference — it is cut into three sub-slices, and this
design covers the first:

- **S3a — Cloud-init renderer (this slice):** a pure function that renders the
  self-destruct cloud-config into the `user_data` string that S2's
  `create_lab_droplet` already requires, plus the `scripts/cloud-init-gpu.yaml.tmpl`
  template and its self-destruct service/timer units. No network, no droplet, no
  state — entirely offline-testable, like S1/S2. A library slice: nothing calls the
  renderer yet.
- **S3b — Remote seam + orchestration + `gpu-up`:** an SSH/remote seam (mockable
  like `do_client`), `wait_for_ssh`, `wait_for_bootstrap` (verifies the
  self-destruct timer marker), the try/finally teardown wiring, and a working
  `gpu-up`. S3b is the first caller of `render_cloud_init`.
- **S3c — Benchmark + artifacts + Spaces + `gpu-run`:** `gpu_benchmark.py`,
  scp/run/pull, boto3 DO Spaces upload, and the full `gpu-run` lifecycle.

The master plan is the spec of record and locks the cloud-init responsibilities,
the arm-before-install ordering, the recurring-timer semantics, and the
metadata-service droplet-id lookup. This design pins the renderer contract, the
templating mechanism, and the test seams; it does not reopen locked decisions.

## Scope

In scope:
- `scripts/cloud-init-gpu.yaml.tmpl` (new) — the master-plan cloud-config, verbatim
  except for collision-proof placeholders. Carries the full bootstrap through
  `bootstrap-ready.json`. Does **not** embed `gpu_benchmark.py`.
- `src/ml_lab/gpu/cloud_init.py` (new) — `render_cloud_init(...)`.
- `tests/test_gpu_cloud_init.py` (new) — offline, no network, no droplet.
- `pyproject.toml` — add `pyyaml` to the **dev** dependency group (test-only
  YAML-parse check; not a runtime dependency).

Out of scope (later slices):
- Reading `DO_DROPLET_DESTROY_TOKEN` from the environment and calling the renderer —
  S3b. S3a takes `destroy_token` as a required parameter; tests pass a fake.
- SSH/bootstrap orchestration, `wait_for_ssh` / `wait_for_bootstrap`, `gpu-up` — S3b.
- `gpu_benchmark.py`, scp/run/pull, Spaces upload, `gpu-run` — S3c.
- Any live DigitalOcean call — no S3a test hits the network.

## Templating mechanism (decided)

Distinct placeholders + `str.replace()`. The cloud-init body contains literal bash
`${DO_DROPLET_DESTROY_TOKEN}` and `${DROPLET_ID}`, so both `str.format` (brace
fields) and `string.Template` (`$`-substitution) collide with the shell syntax
unless heavily escaped. A collision-proof marker (`@@NAME@@`) plus one
`str.replace` per key leaves bash `${...}` untouched and keeps the template
readable. It also keeps the `.tmpl` file the master plan names in the repo tree.

Placeholders:

| Placeholder | Fills |
| --- | --- |
| `@@RUN_ID@@` | `run.env` `RUN_ID` |
| `@@TTL_SECONDS@@` | `run.env` `TTL_SECONDS` **and** timer `OnBootSec` |
| `@@SELF_DESTRUCT_RETRY_SECONDS@@` | `run.env` retry **and** timer `OnUnitActiveSec` |
| `@@DESTROY_TOKEN@@` | `run.env` `DO_DROPLET_DESTROY_TOKEN` (secret) |
| `@@SMOKE_MODEL_ID@@` | `run.env` benchmark model id |

Each timing value has a single source of truth: it fills both the `run.env` copy
and the systemd unit from the same placeholder.

## `scripts/cloud-init-gpu.yaml.tmpl`

The master-plan cloud-config (§3 "Cloud-init design"), verbatim except for the
placeholders above. It preserves the two locked orderings:

1. `systemctl daemon-reload` → `systemctl enable --now ml-lab-self-destruct.timer`
   → `systemctl is-active --quiet ml-lab-self-destruct.timer` come **first** in
   `runcmd`, before any venv/pip step. The self-destruct net is armed before any
   slow, network-dependent command under our control.
2. `bootstrap-ready.json` (JSON equivalent to
   `{"ready": true, "self_destruct_timer_active": true}`) is written **only after** a
   second `systemctl is-active` check. It is emitted space-free
   (`{"ready":true,"self_destruct_timer_active":true}`) because a `: ` inside the
   inline `echo` would make the whole cloud-config fail `yaml.safe_load`; the two
   forms are identical JSON, and S3b's `wait_for_bootstrap` parses the marker as
   JSON rather than string-matching it.

Other locked properties carried verbatim: `package_update: false`; no `packages:`
block; `run.env` at `0600`; `self_destruct.sh` fetches the droplet id from
`http://169.254.169.254/metadata/v1/id` (no droplet id rendered into cloud-init);
timer `OnBootSec=@@TTL_SECONDS@@`, `OnUnitActiveSec=@@SELF_DESTRUCT_RETRY_SECONDS@@`
(recurring, so a single transient destroy failure does not defeat teardown);
`pip install vllm` only (torch comes transitively). The droplet receives only the
destroy-scoped token — never `DO_API_TOKEN_LOCAL` or Spaces credentials.

The committed template contains placeholders only, never a real token.

## `src/ml_lab/gpu/cloud_init.py`

```python
def render_cloud_init(
    *,
    run_id: str,
    destroy_token: str,                       # secret; lands only in run.env (0600)
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    self_destruct_retry_seconds: int = SELF_DESTRUCT_RETRY_SECONDS,
    smoke_model_id: str = SMOKE_MODEL_ID,
) -> str:
```

Behavior:

1. Read the `.tmpl` relative to the repo (a module-anchored path, not CWD), so the
   renderer works regardless of the caller's working directory.
2. Apply one `str.replace` per placeholder.
3. **Fail-loud completeness guard:** raise `ValueError` if `destroy_token` is empty,
   or if any `@@...@@` marker remains after substitution. An unfilled slot would arm
   a broken self-destruct; this converts that into a pre-create hard stop.
4. Return the rendered YAML string.

All constants the defaults reference already exist in `constants.py`
(`DEFAULT_TTL_SECONDS`, `SELF_DESTRUCT_RETRY_SECONDS`, `SMOKE_MODEL_ID`). No new
constants are introduced.

## Data flow

S3a delivers only the renderer; nothing calls it yet (a library slice, like S2).
In S3b the flow becomes:

```
render_cloud_init(run_id=…, destroy_token=os.environ["DO_DROPLET_DESTROY_TOKEN"], …)
    → user_data string
    → create_lab_droplet(user_data=…)            # S2, already requires it
```

## Safety invariants

- No S3a code path reaches the network or creates a droplet.
- The destroy token appears exactly once in the output: the `run.env` write-file at
  `0600`. Tests assert it appears nowhere else.
- The renderer never prints `user_data` (the secret is not logged).
- The committed `.tmpl` holds placeholders only — no secret in git.
- The completeness guard makes an unfilled placeholder or empty token a hard error,
  not a silently broken self-destruct timer.

## Testing strategy

`tests/test_gpu_cloud_init.py` — all offline; no network, no droplet, no doctl.

- Every placeholder is substituted: no `@@` survives a normal render.
- A leftover placeholder (simulate a bad template via monkeypatched template text
  or a crafted input) → `ValueError`.
- Empty `destroy_token` → `ValueError`.
- Bash `${DROPLET_ID}` and `${DO_DROPLET_DESTROY_TOKEN}` survive verbatim in the
  output.
- Timer values: `OnBootSec` equals `ttl_seconds`; `OnUnitActiveSec` equals
  `self_destruct_retry_seconds` (inject non-default values to prove wiring).
- `run.env` contains `RUN_ID`, `TTL_SECONDS`, the token, and `SMOKE_MODEL_ID`.
- The token appears **only** in the `run.env` write-file section, nowhere else.
- `package_update: false` is present; there is no `packages:` block.
- Ordering: index of `enable --now …self-destruct.timer` is before the index of
  `pip install`; the `bootstrap-ready.json` write is after a `systemctl is-active`
  check.
- The output parses as YAML via `yaml.safe_load` (PyYAML, dev-only) — structural
  validation on top of the string assertions.

## Non-goals / deferred

- No environment reading, no `create_lab_droplet` call — S3b.
- No SSH/bootstrap orchestration, `gpu-up`, or `gpu-run` — S3b/S3c.
- No `gpu_benchmark.py` (cloud-init does not embed it; it is scp'd later) — S3c.
- No live DigitalOcean call or failure-path-gate execution — S4.
- PyYAML is a **test** dependency only; the renderer itself does no YAML parsing.
