# Phase 0 — GPU Safety Spine Slice (S1) (Design)

Date: 2026-07-11
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` §3, "Provisioning
script design"). GPU automation is the last remaining Phase 0 item and the first
one that touches a paid cloud API and a real GPU, so it cannot be exercised
end-to-end without spending money. It is therefore decomposed into four balanced
slices:

- **S1 — Safety spine (this slice):** constants, the `do_client` seam, `audit`,
  and `destroy_and_verify`. Ships working `make gpu-audit` / `make gpu-down`.
  Fully mockable, no live infra.
- **S2 — Create path:** shared create + atomic tagging + duplicate refusal +
  create-time identity print, plus the create-preflight validators (destroy-token
  probe wiring, timeout-budget check, slug-availability check).
- **S3 — Provision + benchmark:** cloud-init + systemd self-destruct, SSH/
  bootstrap/benchmark orchestration, `gpu_benchmark.py`, artifact pull, DO Spaces
  upload, and `gpu-run` / `gpu-up` wiring.
- **S4 — Live failure-path gates:** runbook + real execution of the two required
  failure-path gates (Ctrl-C mid-run; `kill -9` + TTL self-destruct).

The master plan is the spec of record and locks nearly every *what* decision. This
design pins the *how* for S1 — the module structure and the test seam — and does
not reopen any locked decision.

**Why S1 first:** everything routes through `destroy_and_verify` and `audit`. Building
the safety spine first means `make gpu-audit` and `make gpu-down` exist as a real
safety net *before* any code in the repo can create a billable droplet.

## Scope

In scope:
- `src/ml_lab/gpu/constants.py` — pinned constants + `validate_timeout_budget`.
- `src/ml_lab/gpu/do_client.py` — the DigitalOcean seam (S1 subset).
- `src/ml_lab/gpu/audit.py` — structured audit report, formatter, `is_clean`.
- `src/ml_lab/gpu/teardown.py` — `destroy_and_verify` (SIGINT-hardened).
- `scripts/do_gpu.py` — thin Typer app exposing `audit` and `down` (S1 only).
- `requests` added to `pyproject.toml` dependencies (explicit; used by the
  destroy-token REST probe).
- Tests: `test_gpu_constants.py`, `test_gpu_do_client.py`, `test_gpu_audit.py`,
  `test_gpu_teardown.py`, `test_gpu_cli.py`.

Out of scope (later slices):
- `create_droplet` and any create-preflight *wiring* — S2. (The destroy-token
  probe function lives in `do_client` here so the seam is complete, but nothing in
  S1 calls it at create time; `validate_timeout_budget` is defined here but wired
  into create-preflight in S2.)
- `run` / `up` commands, cloud-init, `gpu_benchmark.py`, Spaces upload — S3.
- Live execution of any failure-path gate — S4.
- `DO_IMAGE_SLUG` resolution — only `create` needs it; resolved in S2.

## Structure (decided)

**DO interface = a thin `do_client` seam.** Intent-level functions
(`destroy_droplet`, `list_lab_droplets`, `get_droplet`, `probe_destroy_token`, …)
hide the master-plan-pinned transport split — `doctl` for list/create/tag, REST
for the destroy-token probe — behind stable names. `audit` and `destroy_and_verify`
are pure logic over the client; tests mock the client functions, so they read like
the safety contract rather than asserting on `doctl` argv or HTTP plumbing. This
mirrors the logic-vs-I/O isolation of the earlier Phase 0 slices.

Logic lives in the `ml_lab.gpu` package; `scripts/do_gpu.py` is a thin Typer entry
(as `cli.py` backs the `ml-lab` console script). The Makefile `gpu-*` targets
already call `uv run python scripts/do_gpu.py <cmd>`.

## Module design

### `src/ml_lab/gpu/constants.py`

Pinned values from master-plan §3 (no fallback), so later slices import rather than
redefine:

```python
DO_REGION           = "atl1"
DO_SIZE_SLUG        = "gpu-rtx4000x1-20gb"
DO_GPU_RUNG         = "RTX 4000 Ada"
DO_IMAGE_SLUG       = None   # DO NVIDIA AI/ML-ready GPU image; resolved in S2 (create-only)
SMOKE_MODEL_ID      = "facebook/opt-125m"

DEFAULT_TTL_SECONDS           = 7200
SSH_TIMEOUT_SECONDS           = 600
BOOTSTRAP_TIMEOUT_SECONDS     = 1800
BENCHMARK_TIMEOUT_SECONDS     = 1800
DESTROY_POLL_TIMEOUT_SECONDS  = 600
SELF_DESTRUCT_RETRY_SECONDS   = 300
DESTROY_POLL_INTERVAL_SECONDS = 10   # cadence for absence polling (added; not in plan)

DROPLET_NAME_PREFIX = "ml-lab-gpu-"
NAME_FORMAT         = "ml-lab-gpu-phase0-{run_id}"
BASE_TAGS           = ["ml-lab", "ml-pathway", "phase-0", "owner-weldon"]
```

Pure validator (no API; wired into create-preflight in S2):

```python
def validate_timeout_budget(ttl_seconds: int) -> None:
    """Raise ValueError unless SSH+BOOTSTRAP+BENCHMARK budget fits under ttl_seconds."""
```

### `src/ml_lab/gpu/do_client.py` (S1 subset)

```python
def list_lab_droplets() -> list[dict]      # doctl compute droplet list -o json; filter by tag + name-prefix
def get_droplet(droplet_id: int) -> dict | None   # None if absent/404
def destroy_droplet(droplet_id: int) -> str       # "gone" (404) | "accepted" (2xx) | "error"
def probe_destroy_token() -> None          # REST DELETE /v2/droplets/1; expect 404; raise on 401/403/network/unexpected-2xx
def list_lab_volumes() -> list[dict]
def list_lab_snapshots() -> list[dict]
def list_lab_reserved_ips() -> list[dict]
def list_lab_load_balancers() -> list[dict]
```

`create_droplet` is intentionally absent — it belongs to S2.

### `src/ml_lab/gpu/audit.py`

```python
def collect_audit() -> AuditReport   # droplets (via tag + name-prefix) + related resources
def format_report(report: AuditReport) -> str
def is_clean(report: AuditReport) -> bool   # no droplets AND no related resources
```

`AuditReport` is a small dataclass holding `droplets: list[DropletInfo]` plus the
four related-resource lists (volumes, snapshots, reserved IPs, load balancers);
`DropletInfo` carries the per-droplet fields below. Per-droplet fields: id, name,
status, region, size, image, age, `ttl-expiry` tag,
overdue status (computed from the tag), public IP, exact destroy command. Search is
by tag set (esp. `ml-lab`) **and** name-prefix `ml-lab-gpu-` as a backstop for a
mistagged droplet. Any match in any status is a failure (stopped/off is still
billable). Related tagged resources — volumes, snapshots, reserved IPs, load
balancers — are scanned too; Phase 0 never creates them, so any hit is a failure.

### `src/ml_lab/gpu/teardown.py`

```python
def destroy_and_verify(droplet_id: int) -> None   # raises on failure → nonzero exit
```

State machine (master-plan §"Shared destroy-and-verify path"):

1. **Arm:** `signal(SIGINT, SIG_IGN)`; print
   `Teardown in progress — droplet <id> is being destroyed. Do not interrupt.`
2. **Destroy:** `do_client.destroy_droplet(id)` — `404`="gone" ✓, `2xx`="accepted",
   transient error → proceed to poll anyway (poll is the source of truth; the remote
   self-destruct timer is the ultimate backstop).
3. **Poll:** `get_droplet(id)` every `DESTROY_POLL_INTERVAL_SECONDS` up to
   `DESTROY_POLL_TIMEOUT_SECONDS`. A second Ctrl-C is ignored; polling continues.
   Uses `time.monotonic()` for the deadline and `time.sleep()` for the interval
   (both patchable in tests).
4. **Audit:** `collect_audit()`.
5. **Succeed** only if the droplet is absent **and** `is_clean(report)`; else raise.
6. `finally`: restore the previous SIGINT handler.

`404` on destroy is success, not failure — the droplet may already be gone (self-
destruct timer, a prior `gpu-down`, or an earlier `destroy_and_verify`). The same
function backs `gpu-down` now and `run`'s finally block + every timeout handler in
S3.

### `scripts/do_gpu.py` (S1 subset)

Thin Typer `app` with two commands:

- `audit` → `collect_audit()` → print `format_report(...)`; exit nonzero if not
  `is_clean`.
- `down --droplet-id <id>` → `destroy_and_verify(id)`.

`run` / `up` are added in S3. The Makefile already wraps all four commands.

## Testing strategy

All S1 tests are deterministic and offline: `do_client` functions are mocked so no
test can reach a real droplet; `time.sleep` / `time.monotonic` are patched so poll-
timeout cases run instantly.

- **`test_gpu_constants.py`** — `validate_timeout_budget(7200)` passes;
  `validate_timeout_budget(900)` raises `ValueError`.
- **`test_gpu_do_client.py`** (subprocess/HTTP mocked) — `get_droplet` None vs dict;
  `destroy_droplet` maps 404/2xx/other → `"gone"`/`"accepted"`/`"error"`;
  `probe_destroy_token` ok on 404, raises on 401/403/network/unexpected-2xx;
  `list_lab_droplets` filters by tag and name-prefix.
- **`test_gpu_audit.py`** — clean → `is_clean` True; droplet present → False with
  all fields + overdue computed from `ttl-expiry`; name-prefix backstop (droplet
  missing the `ml-lab` tag still caught); a tagged volume → dirty.
- **`test_gpu_teardown.py`** — happy; 404 idempotent; already-absent; poll-then-
  absent (assert poll count); poll timeout raises; audit-dirty-after-absence raises;
  transient-error-then-absent → success; **SIGINT unit** (handler is `SIG_IGN`
  inside the loop, restored after on both success and raise paths); **SIGINT
  subprocess** (real child running `destroy_and_verify` against a mocked client that
  stays present for a couple of polls, `SIGINT` fired mid-poll → not aborted,
  completes to contract).
- **`test_gpu_cli.py`** (Typer `CliRunner`) — `--help` names `audit` and `down`;
  `down --droplet-id N` calls `destroy_and_verify(N)`; `audit` exits nonzero on a
  dirty report.

## Determinism / safety invariants

- No S1 code path can create or reach a real droplet; the seam is always mocked in
  tests.
- `destroy_and_verify` succeeds only when the droplet is absent **and** audit is
  clean — the master-plan safety contract.
- Once armed, SIGINT cannot abort teardown; the handler is always restored.

## Non-goals / deferred

- No create, cloud-init, benchmark, or Spaces upload (S2/S3).
- No live cloud calls or failure-path-gate execution (S4).
- No `run` / `up` commands yet.
- `DO_IMAGE_SLUG` stays `None` until S2.
