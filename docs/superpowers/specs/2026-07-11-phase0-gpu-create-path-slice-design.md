# Phase 0 — GPU Create Path Slice (S2) (Design)

Date: 2026-07-11
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` §3, "Shared
create path", "Resource tagging", "Normal operational path" steps 1–7). Second of
the four GPU-automation slices (S1 safety spine shipped; this is S2):

- **S1 — Safety spine (done):** constants, `do_client` seam, `audit`,
  `destroy_and_verify`. Backs `make gpu-audit` / `make gpu-down`.
- **S2 — Create path (this slice):** the shared `create_lab_droplet` engine +
  preflight validators + atomic tagging + create-time identity print. A library
  slice — no new `make` target; exercised entirely by mocked tests.
- **S3 — Provision + benchmark:** cloud-init + systemd self-destruct, SSH/
  bootstrap/benchmark orchestration, `gpu_benchmark.py`, artifact pull, DO Spaces
  upload, and the `gpu-run` / `gpu-up` wiring that calls `create_lab_droplet`.
- **S4 — Live failure-path gates.**

**Why a library slice:** the create path is the engine `gpu-run`/`gpu-up` (S3)
drive, but those commands need the SSH/bootstrap/benchmark orchestration and the
cloud-init renderer that are S3's job. So S2 delivers the create engine — fully
tested with a mocked `do_client`, like S1's internals — without a user-facing
command.

The master plan is the spec of record and locks the create sequence, tagging, and
identity-print requirements. This design pins the module structure and the
test seams, and records the account-reality corrections below.

### Account-reality corrections to S1 constants

S1 pinned three constants from the master plan that do not match the actual
DigitalOcean account (verified via `doctl compute size/image list`). S2 corrects
them; this is where they first get validated against the account, so it is the
right slice to fix them:

| Constant | S1 value (from master plan) | S2 value | Why |
| --- | --- | --- | --- |
| `DO_REGION` | `atl1` | `nyc2` | atl1 has no GPU capacity in the account. The master plan explicitly sanctions a deliberate region change ("if capacity is unavailable, change the constant deliberately and rerun; do not add automatic region fallback"). |
| `DO_SIZE_SLUG` | `gpu-rtx4000x1-20gb` | `gpu-4000adax1-20gb` | The real RTX 4000 Ada single-GPU slug (GPU type `nvidia_rtx4000_ada`). |
| `DO_IMAGE_SLUG` | `None` | `gpu-h100x1-base` | The only single-GPU generic "NVIDIA AI/ML Ready Image". |

**Image caveat (documented, verify at S3 live):** the image *slug* is
`gpu-h100x1-base` but its *name* is the generic "NVIDIA AI/ML Ready Image" — DO's
slug naming is legacy. Whether this base image boots on an RTX 4000 Ada droplet can
only be confirmed at S3 live testing. If it fails, the self-destruct timer +
failure-path gates mean no stranded billing, and `validate_constants` hard-fails
pre-create if the slug is ever absent.

## Scope

In scope:
- `src/ml_lab/gpu/constants.py` — correct `DO_REGION`, `DO_SIZE_SLUG`,
  `DO_IMAGE_SLUG` (values above).
- `src/ml_lab/gpu/do_client.py` — add `create_droplet` and the slug-availability
  queries `list_region_slugs`, `list_sizes`, `list_image_slugs`.
- `src/ml_lab/gpu/create.py` (new) — `validate_constants`, `generate_run_id`,
  `build_tags`, `create_lab_droplet`, `LabDropletExistsError`, `ConstantsError`.
- Tests: `tests/test_gpu_create.py` (new); extend `tests/test_gpu_do_client.py`.

Out of scope (later slices):
- `gpu-run` / `gpu-up` commands and orchestration (SSH/bootstrap/benchmark) — S3.
- The cloud-init renderer that produces `user_data` — S3. S2 takes `user_data` as
  a required pass-through string; tests pass a stub.
- SSH-key sourcing — S3 supplies `ssh_key_ids` from config; S2 takes it as an
  optional pass-through.
- Live create against real DigitalOcean — no S2 test hits the network.

## Structure (decided)

Logic lives in `ml_lab.gpu.create`, over the `do_client` seam, mirroring how S1
kept `audit`/`teardown` as pure logic over mocked `do_client`. All preflight gates
live *inside* `create_lab_droplet` so that both `gpu-run` and `gpu-up` (S3) get the
identical safety sequence — neither can bypass a check.

Determinism seam: `create_lab_droplet` accepts injectable `run_id` and `now`
(mirroring `audit.collect_audit`'s `now`), so tests are fully deterministic without
mocking the clock or the RNG.

Safety invariant in the signature: `user_data` (the cloud-init that carries the
self-destruct timer) is a **required** positional parameter with no default — the
function cannot be called to create a droplet without its safety net.

## `do_client.py` additions

```python
def create_droplet(*, name, region, size, image, tags, user_data,
                   ssh_key_ids=None) -> dict:
    """doctl compute droplet create — atomic tags, cloud-init, no --wait.

    Writes user_data to a temp file; runs:
      doctl compute droplet create <name> --region <region> --size <size>
        --image <image> --tag-names <csv(tags)> --user-data-file <tmp>
        [--ssh-keys <csv(ssh_key_ids)>] -o json
    Returns the created droplet dict (has id, name, status="new"). NO --wait, so
    the id is available immediately for the create-time identity print.
    """


def list_region_slugs() -> list[str]     # doctl compute region list
def list_sizes()        -> list[dict]     # doctl compute size list (each with "regions")
def list_image_slugs()  -> list[str]      # doctl compute image list --public
```

`--tag-names` applies tags atomically during create (master plan: "Do not create
first and tag later"). `--user-data-file` takes a temp file rendered from the
`user_data` string. `ssh_key_ids` is passed as `--ssh-keys` only when provided.

## `create.py` design

### `validate_constants()`

```python
def validate_constants() -> None:
    if DO_REGION not in do_client.list_region_slugs():
        raise ConstantsError(f"region {DO_REGION} not available in account")
    size = next((s for s in do_client.list_sizes() if s["slug"] == DO_SIZE_SLUG), None)
    if size is None:
        raise ConstantsError(f"size {DO_SIZE_SLUG} not found")
    if DO_REGION not in (size.get("regions") or []):
        raise ConstantsError(f"size {DO_SIZE_SLUG} not available in {DO_REGION}")
    if DO_IMAGE_SLUG not in do_client.list_image_slugs():
        raise ConstantsError(f"image {DO_IMAGE_SLUG} not available in account")
```

The size-in-region cross-check is the real failure mode (a size can exist globally
but not in `nyc2`), so it is worth the extra assertion. No silent fallback: any
mismatch hard-fails before any droplet is created.

### `generate_run_id(now)`

Returns `"{YYYYMMDD}-{6 hex}"` (e.g. `20260711-a1b2c3`) from
`datetime.fromtimestamp(now, tz=timezone.utc)` + `secrets.token_hex(3)` (tz-aware,
matching `audit.py`'s `_fmt_epoch`; avoids the deprecated `utcfromtimestamp`). The
random suffix keeps droplet names unique across same-day runs.

### `build_tags(run_id, ttl_epoch)`

Returns `BASE_TAGS + [f"run-{run_id}", f"ttl-expiry-{ttl_epoch}"]`.

### `create_lab_droplet(...)`

```python
def create_lab_droplet(
    user_data: str,                       # REQUIRED (cloud-init w/ self-destruct)
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    enforce_budget: bool = True,          # gpu-up --ttl passes False
    ssh_key_ids: list[str] | None = None,
    run_id: str | None = None,            # generated if None (test-injectable)
    now: float | None = None,             # ttl-expiry epoch + run_id date (injectable)
) -> dict:                                # {"id", "name", "run_id"}
```

Sequence (master-plan order — every gate before any create; on any failure,
nothing is created):

1. `list_lab_droplets()` non-empty → raise `LabDropletExistsError` (shared
   duplicate refusal; both `gpu-run` and `gpu-up` inherit it).
2. `validate_constants()`.
3. `probe_destroy_token()`.
4. `if enforce_budget: validate_timeout_budget(ttl_seconds)`.
5. `now = now or time.time()`; `run_id = run_id or generate_run_id(now)`;
   `name = NAME_FORMAT.format(run_id=run_id)`;
   `ttl_epoch = int(now) + ttl_seconds`; `tags = build_tags(run_id, ttl_epoch)`.
6. `droplet = do_client.create_droplet(name=name, region=DO_REGION,
   size=DO_SIZE_SLUG, image=DO_IMAGE_SLUG, tags=tags, user_data=user_data,
   ssh_key_ids=ssh_key_ids)`.
7. Print id / name / `os.getpid()` immediately, before returning — so a dying
   local process never strands an anonymous droplet.
8. `return {"id": droplet["id"], "name": name, "run_id": run_id}`.

The `ttl-expiry` tag is `created_at + TTL` approximated as local `now + ttl` at
create time; it drives `audit`'s overdue detection (S1).

## Errors

- `LabDropletExistsError(RuntimeError)` — pre-create refusal.
- `ConstantsError(RuntimeError)` — a pinned slug is unavailable in the account.
- Propagated unchanged: `ValueError` (timeout budget), `DOClientError` (destroy-
  token probe, doctl failures).

## Testing strategy

All offline; `do_client` mocked; no test reaches the network or creates a droplet.

`tests/test_gpu_create.py`:
- `validate_constants` passes when region present, size present **and** lists
  `nyc2`, image present; raises `ConstantsError` for each: region missing, size
  missing, size-not-in-region, image missing.
- `generate_run_id(now)` → deterministic `YYYYMMDD-` prefix for a fixed `now`;
  format shape asserted.
- `build_tags` → exact `BASE_TAGS + ["run-<id>", "ttl-expiry-<epoch>"]`.
- Create happy path: empty `list_lab_droplets`, queries pass, token env set,
  `create_droplet` returns `{"id": 42, ...}` → returns `{"id":42,
  "name":"ml-lab-gpu-phase0-<run_id>", "run_id":<run_id>}`; asserts `create_droplet`
  called with the exact name/region/size/image/tags/user_data; identity line
  printed (`capsys`).
- Refuse when a lab droplet exists → `LabDropletExistsError` **and**
  `create_droplet` never called.
- `ttl-expiry` tag == `int(now) + ttl_seconds` (inject `now`).
- `enforce_budget=False` lets a 900s TTL through; `enforce_budget=True` with 900s
  raises `ValueError`.
- `ssh_key_ids` and `user_data` pass-through to `create_droplet`.
- Preflight-before-create ordering: if `validate_constants` raises,
  `create_droplet` is never called (nothing created).

Extend `tests/test_gpu_do_client.py`:
- `create_droplet` builds the expected `doctl` argv (mock `subprocess`; assert
  `--tag-names`, `--user-data-file`, `--image`, `--size`, `--region`, `-o json`
  present; no `--wait`) and parses the returned id.
- `list_region_slugs` / `list_sizes` / `list_image_slugs` parse their JSON.

## Safety invariants

- No S2 code path creates or reaches a real droplet (the seam is mocked in every
  test).
- `user_data` is required — no droplet can be created without its self-destruct
  cloud-init.
- All preflight gates live inside the shared create path, so `gpu-run` and
  `gpu-up` (S3) cannot bypass duplicate refusal, constant validation, destroy-token
  probe, or the timeout-budget check.
- Tags are applied atomically at create (never create-then-tag), so audit can
  always find a lab droplet.
- Identity (id/name/pid) is printed immediately after create, before any return.

## Non-goals / deferred

- No `gpu-run` / `gpu-up` command, cloud-init renderer, SSH/bootstrap/benchmark,
  Spaces upload (S3/S4).
- No live DigitalOcean calls or failure-path-gate execution.
- `ssh_key_ids` sourcing and `user_data` rendering are S3's; S2 only threads them
  through as parameters.
