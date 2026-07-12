# Phase 0 — Spaces Upload + `gpu_run` Lifecycle + `run` Command Slice (S3c-2) (Design)

Date: 2026-07-12
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` §3, "Normal
operational path" steps 1–16, "Shared destroy-and-verify path", "vLLM smoke check",
and the DO Spaces upload rows in the constants block). The GPU-automation work is
decomposed into slices; S1 (safety spine), S2 (create path), S3a (cloud-init
renderer), S3b (remote seam + `gpu-up`), and S3c-1 (scp seam + `gpu_benchmark.py` +
local drive/pull driver) shipped. S3c was cut into two sub-slices per the
one-plan-per-slice preference; this design covers the second and last:

- **S3c-1 (shipped) — scp seam + benchmark script + local drive/pull:** the scp
  primitives, `scripts/gpu_benchmark.py`, and the local driver
  (`deliver_and_run_benchmark`, `pull_artifacts`). A library slice: nothing wired
  these into a lifecycle.
- **S3c-2 — Spaces upload + `gpu_run` lifecycle + `run` command (this slice):** the
  boto3 DO Spaces upload seam, the full create → wait → benchmark → pull → upload →
  always-destroy `gpu_run` lifecycle, and the `run` Typer command. S3c-2 is the
  first (and only) caller of S3c-1's driver functions and the last slice before the
  S4 live run.

The master plan is the spec of record. It locks the `gpu-run` step sequence (1–16),
the "destroy beats artifact preservation" ordering, the exit-code contract (a failed
benchmark still pulls/uploads then surfaces failure; transport/timeout failures
route through the shared teardown), the Spaces object layout, and the fact that
`run` and `up` share one create path. This design pins the module layout, the Spaces
seam interface, the Spaces env/config source, the `gpu_run` control flow, and the
test seams; it does not reopen locked decisions.

## Scope

In scope:
- `src/ml_lab/gpu/spaces.py` (new) — the boto3 DO Spaces seam: `upload_bundle` plus
  an injectable client factory and a `SpacesError`.
- `src/ml_lab/gpu/gpu_env.py` (extend) — `SpacesEnv` + `load_spaces_env()` beside the
  existing `GpuEnv` / `load_gpu_env()`.
- `src/ml_lab/gpu/constants.py` (extend) — `SPACES_REGION`, `SPACES_ENDPOINT`,
  `SPACES_KEY_PREFIX`.
- `src/ml_lab/gpu/lifecycle.py` (extend) — `gpu_run()` beside `gpu_up()`.
- `src/ml_lab/gpu/cli.py` (extend) — the `run` command.
- `scripts/do_gpu.py` (docstring) — mention `run`.
- `tests/test_gpu_spaces.py` (new), `tests/test_gpu_env.py` (extend),
  `tests/test_gpu_lifecycle.py` (extend), `tests/test_gpu_cli.py` (extend) — all
  offline, every seam mocked.

Out of scope (later slices / gates):
- Any live DigitalOcean call, live SSH, live scp, live boto3 upload, or actual
  `torch` / `vllm` execution — S4 (the first live run also confirms the image boots
  on RTX 4000 Ada).
- No changes to the internals of `create.py`, `remote.py`, `teardown.py`,
  `benchmark.py`, or `scripts/gpu_benchmark.py` — S3c-2 only *calls* them.
- No new `make` target. `Makefile`'s `gpu-run` already targets `do_gpu.py run`; this
  slice makes that target functional with no Makefile change.

## Module layout (decided)

The boto3 upload lives in a **new** `ml_lab/gpu/spaces.py`, isolated exactly like the
existing `do_client.py` (doctl/REST) and `remote.py` (ssh/scp) seams: one external
dependency, one public function, an injectable factory for the network client, and a
seam-specific error. This keeps `lifecycle.py` free of boto3 specifics — it orchestrates
seams, it does not embed them.

Spaces credentials load through a **second loader in the existing `gpu_env.py`**
(`load_spaces_env`), not a merge into `GpuEnv`. `gpu-up` needs the DO/SSH vars but
never Spaces; only `gpu-run` needs Spaces. Two loaders keep each command's fail-loud
surface to exactly what it uses: `up` does not fail on an unset `SPACES_BUCKET`.

`gpu_run` is a **self-contained function beside `gpu_up`**, not a refactor that
factors out a shared create→wait helper. The two functions share the first four steps
(create, wait-for-IP, wait-for-SSH, wait-for-bootstrap) but diverge fundamentally:
`gpu_up` leaves a verified droplet **alive** (the debug path) while `gpu_run`
**always destroys** and adds three steps (benchmark, pull, upload). A few lines of
provision duplication buys two independently-readable, independently-testable
functions and keeps the always-destroy `finally` legible in one screen — consistent
with this repo's clarity-over-DRY lifecycle style. No premature extraction.

## Spaces seam interface (decided)

```
upload_bundle(local_dir, run_id, *, env, client=None) -> str
```

- `local_dir` is the `artifacts/<run-id>/` directory `pull_artifacts` populated.
- Walks every file under `local_dir` (recursively) and uploads each to key
  `f"{SPACES_KEY_PREFIX}/{run_id}/{relpath}"` in `env.bucket`, where `relpath` is the
  file's path relative to `local_dir` (POSIX separators). This reproduces the layout
  the `artifacts/README.md` promises: `s3://<bucket>/ml-pathway/phase0/<run-id>/…`.
- `client` defaults to a boto3 S3 client built against `SPACES_ENDPOINT` with
  `env.access_key` / `env.secret_key`. Tests always pass a mock, so no test touches
  boto3's network or requires real credentials.
- Raises `SpacesError` on a missing/empty `local_dir` or any boto3 failure
  (`BotoCoreError` / `ClientError`), chaining the original.
- Returns the `s3://<bucket>/{SPACES_KEY_PREFIX}/<run-id>/` URI for the confirmation
  print.

### Constants (decided)

```
SPACES_REGION = "nyc3"                                      # DO Spaces region; change deliberately, no fallback
SPACES_ENDPOINT = f"https://{SPACES_REGION}.digitaloceanspaces.com"
SPACES_KEY_PREFIX = "ml-pathway/phase0"                     # matches artifacts/README.md
```

`SPACES_REGION` is a pinned constant, not an env var — the same "change the constant
deliberately, never auto-fall-back" discipline as `DO_REGION` / `DO_SIZE_SLUG` /
`DO_IMAGE_SLUG`. The bucket is per-region, so the region belongs with the other pinned
placement constants. `nyc3` is a DO Spaces region geographically reasonable for the
`nyc2` droplet region; an operator whose bucket lives elsewhere edits one line.

### `load_spaces_env()` (decided)

Returns a frozen `SpacesEnv(access_key, secret_key, bucket)` from
`SPACES_ACCESS_KEY_ID` / `SPACES_SECRET_ACCESS_KEY` / `SPACES_BUCKET`, raising
`GpuEnvError` (the existing error type) naming **all** missing/empty vars at once —
identical shape to `load_gpu_env`. No new env vars are added to `.env.example`; the
three Spaces vars are already present.

## `gpu_run` control flow and exit-code contract (decided)

```python
def gpu_run(*, ttl_seconds=DEFAULT_TTL_SECONDS, env=None, spaces=None, now=None) -> int:
    if env is None:
        env = load_gpu_env()
    if spaces is None:
        spaces = load_spaces_env()          # preflight — before any droplet exists
    run_now = now if now is not None else time.time()
    run_id = generate_run_id(run_now)
    user_data = render_cloud_init(
        run_id=run_id, destroy_token=env.destroy_token, ttl_seconds=ttl_seconds
    )

    droplet_id = None
    try:
        result = create_lab_droplet(
            user_data, ttl_seconds=ttl_seconds, enforce_budget=True,
            ssh_key_ids=env.ssh_key_ids, run_id=run_id, now=run_now,
        )                                    # prints id/name/pid; refuses if a lab droplet exists
        droplet_id = result["id"]
        ip = wait_for_public_ip(droplet_id, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_ssh(ip, key_path=env.ssh_key_path, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_bootstrap(ip, key_path=env.ssh_key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        code = deliver_and_run_benchmark(ip, run_id, key_path=env.ssh_key_path)  # nonzero ≠ raise
        dest = pull_artifacts(ip, run_id, key_path=env.ssh_key_path)
        uri = upload_bundle(dest, run_id, env=spaces)
        print(f"Uploaded benchmark bundle to {uri}", flush=True)
        return code
    finally:
        if droplet_id is not None:
            destroy_and_verify(droplet_id)
```

Behavior, mapped to master-plan `gpu-run` steps 1–16 and the failure clauses:

- **Preflight (env + Spaces) runs before create.** A misconfigured token, key, or
  bucket fails loud with zero billable resources — nothing to tear down.
- **Completed-but-failed benchmark** (`deliver_and_run_benchmark` returns a nonzero
  code without raising): the lifecycle still pulls the partial bundle, still uploads
  it, then returns the nonzero code; the `finally` still destroys. Master plan:
  "If benchmark fails, destroy still runs" and the partial bundle is preserved.
- **SSH-timeout / bootstrap-timeout / benchmark-timeout / transport / upload
  failures** raise (`RemoteError` from the driver and waits, `SpacesError` from
  upload). The `finally` destroys, then the exception propagates to a nonzero exit.
  Master plan: "Destroy beats artifact preservation."
- **Teardown failure** in the `finally` raises `TeardownError` — the louder alarm —
  which overrides any in-flight exception, chaining it as `__context__` (the existing
  destroy-and-verify invariant; unchanged).
- **Success exit code is the benchmark's exit code.** A clean run returns `0` only if
  the benchmark passed, the bundle uploaded, and `destroy_and_verify` confirmed the
  droplet is gone and audit is clean.

`gpu_run` prints no "leave alive / do not power off" handoff — unlike `gpu_up`, it
always destroys. The create-time id/name/pid print (inside `create_lab_droplet`) and
the teardown print (inside `destroy_and_verify`) remain the operational breadcrumbs;
`gpu_run` adds only the one-line upload confirmation.

## `run` command (decided)

```python
@app.command("run")
def run_command() -> None:
    """Full GPU lifecycle: create, benchmark, pull, upload to Spaces, always destroy."""
    raise typer.Exit(code=gpu_run())
```

Default TTL, so `create_lab_droplet` enforces the timeout-budget gate
(`enforce_budget=True`). No `--ttl-seconds` override: the short-fuse mode is a
debug affordance of `up`, not `run`. `scripts/do_gpu.py`'s docstring is updated to
list `run`; the Makefile already wraps it.

## Test seams (all offline)

- `tests/test_gpu_spaces.py` (new): pass a mock client to `upload_bundle`; assert
  every file under a temp bundle dir is uploaded once with the correct
  `ml-pathway/phase0/<run-id>/<relpath>` key and bucket; nested subdirs use POSIX
  relpaths; a client that raises `ClientError` → `SpacesError`; a missing `local_dir`
  → `SpacesError`; the returned URI is well-formed. The default (real-boto3) client
  factory is exercised only via a monkeypatched `boto3.client` asserting endpoint and
  credentials are passed — never a live call.
- `tests/test_gpu_env.py` (extend): `load_spaces_env` happy path returns a
  `SpacesEnv`; missing vars raise `GpuEnvError` naming all of them.
- `tests/test_gpu_lifecycle.py` (extend): mock `create_lab_droplet`, the waits,
  `deliver_and_run_benchmark`, `pull_artifacts`, `upload_bundle`, and
  `destroy_and_verify`. Assert: step ordering; preflight (`load_gpu_env` /
  `load_spaces_env`) runs before create and a preflight failure creates nothing;
  the exit code equals the benchmark code; a nonzero benchmark code still pulls +
  uploads before destroy; each raising step (wait/benchmark/pull/upload) still routes
  through `destroy_and_verify`; a `TeardownError` in the `finally` overrides an
  in-flight error.
- `tests/test_gpu_cli.py` (extend): `run` exits with `gpu_run`'s returned code (0 and
  nonzero), with `gpu_run` mocked.

No S3c-2 code path reaches the network, a droplet, an SSH/scp connection, boto3's
wire, or a GPU. Every seam is injected or monkeypatched.

## What this slice deliberately does not do

- No live DO/SSH/scp/boto3 call and no real `torch`/`vllm` — S4.
- No new GPU class, region change, or image change — the pinned constants stand.
- No changes to the shared create/teardown/remote/driver internals — `gpu_run` is a
  caller, not an editor, of those seams.
- No `--ttl-seconds` on `run` and no override of the "refuse if a lab droplet exists"
  guard — both are intentional Phase 0 boundaries.
