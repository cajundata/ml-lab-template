# Phase 0 — GPU Remote Seam + `gpu-up` Slice (S3b) (Design)

Date: 2026-07-12
Status: Approved (design), pending implementation plan

## Context

Part of ML Pathway Phase 0 (see `.master_plan/ml_pathway_phase_00` §3, "Interactive
debug path", "Bootstrap readiness", "Shared destroy-and-verify path", "Required
teardown safeguards"). The GPU-automation work is decomposed into four slices; S1
(safety spine), S2 (create path), and S3a (cloud-init renderer) shipped. S3 was
itself cut into three sub-slices, and this design covers the second:

- **S3a — Cloud-init renderer (done):** `render_cloud_init(...)` over
  `scripts/cloud-init-gpu.yaml.tmpl`, producing the `user_data` self-destruct
  cloud-init. Library slice; S3b is its first caller.
- **S3b — Remote seam + `gpu-up` (this slice):** an SSH/scp subprocess seam
  (`wait_for_ssh`, `wait_for_bootstrap`), public-IP resolution, the `gpu_up`
  orchestration with its try/except-teardown discipline, env-input loading, and the
  `up` CLI command. First real caller of `render_cloud_init` and `create_lab_droplet`.
- **S3c — Benchmark + artifacts + Spaces + `gpu-run`:** `_run_scp`,
  `gpu_benchmark.py`, scp/run/pull, boto3 DO Spaces upload, full `gpu-run` lifecycle.
- **S4 — Live failure-path gates:** real droplets, real money; the Ctrl-C and
  `kill -9` self-destruct tests; first confirmation the image boots on RTX 4000 Ada.

The master plan is the spec of record and locks the `gpu-up` sequence, the
verified-handoff requirement, the "leave alive on success / destroy on any other
exit" rule, and the shared-destroy chokepoint. This design pins module structure,
the remote seam mechanism, env-input sourcing, and the test seams. Every S3b test is
offline: `remote`, `do_client`, and `destroy_and_verify` are mocked, exactly like
S1/S2. Live SSH/droplet execution is S4.

## Scope

In scope:
- `src/ml_lab/gpu/remote.py` (new) — SSH/scp subprocess seam: `wait_for_ssh`,
  `wait_for_bootstrap`, `RemoteError`.
- `src/ml_lab/gpu/gpu_env.py` (new) — `GpuEnv`, `load_gpu_env`, `GpuEnvError`;
  first `load_dotenv()` in the project.
- `src/ml_lab/gpu/lifecycle.py` (new) — `gpu_up`, `wait_for_public_ip`.
- `src/ml_lab/gpu/do_client.py` (modify) — add `public_ipv4(droplet)` helper
  (factored from the extraction `audit.py` already does); wrap `_run_doctl`'s
  `subprocess.CalledProcessError` in `DOClientError` (the deferred S1 follow-up).
- `src/ml_lab/gpu/cli.py` (modify) — add the `up` Typer command.
- `.env.example` (modify) — ensure `DO_DROPLET_DESTROY_TOKEN`, `DO_SSH_KEY_IDS`,
  `DO_SSH_KEY_PATH` are present.
- Tests: `tests/test_gpu_remote.py`, `tests/test_gpu_env.py`,
  `tests/test_gpu_lifecycle.py` (new); extend `tests/test_gpu_do_client.py` and
  `tests/test_gpu_cli.py`.

Out of scope (later slices):
- `_run_scp`, `gpu_benchmark.py`, artifact pull, DO Spaces upload, `gpu-run` — S3c.
- Live SSH, live droplets, the failure-path gates, RTX 4000 Ada boot confirmation — S4.

## Remote seam mechanism (decided)

Subprocess around the system OpenSSH `ssh`/`scp` binaries, mirroring `do_client`'s
doctl-subprocess pattern — no new dependency, and tests monkeypatch
`remote.subprocess.run` the same way the `do_client` tests already do. (paramiko and
Fabric were considered and rejected: a runtime SSH dependency and a second SSH idiom
alongside the existing doctl-subprocess style, for what are a few one-shot calls.)

## SSH-key sourcing (decided)

Two explicit environment variables, read by `load_gpu_env`:
- `DO_SSH_KEY_IDS` — comma-separated DigitalOcean key ids/fingerprints; split into a
  list and passed to `create_lab_droplet(ssh_key_ids=...)` so the droplet authorizes
  the operator's key at create.
- `DO_SSH_KEY_PATH` — the local private key; passed to `ssh` as `-i <path>`.

Explicit and reproducible across machines (preferred for a reusable template over
relying on the operator's ssh-agent). The outbound connection is always `root@<ip>`
with hardening options baked in (below); the key path is a **pure env read** — it is
not stat-ed pre-create. A bad path surfaces as a loud `wait_for_ssh` failure that
then tears the droplet down, so there is no billing risk in deferring that check.

## `remote.py`

Hardened, non-interactive SSH so a hung network cannot block forever and a fresh host
key cannot prompt:

```python
SSH_OPTS = [
    "-o", "BatchMode=yes",                 # never prompt (no password/passphrase stalls)
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "UserKnownHostsFile=/dev/null",  # ephemeral hosts; don't pollute known_hosts
    "-o", "LogLevel=ERROR",
]


class RemoteError(RuntimeError):
    """SSH was not reachable, or bootstrap was not verified, before the deadline."""


def _run_ssh(host, argv, *, key_path, timeout):
    # ssh -i <key_path> <SSH_OPTS> root@<host> <argv...>, capture_output=True, text=True
    ...
```

- **`wait_for_ssh(host, *, key_path, timeout=SSH_TIMEOUT_SECONDS, interval=..., now=None, sleep=None)`**
  — deadline loop (mirroring `teardown._poll_until_absent`) running
  `_run_ssh(host, ["true"], ...)` until exit 0; raises `RemoteError` ("not reachable
  after Ns") at the deadline. Each attempt's own `ConnectTimeout` bounds one try; the
  outer deadline bounds the whole wait.
- **`wait_for_bootstrap(host, *, key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS, interval=..., now=None, sleep=None)`**
  — deadline loop running `_run_ssh(host, ["cat", "/opt/ml-lab/bootstrap-ready.json"], ...)`.
  On a zero-exit, `json.loads` the stdout and require **both** `ready is True` and
  `self_destruct_timer_active is True` (parse-as-JSON, not string-match — the reason
  S3a emits the marker as space-free JSON). Missing file / non-zero exit / unparseable
  / flags-not-both-true → keep polling; at the deadline raise `RemoteError`. Returns the
  parsed marker dict on success.

The injectable `now`/`sleep` seam keeps the deadline tests instant and deterministic.
`scp` is not built here — it is unused until S3c's benchmark delivery (YAGNI).

## `gpu_env.py`

```python
@dataclass(frozen=True)
class GpuEnv:
    destroy_token: str      # DO_DROPLET_DESTROY_TOKEN
    ssh_key_ids: list[str]  # DO_SSH_KEY_IDS, comma-split
    ssh_key_path: str       # DO_SSH_KEY_PATH


class GpuEnvError(RuntimeError):
    """One or more required GPU env vars are missing/empty."""


def load_gpu_env() -> GpuEnv:
    load_dotenv()  # first .env load in the project
    # read the three vars; raise GpuEnvError listing EVERY missing/empty one at once
```

- `load_dotenv()` (python-dotenv, already a dependency) reads `.env` — the project's
  first use of it.
- **Fail loud, all at once:** `GpuEnvError` reports every missing/empty var in a single
  message so the operator fixes `.env` in one pass before any droplet exists.
- `destroy_token` non-empty is validated here, then handed to `render_cloud_init`
  (which independently guards empty — defense in depth) and, at create time, to
  `probe_destroy_token` (which reads the same env var).
- `ssh_key_path` is validated set/non-empty only — not stat-ed (pure env read).

## IP resolution

`create_lab_droplet` returns `status="new"` with no network populated, so the
public IPv4 is resolved by polling after create, via a shared extractor:

```python
# do_client.py — factored from the logic audit.py already uses
def public_ipv4(droplet: dict) -> str | None:
    for net in (droplet.get("networks", {}).get("v4") or []):
        if net.get("type") == "public":
            return net.get("ip_address")
    return None
```

`lifecycle.wait_for_public_ip(droplet_id, *, timeout=SSH_TIMEOUT_SECONDS, now=None, sleep=None)`
wraps it in a deadline loop (reusing `do_client.get_droplet`), raising `RemoteError`
if no public IP appears in time. It runs between create and `wait_for_ssh`.

## `lifecycle.gpu_up`

The heart of the slice. Invariant (master-plan-locked): **once a droplet exists,
every non-clean exit routes through `destroy_and_verify`; a clean, fully verified
success leaves the droplet alive** (the interactive/debug path).

```python
def gpu_up(*, ttl_seconds=DEFAULT_TTL_SECONDS, enforce_budget=True,
           env=None, now=None) -> dict:
    env = env or load_gpu_env()
    run_id = generate_run_id(now if now is not None else time.time())
    user_data = render_cloud_init(run_id=run_id, destroy_token=env.destroy_token,
                                  ttl_seconds=ttl_seconds)
    droplet_id = None
    try:
        result = create_lab_droplet(               # preflight gates + prints id/name/pid
            user_data, ttl_seconds=ttl_seconds, enforce_budget=enforce_budget,
            ssh_key_ids=env.ssh_key_ids, run_id=run_id, now=now)
        droplet_id = result["id"]
        ip = wait_for_public_ip(droplet_id, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_ssh(ip, key_path=env.ssh_key_path, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_bootstrap(ip, key_path=env.ssh_key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        _print_handoff(result, ip, ttl_seconds)     # only after bootstrap verified
        return {**result, "ip": ip}                  # SUCCESS: droplet left ALIVE
    except BaseException:
        if droplet_id is not None:
            destroy_and_verify(droplet_id)           # SIGINT-hardened; 404/absent = success
        raise
```

- **`except BaseException`** (not `Exception`) so a `KeyboardInterrupt` *after* create
  still tears down. `destroy_and_verify` already ignores SIGINT once armed, so a second
  Ctrl-C during teardown cannot strand the droplet.
- **`droplet_id is None` guard:** a preflight failure (duplicate refusal, bad constants,
  dead destroy-token, budget) raises *before* any droplet exists — nothing to destroy.
  `create_lab_droplet` prints id/name/pid itself the instant create returns, so a dying
  process never strands an anonymous droplet.
- **Handoff prints only after `wait_for_bootstrap` succeeds** — never claims the
  self-destruct timer is armed unverified. Exact master-plan output: id, name, region,
  size, ttl, self-destruct retry, `make gpu-down DROPLET_ID=<id>`, "Self-destruct timer
  has been verified active on the droplet.", "Do not power off this droplet. Destroy it."
- **Two modes, one path:** default `gpu-up` → `enforce_budget=True`; `gpu-up
  --ttl-seconds 900` → `ttl_seconds=900, enforce_budget=False` (the `create_lab_droplet`
  escape hatch from S2). The `kill -9` self-destruct fixture is just the operator killing
  the process after the id print — the same code, no special branch.
- **Timeout → teardown:** any `RemoteError` (no IP, SSH unreachable, bootstrap
  unverified) propagates into the `except`, destroys, and re-raises → the CLI exits
  nonzero.

Noted interaction: a `--ttl-seconds 900` run that is *not* killed has a TTL shorter than
the SSH+bootstrap wait budget, so the droplet may self-destruct mid-wait → `wait_for_*`
fails → `destroy_and_verify` finds it already gone (404 → success). Correct behavior.

## CLI wiring

`cli.py` gains a thin `up` command (matching the existing `audit`/`down` style):

```python
@app.command("up")
def up_command(
    ttl_seconds: int = typer.Option(None, "--ttl-seconds",
        help="Short-fuse self-destruct test; bypasses the benchmark-budget check."),
):
    if ttl_seconds is None:
        lifecycle.gpu_up()                                   # default: enforce_budget=True
    else:
        lifecycle.gpu_up(ttl_seconds=ttl_seconds, enforce_budget=False)
```

`GpuEnvError` / `RemoteError` / `ConstantsError` / `LabDropletExistsError` surface as a
nonzero exit. `scripts/do_gpu.py` already imports `app`, so `make gpu-up` and
`make gpu-up --ttl-seconds …` start working.

## Errors

- `RemoteError(RuntimeError)` — SSH unreachable or bootstrap not verified before the
  deadline; also raised by `wait_for_public_ip` when no public IP appears in time.
- `GpuEnvError(RuntimeError)` — missing/empty required env vars.
- Propagated unchanged: `ConstantsError`, `LabDropletExistsError`, `DOClientError`,
  `ValueError` (timeout budget / empty token), `TeardownError`.
- `_run_doctl` now wraps `subprocess.CalledProcessError` in `DOClientError` so an
  unauthenticated `doctl` fails cleanly instead of dumping a traceback.

## Testing strategy

All offline; `remote`, `do_client`, and `destroy_and_verify` mocked; no test reaches
the network, a droplet, or SSH.

`tests/test_gpu_lifecycle.py`:
- Happy path → creates, resolves IP, waits ssh+bootstrap, prints the handoff, returns
  `{id,name,run_id,ip}`; **`destroy_and_verify` never called** (alive on success).
- SSH timeout (`wait_for_ssh` raises `RemoteError`) → `destroy_and_verify(id)` called
  exactly once → re-raises.
- Bootstrap unverified (`wait_for_bootstrap` raises) → same teardown path.
- `KeyboardInterrupt` raised after create → teardown runs (the `BaseException` guard),
  asserted.
- Preflight failure (`create_lab_droplet` raises `LabDropletExistsError`) →
  `destroy_and_verify` **not** called (no droplet existed).
- `enforce_budget` threading: default rejects a 900s TTL; `--ttl-seconds 900`
  (`enforce_budget=False`) is allowed through.
- `wait_for_public_ip` returns the IP once networks populate; times out → `RemoteError`.

`tests/test_gpu_remote.py`:
- `wait_for_ssh` succeeds after N failed polls; never reachable → `RemoteError` at the
  deadline (injected clock).
- `wait_for_bootstrap` returns the marker dict when both flags true; keeps polling then
  `RemoteError` when `self_destruct_timer_active` is false; keeps polling on malformed
  JSON / missing file.

`tests/test_gpu_env.py`:
- `load_gpu_env` happy read splits `DO_SSH_KEY_IDS`; every missing/empty var is named in
  a single `GpuEnvError`.

Extend `tests/test_gpu_do_client.py`:
- `public_ipv4` extracts the public IPv4; returns `None` when only a private network (or
  none) is present.
- `_run_doctl` raises `DOClientError` (not raw `CalledProcessError`) on a non-zero
  doctl exit.

Extend `tests/test_gpu_cli.py`:
- `up` with no `--ttl-seconds` calls `gpu_up()` with defaults; `up --ttl-seconds 900`
  calls `gpu_up(ttl_seconds=900, enforce_budget=False)` (monkeypatch `lifecycle.gpu_up`).

## Safety invariants

- No S3b test touches a real droplet or a real SSH connection (every seam mocked).
- The only success path that leaves a droplet alive is a **fully verified** bootstrap;
  every other post-create exit (error, timeout, `KeyboardInterrupt`) destroys.
- `destroy_and_verify` remains the single teardown chokepoint; S3b adds callers, not a
  second teardown path.
- The destroy token never leaves env → `run.env`: it is not logged and never appears in
  an ssh/doctl argv.
- The verified-handoff message is never printed before `wait_for_bootstrap` returns.

## Non-goals / deferred

- No `_run_scp`, `gpu_benchmark.py`, artifact pull, or DO Spaces upload — S3c.
- No `gpu-run` command or its always-destroy `finally` lifecycle — S3c (which will
  reuse the create/try-teardown spine established here).
- No live SSH, live droplets, or failure-path-gate execution — S4.
- No RTX 4000 Ada boot confirmation — S4 (first live run).
