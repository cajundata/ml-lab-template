# Spaces Upload + `gpu_run` Lifecycle + `run` Command Slice (S3c-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the shipped S3c-1 driver into a full lifecycle: add the boto3 DO Spaces upload seam, the create → wait → benchmark → pull → upload → always-destroy `gpu_run` function, and the `run` Typer command — every seam mocked, nothing live.

**Architecture:** A new `ml_lab/gpu/spaces.py` isolates boto3 exactly like `do_client.py`/`remote.py`: one public `upload_bundle`, a lazy-imported injectable client factory, one `SpacesError`. Spaces credentials load through a second fail-loud loader in the existing `gpu_env.py` (`load_spaces_env`), separate from `GpuEnv` because only `run` needs them. `gpu_run` lives beside `gpu_up` in `lifecycle.py` as a self-contained function whose `finally` always calls `destroy_and_verify`; it returns the benchmark's exit code, and a completed-but-failed benchmark still pulls + uploads the partial bundle before teardown. The `run` command is a one-liner that exits with `gpu_run`'s code.

**Tech Stack:** Python 3, `boto3` (already a dependency), `pathlib` (bundle walk), Typer (CLI), pytest + `monkeypatch` (every seam injected/mocked — no boto3 wire, no droplet, no SSH). Reuses `constants.py`, `gpu_env.py`, `create.py`, `remote.py`, `benchmark.py`, `teardown.py`.

---

## File structure

- Modify `src/ml_lab/gpu/constants.py` — add `SPACES_REGION`, `SPACES_ENDPOINT`, `SPACES_KEY_PREFIX`.
- Modify `src/ml_lab/gpu/gpu_env.py` — add `SpacesEnv` dataclass + `load_spaces_env()`.
- Create `src/ml_lab/gpu/spaces.py` — the boto3 Spaces seam (`SpacesError`, `_make_client`, `upload_bundle`).
- Modify `src/ml_lab/gpu/lifecycle.py` — add `gpu_run()` (+ imports of the driver, upload, and Spaces-env loader).
- Modify `src/ml_lab/gpu/cli.py` — add the `run` command.
- Modify `scripts/do_gpu.py` — update the entry-point docstring to list `run`/`up`.
- Modify `tests/test_gpu_constants.py` — assert the pinned Spaces constants.
- Modify `tests/test_gpu_env.py` — `load_spaces_env` happy path + names-all-missing.
- Create `tests/test_gpu_spaces.py` — `upload_bundle` key/prefix/error tests, default-factory endpoint test.
- Modify `tests/test_gpu_lifecycle.py` — `gpu_run` ordering, exit code, always-destroy, preflight.
- Modify `tests/test_gpu_cli.py` — `run` exits with `gpu_run`'s code.

---

## Task 1: Pinned Spaces constants

**Files:**
- Modify: `src/ml_lab/gpu/constants.py`
- Test: `tests/test_gpu_constants.py`

- [ ] **Step 1: Write the failing constants test**

Append to `tests/test_gpu_constants.py`:

```python
def test_pinned_spaces_constants_present():
    assert constants.SPACES_REGION == "nyc3"
    assert constants.SPACES_ENDPOINT == "https://nyc3.digitaloceanspaces.com"
    assert constants.SPACES_KEY_PREFIX == "ml-pathway/phase0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gpu_constants.py::test_pinned_spaces_constants_present -v`
Expected: FAIL with `AttributeError: module 'ml_lab.gpu.constants' has no attribute 'SPACES_REGION'`

- [ ] **Step 3: Add the constants**

In `src/ml_lab/gpu/constants.py`, after the `# Toolchain / image / placement` block (right after the `SMOKE_MODEL_ID = "facebook/opt-125m"` line), add:

```python
# DO Spaces (artifact upload) — region pinned like DO_REGION; change deliberately, no fallback.
SPACES_REGION = "nyc3"  # nyc2 has no Spaces; nyc3 is the nearest Spaces region
SPACES_ENDPOINT = f"https://{SPACES_REGION}.digitaloceanspaces.com"
SPACES_KEY_PREFIX = "ml-pathway/phase0"  # s3://<bucket>/ml-pathway/phase0/<run-id>/ (artifacts/README.md)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gpu_constants.py -v`
Expected: PASS (all constants tests green)

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/constants.py tests/test_gpu_constants.py
git commit -m "feat: pin DO Spaces region/endpoint/key-prefix constants"
```

---

## Task 2: `load_spaces_env` in `gpu_env.py`

**Files:**
- Modify: `src/ml_lab/gpu/gpu_env.py`
- Test: `tests/test_gpu_env.py`

- [ ] **Step 1: Write the failing env tests**

Append to `tests/test_gpu_env.py`:

```python
def test_load_spaces_env_happy(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("SPACES_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("SPACES_SECRET_ACCESS_KEY", "SK")
    monkeypatch.setenv("SPACES_BUCKET", "ml-lab-artifacts")
    env = gpu_env.load_spaces_env()
    assert env.access_key == "AK"
    assert env.secret_key == "SK"
    assert env.bucket == "ml-lab-artifacts"


def test_load_spaces_env_missing_lists_every_var(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.delenv("SPACES_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("SPACES_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("SPACES_BUCKET", raising=False)
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_spaces_env()
    msg = str(exc.value)
    assert "SPACES_ACCESS_KEY_ID" in msg
    assert "SPACES_SECRET_ACCESS_KEY" in msg
    assert "SPACES_BUCKET" in msg


def test_load_spaces_env_partial_missing_names_only_the_absent(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("SPACES_ACCESS_KEY_ID", "AK")  # present
    monkeypatch.setenv("SPACES_SECRET_ACCESS_KEY", "SK")  # present
    monkeypatch.delenv("SPACES_BUCKET", raising=False)  # missing
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_spaces_env()
    msg = str(exc.value)
    assert "SPACES_BUCKET" in msg
    assert "SPACES_ACCESS_KEY_ID" not in msg
    assert "SPACES_SECRET_ACCESS_KEY" not in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gpu_env.py -k spaces -v`
Expected: FAIL with `AttributeError: module 'ml_lab.gpu.gpu_env' has no attribute 'load_spaces_env'`

- [ ] **Step 3: Add `SpacesEnv` and `load_spaces_env`**

In `src/ml_lab/gpu/gpu_env.py`, after the `GpuEnv` dataclass, add a second dataclass:

```python
@dataclass(frozen=True)
class SpacesEnv:
    access_key: str  # SPACES_ACCESS_KEY_ID
    secret_key: str  # SPACES_SECRET_ACCESS_KEY
    bucket: str      # SPACES_BUCKET
```

Then, after `load_gpu_env`, add:

```python
def load_spaces_env() -> SpacesEnv:
    """Load .env, then read the three DO Spaces vars. Raise GpuEnvError naming all missing.

    Separate from load_gpu_env: only gpu-run uploads artifacts, so gpu-up must not
    fail on unset Spaces credentials.
    """
    load_dotenv()
    access_key = os.environ.get("SPACES_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("SPACES_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("SPACES_BUCKET", "").strip()

    missing = [
        name
        for name, value in (
            ("SPACES_ACCESS_KEY_ID", access_key),
            ("SPACES_SECRET_ACCESS_KEY", secret_key),
            ("SPACES_BUCKET", bucket),
        )
        if not value
    ]
    if missing:
        raise GpuEnvError(f"missing/empty Spaces env vars: {', '.join(missing)}")

    return SpacesEnv(access_key=access_key, secret_key=secret_key, bucket=bucket)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gpu_env.py -v`
Expected: PASS (all env tests green)

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/gpu_env.py tests/test_gpu_env.py
git commit -m "feat: load_spaces_env + SpacesEnv (fail-loud, separate from GpuEnv)"
```

---

## Task 3: `spaces.upload_bundle` seam

**Files:**
- Create: `src/ml_lab/gpu/spaces.py`
- Test: `tests/test_gpu_spaces.py`

- [ ] **Step 1: Write the failing seam tests**

Create `tests/test_gpu_spaces.py`:

```python
import pytest

from ml_lab.gpu import gpu_env, spaces


class _RecordingClient:
    """Captures upload_file(Filename, Bucket, Key) calls; optionally raises."""

    def __init__(self, boom=False):
        self.calls = []
        self.boom = boom

    def upload_file(self, filename, bucket, key):
        if self.boom:
            raise RuntimeError("boto3 exploded")
        self.calls.append((filename, bucket, key))


def _env():
    return gpu_env.SpacesEnv(access_key="AK", secret_key="SK", bucket="ml-lab-artifacts")


def _bundle(tmp_path):
    d = tmp_path / "R1"
    d.mkdir()
    (d / "benchmark.json").write_text("{}")
    (d / "nvidia-smi.txt").write_text("gpu")
    nested = d / "logs"
    nested.mkdir()
    (nested / "bootstrap.log").write_text("boot")
    return d


def test_upload_bundle_uploads_every_file_with_prefixed_keys(tmp_path):
    client = _RecordingClient()
    uri = spaces.upload_bundle(_bundle(tmp_path), "R1", env=_env(), client=client)
    keys = {key for (_f, _b, key) in client.calls}
    assert keys == {
        "ml-pathway/phase0/R1/benchmark.json",
        "ml-pathway/phase0/R1/nvidia-smi.txt",
        "ml-pathway/phase0/R1/logs/bootstrap.log",  # nested → POSIX relpath
    }
    assert all(bucket == "ml-lab-artifacts" for (_f, bucket, _k) in client.calls)
    assert uri == "s3://ml-lab-artifacts/ml-pathway/phase0/R1/"


def test_upload_bundle_missing_dir_raises_spaces_error(tmp_path):
    with pytest.raises(spaces.SpacesError):
        spaces.upload_bundle(tmp_path / "nope", "R1", env=_env(), client=_RecordingClient())


def test_upload_bundle_empty_dir_raises_spaces_error(tmp_path):
    empty = tmp_path / "R1"
    empty.mkdir()
    with pytest.raises(spaces.SpacesError):
        spaces.upload_bundle(empty, "R1", env=_env(), client=_RecordingClient())


def test_upload_bundle_wraps_client_failure_as_spaces_error(tmp_path):
    with pytest.raises(spaces.SpacesError):
        spaces.upload_bundle(_bundle(tmp_path), "R1", env=_env(), client=_RecordingClient(boom=True))


def test_default_client_uses_spaces_endpoint_and_credentials(monkeypatch):
    import boto3

    seen = {}

    def fake_client(service, **kwargs):
        seen["service"] = service
        seen.update(kwargs)
        return "CLIENT"

    monkeypatch.setattr(boto3, "client", fake_client)
    client = spaces._make_client(_env())
    assert client == "CLIENT"
    assert seen["service"] == "s3"
    assert seen["endpoint_url"] == spaces.SPACES_ENDPOINT
    assert seen["aws_access_key_id"] == "AK"
    assert seen["aws_secret_access_key"] == "SK"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gpu_spaces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_lab.gpu.spaces'`

- [ ] **Step 3: Write the seam**

Create `src/ml_lab/gpu/spaces.py`:

```python
"""The DO Spaces seam: boto3 upload of a pulled artifact bundle, mocked in every test.

Isolated like do_client.py / remote.py — one external dependency (boto3), one public
function, an injectable client factory so no test touches boto3's wire. boto3 is
imported lazily inside the factory so importing this module (and collecting tests that
inject a client) never needs the SDK loaded.
"""

from __future__ import annotations

from pathlib import Path

from ml_lab.gpu.constants import SPACES_ENDPOINT, SPACES_KEY_PREFIX


class SpacesError(RuntimeError):
    """A DO Spaces upload failed, or the bundle to upload was missing/empty."""


def _make_client(env):
    """Build a boto3 S3 client against the DO Spaces endpoint with env credentials."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=env.access_key,
        aws_secret_access_key=env.secret_key,
    )


def upload_bundle(local_dir, run_id, *, env, client=None) -> str:
    """Upload every file under local_dir to s3://<bucket>/<prefix>/<run-id>/<relpath>.

    Returns the s3:// URI of the run's prefix. Raises SpacesError on a missing/empty
    bundle or any upload failure (the boto3 error is chained). `client` is injected in
    every test; in production it defaults to a real DO Spaces S3 client.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise SpacesError(f"bundle directory not found: {local_dir}")
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise SpacesError(f"bundle directory is empty: {local_dir}")

    if client is None:
        client = _make_client(env)

    prefix = f"{SPACES_KEY_PREFIX}/{run_id}"
    for path in files:
        rel = path.relative_to(local_dir).as_posix()  # POSIX keys even on Windows-ish paths
        key = f"{prefix}/{rel}"
        try:
            client.upload_file(str(path), env.bucket, key)
        except Exception as exc:  # boto3 ClientError/BotoCoreError — any failure is a Spaces failure
            raise SpacesError(f"upload of {rel} to {env.bucket} failed: {exc}") from exc

    return f"s3://{env.bucket}/{prefix}/"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gpu_spaces.py -v`
Expected: PASS (5 tests green)

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/spaces.py tests/test_gpu_spaces.py
git commit -m "feat: DO Spaces upload seam (upload_bundle, injectable boto3 client)"
```

---

## Task 4: `gpu_run` lifecycle

**Files:**
- Modify: `src/ml_lab/gpu/lifecycle.py`
- Test: `tests/test_gpu_lifecycle.py`

- [ ] **Step 1: Write the failing lifecycle tests**

Append to `tests/test_gpu_lifecycle.py`. These reuse the existing `_env()` and `_stub_waits_ok(monkeypatch)` helpers already defined in this file.

```python
def _spaces_env():
    return gpu_env.SpacesEnv(access_key="AK", secret_key="SK", bucket="bkt")


def _stub_run_seams(monkeypatch, *, benchmark_code=0, order=None):
    """Stub create + waits + driver + upload + teardown; optionally record call order."""
    order = order if order is not None else []

    def create(ud, **k):
        order.append("create")
        return {"id": 42, "name": "n", "run_id": "r"}

    monkeypatch.setattr(lifecycle, "create_lab_droplet", create)
    monkeypatch.setattr(
        lifecycle, "wait_for_public_ip",
        lambda did, **k: (order.append("ip"), "1.2.3.4")[1],
    )
    monkeypatch.setattr(lifecycle, "wait_for_ssh", lambda ip, **k: order.append("ssh"))
    monkeypatch.setattr(lifecycle, "wait_for_bootstrap", lambda ip, **k: order.append("bootstrap"))
    monkeypatch.setattr(
        lifecycle, "deliver_and_run_benchmark",
        lambda ip, rid, **k: (order.append("benchmark"), benchmark_code)[1],
    )
    monkeypatch.setattr(
        lifecycle, "pull_artifacts",
        lambda ip, rid, **k: (order.append("pull"), "/artifacts/r")[1],
    )
    monkeypatch.setattr(
        lifecycle, "upload_bundle",
        lambda dest, rid, **k: (order.append("upload"), "s3://bkt/ml-pathway/phase0/r/")[1],
    )
    return order


def test_gpu_run_happy_full_order_and_destroys(monkeypatch, capsys):
    destroyed = {}
    order = _stub_run_seams(monkeypatch)
    monkeypatch.setattr(
        lifecycle, "destroy_and_verify",
        lambda did: (order.append("destroy"), destroyed.setdefault("id", did)),
    )
    code = lifecycle.gpu_run(env=_env(), spaces=_spaces_env(), now=1000.0)
    assert code == 0
    assert order == ["create", "ip", "ssh", "bootstrap", "benchmark", "pull", "upload", "destroy"]
    assert destroyed["id"] == 42  # gpu_run ALWAYS destroys, even on success
    assert "s3://bkt/ml-pathway/phase0/r/" in capsys.readouterr().out


def test_gpu_run_returns_benchmark_exit_code_and_still_uploads(monkeypatch):
    order = _stub_run_seams(monkeypatch, benchmark_code=7)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: order.append("destroy"))
    code = lifecycle.gpu_run(env=_env(), spaces=_spaces_env(), now=1000.0)
    assert code == 7  # completed-but-failed benchmark surfaces its code
    assert "pull" in order and "upload" in order  # partial bundle still pulled + uploaded
    assert order[-1] == "destroy"


def test_gpu_run_benchmark_timeout_destroys_and_reraises(monkeypatch):
    destroyed = {}
    _stub_run_seams(monkeypatch)

    def boom(ip, rid, **k):
        raise RemoteError("benchmark timed out")

    monkeypatch.setattr(lifecycle, "deliver_and_run_benchmark", boom)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(RemoteError):
        lifecycle.gpu_run(env=_env(), spaces=_spaces_env(), now=1000.0)
    assert destroyed["id"] == 42


def test_gpu_run_upload_failure_still_destroys(monkeypatch):
    from ml_lab.gpu.spaces import SpacesError

    destroyed = {}
    _stub_run_seams(monkeypatch)

    def boom(dest, rid, **k):
        raise SpacesError("upload failed")

    monkeypatch.setattr(lifecycle, "upload_bundle", boom)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(SpacesError):
        lifecycle.gpu_run(env=_env(), spaces=_spaces_env(), now=1000.0)
    assert destroyed["id"] == 42  # destroy beats artifact preservation


def test_gpu_run_preflight_spaces_failure_creates_nothing(monkeypatch):
    created = {}
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.delenv("SPACES_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("SPACES_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("SPACES_BUCKET", raising=False)
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet",
        lambda ud, **k: created.setdefault("hit", True) or {"id": 42, "name": "n", "run_id": "r"},
    )
    # env supplied, spaces left to load from the (empty) environment → fails before create.
    with pytest.raises(gpu_env.GpuEnvError):
        lifecycle.gpu_run(env=_env(), now=1000.0)
    assert "hit" not in created


def test_gpu_run_teardown_failure_overrides_benchmark_error(monkeypatch):
    _stub_run_seams(monkeypatch)

    def bench_boom(ip, rid, **k):
        raise RemoteError("benchmark timed out")

    def teardown_boom(did):
        raise TeardownError("droplet still present")

    monkeypatch.setattr(lifecycle, "deliver_and_run_benchmark", bench_boom)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", teardown_boom)
    with pytest.raises(TeardownError) as exc:
        lifecycle.gpu_run(env=_env(), spaces=_spaces_env(), now=1000.0)
    assert isinstance(exc.value.__context__, RemoteError)  # original chained
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gpu_lifecycle.py -k gpu_run -v`
Expected: FAIL with `AttributeError: module 'ml_lab.gpu.lifecycle' has no attribute 'gpu_run'`

- [ ] **Step 3: Add driver/upload/env imports to `lifecycle.py`**

In `src/ml_lab/gpu/lifecycle.py`, update the import block. Change:

```python
from ml_lab.gpu.create import create_lab_droplet, generate_run_id
from ml_lab.gpu.gpu_env import load_gpu_env
from ml_lab.gpu.remote import RemoteError, wait_for_bootstrap, wait_for_ssh
from ml_lab.gpu.teardown import destroy_and_verify
```

to:

```python
from ml_lab.gpu.benchmark import deliver_and_run_benchmark, pull_artifacts
from ml_lab.gpu.create import create_lab_droplet, generate_run_id
from ml_lab.gpu.gpu_env import load_gpu_env, load_spaces_env
from ml_lab.gpu.remote import RemoteError, wait_for_bootstrap, wait_for_ssh
from ml_lab.gpu.spaces import upload_bundle
from ml_lab.gpu.teardown import destroy_and_verify
```

(Importing the names into `lifecycle`'s namespace is deliberate — the tests monkeypatch `lifecycle.deliver_and_run_benchmark` / `lifecycle.upload_bundle`, matching how existing tests patch `lifecycle.wait_for_ssh`.)

- [ ] **Step 4: Add `gpu_run`**

At the end of `src/ml_lab/gpu/lifecycle.py`, after `gpu_up`, add:

```python
def gpu_run(*, ttl_seconds=DEFAULT_TTL_SECONDS, env=None, spaces=None, now=None) -> int:
    """Full lifecycle: create, benchmark, pull, upload to Spaces, always destroy.

    Returns the benchmark's exit code. A completed-but-failed benchmark still pulls
    and uploads the partial bundle before the finally destroys. Any raised failure
    (SSH/bootstrap/benchmark timeout, transport, or upload) destroys then propagates —
    destroy beats artifact preservation. Preflight (env + Spaces credentials) runs
    before any droplet is created, so a misconfiguration strands nothing. If
    destroy_and_verify itself fails, that TeardownError is the louder alarm and
    overrides any in-flight error (chained as __context__).
    """
    if env is None:
        env = load_gpu_env()
    if spaces is None:
        spaces = load_spaces_env()
    run_now = now if now is not None else time.time()
    run_id = generate_run_id(run_now)
    user_data = render_cloud_init(
        run_id=run_id, destroy_token=env.destroy_token, ttl_seconds=ttl_seconds
    )

    droplet_id = None
    try:
        result = create_lab_droplet(
            user_data,
            ttl_seconds=ttl_seconds,
            enforce_budget=True,
            ssh_key_ids=env.ssh_key_ids,
            run_id=run_id,
            now=run_now,
        )
        droplet_id = result["id"]
        ip = wait_for_public_ip(droplet_id, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_ssh(ip, key_path=env.ssh_key_path, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_bootstrap(ip, key_path=env.ssh_key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        code = deliver_and_run_benchmark(ip, run_id, key_path=env.ssh_key_path)
        dest = pull_artifacts(ip, run_id, key_path=env.ssh_key_path)
        uri = upload_bundle(dest, run_id, env=spaces)
        print(f"Uploaded benchmark bundle to {uri}", flush=True)
        return code
    finally:
        if droplet_id is not None:
            destroy_and_verify(droplet_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gpu_lifecycle.py -v`
Expected: PASS (all lifecycle tests green — existing `gpu_up` + new `gpu_run`)

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/lifecycle.py tests/test_gpu_lifecycle.py
git commit -m "feat: gpu_run full lifecycle (benchmark→pull→upload, always-destroy finally)"
```

---

## Task 5: `run` command + entry-point docstring

**Files:**
- Modify: `src/ml_lab/gpu/cli.py`
- Modify: `scripts/do_gpu.py`
- Test: `tests/test_gpu_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_gpu_cli.py`:

```python
def test_help_names_run():
    result = CliRunner().invoke(gpu_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output


def test_run_exits_zero_when_gpu_run_returns_zero(monkeypatch):
    monkeypatch.setattr(gpu_cli, "gpu_run", lambda: 0)
    result = CliRunner().invoke(gpu_cli.app, ["run"])
    assert result.exit_code == 0


def test_run_exits_with_gpu_run_nonzero_code(monkeypatch):
    monkeypatch.setattr(gpu_cli, "gpu_run", lambda: 7)
    result = CliRunner().invoke(gpu_cli.app, ["run"])
    assert result.exit_code == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gpu_cli.py -k run -v`
Expected: FAIL — `test_run_*` fail because `gpu_cli` has no `gpu_run` attribute / no `run` command (`Usage` error, exit code 2).

- [ ] **Step 3: Add the `run` command**

In `src/ml_lab/gpu/cli.py`, change the lifecycle import:

```python
from ml_lab.gpu.lifecycle import gpu_up
```

to:

```python
from ml_lab.gpu.lifecycle import gpu_run, gpu_up
```

Then, after `up_command`, add:

```python
@app.command("run")
def run_command() -> None:
    """Full GPU lifecycle: create, benchmark, pull, upload to Spaces, always destroy."""
    raise typer.Exit(code=gpu_run())
```

- [ ] **Step 4: Update the entry-point docstring**

In `scripts/do_gpu.py`, change the module docstring:

```python
"""Entry point: `uv run python scripts/do_gpu.py {audit,down}` (Makefile wraps this)."""
```

to:

```python
"""Entry point: `uv run python scripts/do_gpu.py {run,up,down,audit}` (Makefile wraps this)."""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gpu_cli.py -v`
Expected: PASS (all CLI tests green)

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/cli.py scripts/do_gpu.py tests/test_gpu_cli.py
git commit -m "feat: run command (gpu_run) + do_gpu docstring; make gpu-run live"
```

---

## Task 6: Full-suite green + slice wrap-up

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest`
Expected: PASS — all tests green (no regressions in `gpu_up`, benchmark driver, remote, create, audit, teardown, etc.).

- [ ] **Step 2: Confirm the Makefile target resolves**

Run: `uv run python scripts/do_gpu.py --help`
Expected: the command list includes `run`, `up`, `down`, `audit`. (Do NOT run `make gpu-run` — that would attempt a live DigitalOcean create; live execution is S4.)

- [ ] **Step 3: Commit any incidental cleanup (if needed)**

If Steps 1–2 surfaced nothing to change, skip. Otherwise fix, then:

```bash
git add -A
git commit -m "chore: S3c-2 slice wrap-up"
```

---

## Scope guard (what this slice must NOT do)

- No live DigitalOcean call, live SSH/scp, live boto3 upload, or actual `torch`/`vllm` — S4. Every seam is injected or monkeypatched; no test reaches the network or a GPU.
- No changes to the internals of `create.py`, `remote.py`, `teardown.py`, `benchmark.py`, or `scripts/gpu_benchmark.py` — `gpu_run` only *calls* them.
- No `--ttl-seconds` on `run`, no override of the "refuse if a lab droplet exists" guard, no new GPU class / region / image — Phase 0 boundaries stand.
- No new `make` target and no Makefile edit — `gpu-run` already targets `do_gpu.py run`; this slice makes it functional.
- No new env var in `.env.example` — the three Spaces vars already exist; `SPACES_REGION` is a pinned constant.
```
