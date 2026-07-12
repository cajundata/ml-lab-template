# Phase 0 GPU Remote Seam + `gpu-up` Slice (S3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `gpu-up` path — an SSH/scp subprocess remote seam (`wait_for_ssh`, `wait_for_bootstrap`), env-input loading, public-IP resolution, and the `gpu_up` orchestration whose invariant is "leave the droplet alive only on a fully verified bootstrap; destroy on every other post-create exit" — wired to a Typer `up` command.

**Architecture:** New `remote.py` wraps the system `ssh` binary via subprocess (mirroring `do_client`'s doctl pattern); new `gpu_env.py` reads the three GPU env vars (fail-loud, all at once) after `load_dotenv()`; new `lifecycle.py` holds `wait_for_public_ip` and `gpu_up` (create → resolve IP → wait ssh → wait bootstrap → verified handoff, all inside a `try/except BaseException` that routes any post-create failure through the shared `destroy_and_verify`). Every seam is mocked in tests — no test touches a real droplet or SSH. Live execution is S4.

**Tech Stack:** Python 3.11+, OpenSSH `ssh` (subprocess, mocked in tests), python-dotenv (already a dep), Typer, pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Modify:** `src/ml_lab/gpu/do_client.py` — add `public_ipv4(droplet)` (Task 1); wrap `_run_doctl`'s `CalledProcessError` in `DOClientError` (Task 2).
- **Modify:** `src/ml_lab/gpu/audit.py` — refactor `_to_info` to use `do_client.public_ipv4`, delete `_public_ip` (Task 1).
- **Modify:** `src/ml_lab/gpu/constants.py` — add `REMOTE_POLL_INTERVAL_SECONDS`, `SSH_ATTEMPT_TIMEOUT_SECONDS` (Task 3).
- **Create:** `src/ml_lab/gpu/remote.py` — `SSH_OPTS`, `RemoteError`, `_run_ssh`, `wait_for_ssh` (Task 3), `wait_for_bootstrap` (Task 4).
- **Create:** `src/ml_lab/gpu/gpu_env.py` — `GpuEnv`, `GpuEnvError`, `load_gpu_env` (Task 5).
- **Modify:** `.env.example` — add `DO_SSH_KEY_IDS`, `DO_SSH_KEY_PATH` (Task 5).
- **Create:** `src/ml_lab/gpu/lifecycle.py` — `wait_for_public_ip` (Task 6), `gpu_up` + `_print_handoff` (Task 7).
- **Modify:** `src/ml_lab/gpu/cli.py` — add the `up` command (Task 8).
- **Create tests:** `tests/test_gpu_remote.py` (Tasks 3–4), `tests/test_gpu_env.py` (Task 5), `tests/test_gpu_lifecycle.py` (Tasks 6–7).
- **Modify tests:** `tests/test_gpu_do_client.py` (Tasks 1–2), `tests/test_gpu_cli.py` (Task 8).

Depends on (already committed, do not modify): `create.create_lab_droplet` / `create.generate_run_id` / `create.LabDropletExistsError`, `cloud_init.render_cloud_init`, `teardown.destroy_and_verify`, `do_client.get_droplet` / `do_client.DOClientError`, and the timing constants (`DEFAULT_TTL_SECONDS`, `SSH_TIMEOUT_SECONDS`, `BOOTSTRAP_TIMEOUT_SECONDS`, `SELF_DESTRUCT_RETRY_SECONDS`, `DO_REGION`, `DO_SIZE_SLUG`). pytest is configured with `pythonpath = ["src"]`. `scp`, `gpu_benchmark.py`, Spaces upload, and `gpu-run` are OUT of scope (S3c); live SSH/droplet is S4.

---

### Task 1: `do_client.public_ipv4` + audit refactor

**Files:**
- Modify: `src/ml_lab/gpu/do_client.py`
- Modify: `src/ml_lab/gpu/audit.py`
- Test: `tests/test_gpu_do_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_do_client.py`:

```python
def test_public_ipv4_extracts_public_address():
    droplet = {
        "networks": {
            "v4": [
                {"type": "private", "ip_address": "10.0.0.1"},
                {"type": "public", "ip_address": "1.2.3.4"},
            ]
        }
    }
    assert do_client.public_ipv4(droplet) == "1.2.3.4"


def test_public_ipv4_none_when_no_public():
    private_only = {"networks": {"v4": [{"type": "private", "ip_address": "10.0.0.1"}]}}
    assert do_client.public_ipv4(private_only) is None
    assert do_client.public_ipv4({}) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_do_client.py -k public_ipv4 -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.do_client' has no attribute 'public_ipv4'`.

- [ ] **Step 3: Add `public_ipv4` to `do_client`**

Append to `src/ml_lab/gpu/do_client.py`:

```python
def public_ipv4(droplet: dict) -> str | None:
    """Return the droplet's public IPv4 address, or None if it has none yet."""
    for net in ((droplet.get("networks") or {}).get("v4") or []):
        if net.get("type") == "public":
            return net.get("ip_address")
    return None
```

- [ ] **Step 4: Refactor `audit` to use it (DRY — one implementation)**

In `src/ml_lab/gpu/audit.py`, delete the `_public_ip` helper (the `def _public_ip(d): ...` block, lines ~48–52):

```python
def _public_ip(d: dict) -> str:
    for net in (d.get("networks") or {}).get("v4") or []:
        if net.get("type") == "public":
            return net.get("ip_address", "")
    return ""
```

Then, in `_to_info`, change the `public_ip=` argument from:

```python
        public_ip=_public_ip(d),
```

to:

```python
        public_ip=do_client.public_ipv4(d) or "",
```

(`audit` already imports `do_client`. The `or ""` preserves the existing behavior of an empty string when there is no public IP.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_do_client.py tests/test_gpu_audit.py -v`
Expected: PASS — the two new `public_ipv4` tests plus all existing `do_client` and `audit` tests (the refactor is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/do_client.py src/ml_lab/gpu/audit.py tests/test_gpu_do_client.py
git commit -m "feat: do_client.public_ipv4 + audit reuses it (drop duplicate _public_ip)"
```

---

### Task 2: `_run_doctl` wraps `CalledProcessError` in `DOClientError`

**Files:**
- Modify: `src/ml_lab/gpu/do_client.py`
- Test: `tests/test_gpu_do_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_do_client.py`:

```python
def test_run_doctl_wraps_calledprocesserror(monkeypatch):
    def boom(*a, **k):
        raise do_client.subprocess.CalledProcessError(1, "doctl", stderr="unauthorized")

    monkeypatch.setattr(do_client.subprocess, "run", boom)
    with pytest.raises(DOClientError):
        do_client._run_doctl(["compute", "droplet", "list"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_gpu_do_client.py -k run_doctl_wraps -v`
Expected: FAIL — the raw `CalledProcessError` propagates instead of a `DOClientError`.

- [ ] **Step 3: Wrap the error in `_run_doctl`**

In `src/ml_lab/gpu/do_client.py`, change `_run_doctl` from:

```python
def _run_doctl(args: list[str]) -> list[dict]:
    result = subprocess.run(
        ["doctl", *args, "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    text = (result.stdout or "").strip()
    if not text or text == "null":
        return []
    return json.loads(text)
```

to:

```python
def _run_doctl(args: list[str]) -> list[dict]:
    try:
        result = subprocess.run(
            ["doctl", *args, "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DOClientError(
            f"doctl {' '.join(args)} failed: {(exc.stderr or '').strip()}"
        ) from exc
    text = (result.stdout or "").strip()
    if not text or text == "null":
        return []
    return json.loads(text)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_do_client.py -v`
Expected: PASS — the new test plus all existing `do_client` tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/do_client.py tests/test_gpu_do_client.py
git commit -m "fix: _run_doctl wraps CalledProcessError in DOClientError (clean unauth failure)"
```

---

### Task 3: `remote.py` — `_run_ssh` + `wait_for_ssh`

**Files:**
- Modify: `src/ml_lab/gpu/constants.py`
- Create: `src/ml_lab/gpu/remote.py`
- Test: `tests/test_gpu_remote.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_remote.py`:

```python
import subprocess
import types

import pytest

from ml_lab.gpu import remote
from ml_lab.gpu.remote import RemoteError


def _completed(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeClock:
    """Deterministic monotonic clock: sleep() advances virtual time."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_wait_for_ssh_returns_once_reachable(monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, *a, **k):
        calls["n"] += 1
        return _completed(returncode=0 if calls["n"] >= 3 else 255)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    clock = _FakeClock()
    remote.wait_for_ssh("1.2.3.4", key_path="/key", timeout=1000, now=clock.now, sleep=clock.sleep)
    assert calls["n"] == 3  # two failures, third attempt succeeds


def test_wait_for_ssh_times_out(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(returncode=255))
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_ssh("1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep)


def test_wait_for_ssh_treats_attempt_timeout_as_not_ready(monkeypatch):
    def timeout_then_never(argv, *a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(remote.subprocess, "run", timeout_then_never)
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_ssh("1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep)


def test_run_ssh_builds_argv(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(returncode=0)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    remote._run_ssh("1.2.3.4", ["true"], key_path="/key", timeout=30)
    argv = captured["argv"]
    assert argv[0] == "ssh"
    assert "-i" in argv and argv[argv.index("-i") + 1] == "/key"
    assert "root@1.2.3.4" in argv
    assert argv[-1] == "true"
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=accept-new" in argv
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_remote.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.remote'`.

- [ ] **Step 3: Add the polling constants**

In `src/ml_lab/gpu/constants.py`, add these two lines to the "Lifecycle timing (seconds)" block (just after `DESTROY_POLL_INTERVAL_SECONDS`):

```python
REMOTE_POLL_INTERVAL_SECONDS = 15  # cadence for ssh-readiness / bootstrap-marker polling
SSH_ATTEMPT_TIMEOUT_SECONDS = 30  # subprocess timeout bounding a single ssh attempt
```

- [ ] **Step 4: Create `remote.py` with `_run_ssh` and `wait_for_ssh`**

Create `src/ml_lab/gpu/remote.py`:

```python
"""The SSH seam: hardened, non-interactive ssh over subprocess, mocked in every test.

Mirrors do_client's doctl-subprocess pattern. wait_for_ssh / wait_for_bootstrap poll
on an injectable clock so deadline tests are instant. scp is deferred to S3c.
"""

from __future__ import annotations

import json
import subprocess
import time

from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    REMOTE_POLL_INTERVAL_SECONDS,
    SSH_ATTEMPT_TIMEOUT_SECONDS,
    SSH_TIMEOUT_SECONDS,
)

# Hardened, non-interactive: never prompt, bound each connect, don't pollute known_hosts.
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


class RemoteError(RuntimeError):
    """SSH was not reachable, or bootstrap was not verified, before the deadline."""


def _run_ssh(host, argv, *, key_path, timeout):
    """Run `ssh -i <key> <opts> root@<host> <argv...>`; return the CompletedProcess."""
    return subprocess.run(
        ["ssh", "-i", key_path, *SSH_OPTS, f"root@{host}", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def wait_for_ssh(
    host,
    *,
    key_path,
    timeout=SSH_TIMEOUT_SECONDS,
    interval=REMOTE_POLL_INTERVAL_SECONDS,
    now=None,
    sleep=None,
):
    """Poll `ssh ... true` until it exits 0; raise RemoteError at the deadline."""
    now = now or time.monotonic
    sleep = sleep or time.sleep
    deadline = now() + timeout
    while True:
        try:
            ok = _run_ssh(host, ["true"], key_path=key_path,
                          timeout=SSH_ATTEMPT_TIMEOUT_SECONDS).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        if ok:
            return
        if now() >= deadline:
            raise RemoteError(f"ssh to {host} not reachable after {timeout}s")
        sleep(interval)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_remote.py -v`
Expected: PASS — the four `wait_for_ssh`/`_run_ssh` tests green.

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/constants.py src/ml_lab/gpu/remote.py tests/test_gpu_remote.py
git commit -m "feat: remote.py ssh seam + wait_for_ssh (hardened, injectable-clock poll)"
```

---

### Task 4: `remote.wait_for_bootstrap`

**Files:**
- Modify: `src/ml_lab/gpu/remote.py`
- Test: `tests/test_gpu_remote.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_remote.py`:

```python
READY_JSON = '{"ready":true,"self_destruct_timer_active":true}'


def test_wait_for_bootstrap_returns_marker_when_both_flags_true(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(stdout=READY_JSON))
    clock = _FakeClock()
    marker = remote.wait_for_bootstrap(
        "1.2.3.4", key_path="/key", timeout=1000, now=clock.now, sleep=clock.sleep
    )
    assert marker == {"ready": True, "self_destruct_timer_active": True}


def test_wait_for_bootstrap_polls_until_ready(monkeypatch):
    seq = iter(
        [
            _completed(returncode=1, stdout=""),  # file not there yet
            _completed(stdout='{"ready":true,"self_destruct_timer_active":false}'),  # timer not up
            _completed(stdout=READY_JSON),  # ready
        ]
    )
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: next(seq))
    clock = _FakeClock()
    marker = remote.wait_for_bootstrap(
        "1.2.3.4", key_path="/key", timeout=1000, now=clock.now, sleep=clock.sleep
    )
    assert marker["self_destruct_timer_active"] is True


def test_wait_for_bootstrap_times_out_when_timer_never_active(monkeypatch):
    stale = '{"ready":true,"self_destruct_timer_active":false}'
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(stdout=stale))
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_bootstrap(
            "1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep
        )


def test_wait_for_bootstrap_ignores_malformed_json(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(stdout="not json"))
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_bootstrap(
            "1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_remote.py -k bootstrap -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.remote' has no attribute 'wait_for_bootstrap'`.

- [ ] **Step 3: Add `wait_for_bootstrap`**

Append to `src/ml_lab/gpu/remote.py`:

```python
def wait_for_bootstrap(
    host,
    *,
    key_path,
    timeout=BOOTSTRAP_TIMEOUT_SECONDS,
    interval=REMOTE_POLL_INTERVAL_SECONDS,
    now=None,
    sleep=None,
):
    """Poll /opt/ml-lab/bootstrap-ready.json until ready AND timer active; else RemoteError.

    Parses the marker as JSON (not string-match), so both flags must be exactly True.
    """
    now = now or time.monotonic
    sleep = sleep or time.sleep
    deadline = now() + timeout
    while True:
        marker = None
        try:
            result = _run_ssh(
                host,
                ["cat", "/opt/ml-lab/bootstrap-ready.json"],
                key_path=key_path,
                timeout=SSH_ATTEMPT_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                try:
                    parsed = json.loads(result.stdout)
                except json.JSONDecodeError:
                    parsed = None
                if (
                    isinstance(parsed, dict)
                    and parsed.get("ready") is True
                    and parsed.get("self_destruct_timer_active") is True
                ):
                    marker = parsed
        except subprocess.TimeoutExpired:
            marker = None
        if marker is not None:
            return marker
        if now() >= deadline:
            raise RemoteError(f"bootstrap not verified on {host} after {timeout}s")
        sleep(interval)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_remote.py -v`
Expected: PASS — the four bootstrap tests plus the Task 3 ssh tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/remote.py tests/test_gpu_remote.py
git commit -m "feat: remote.wait_for_bootstrap (JSON marker, both flags, injectable-clock poll)"
```

---

### Task 5: `gpu_env.py` — env inputs + `.env.example`

**Files:**
- Create: `src/ml_lab/gpu/gpu_env.py`
- Modify: `.env.example`
- Test: `tests/test_gpu_env.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_env.py`:

```python
import pytest

from ml_lab.gpu import gpu_env


def test_load_gpu_env_happy(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")
    monkeypatch.setenv("DO_SSH_KEY_IDS", "a1:b2, c3:d4")
    monkeypatch.setenv("DO_SSH_KEY_PATH", "/home/me/.ssh/id_ed25519")
    env = gpu_env.load_gpu_env()
    assert env.destroy_token == "tok"
    assert env.ssh_key_ids == ["a1:b2", "c3:d4"]  # comma-split, whitespace trimmed
    assert env.ssh_key_path == "/home/me/.ssh/id_ed25519"


def test_load_gpu_env_missing_lists_every_var(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.delenv("DO_DROPLET_DESTROY_TOKEN", raising=False)
    monkeypatch.delenv("DO_SSH_KEY_IDS", raising=False)
    monkeypatch.delenv("DO_SSH_KEY_PATH", raising=False)
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_gpu_env()
    msg = str(exc.value)
    assert "DO_DROPLET_DESTROY_TOKEN" in msg
    assert "DO_SSH_KEY_IDS" in msg
    assert "DO_SSH_KEY_PATH" in msg
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.gpu_env'`.

- [ ] **Step 3: Create `gpu_env.py`**

Create `src/ml_lab/gpu/gpu_env.py`:

```python
"""Read and validate the operator's local GPU env vars (fail loud, all at once)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class GpuEnvError(RuntimeError):
    """One or more required GPU env vars are missing or empty."""


@dataclass(frozen=True)
class GpuEnv:
    destroy_token: str      # DO_DROPLET_DESTROY_TOKEN
    ssh_key_ids: list[str]  # DO_SSH_KEY_IDS, comma-split
    ssh_key_path: str       # DO_SSH_KEY_PATH


def load_gpu_env() -> GpuEnv:
    """Load .env, then read the three GPU vars. Raise GpuEnvError naming all missing ones."""
    load_dotenv()
    destroy_token = os.environ.get("DO_DROPLET_DESTROY_TOKEN", "").strip()
    ssh_key_ids_raw = os.environ.get("DO_SSH_KEY_IDS", "").strip()
    ssh_key_path = os.environ.get("DO_SSH_KEY_PATH", "").strip()

    missing = [
        name
        for name, value in (
            ("DO_DROPLET_DESTROY_TOKEN", destroy_token),
            ("DO_SSH_KEY_IDS", ssh_key_ids_raw),
            ("DO_SSH_KEY_PATH", ssh_key_path),
        )
        if not value
    ]
    if missing:
        raise GpuEnvError(f"missing/empty GPU env vars: {', '.join(missing)}")

    ssh_key_ids = [k.strip() for k in ssh_key_ids_raw.split(",") if k.strip()]
    return GpuEnv(
        destroy_token=destroy_token, ssh_key_ids=ssh_key_ids, ssh_key_path=ssh_key_path
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_env.py -v`
Expected: PASS — both env tests green.

- [ ] **Step 5: Add the two SSH vars to `.env.example`**

In `.env.example`, add these two lines after the `DO_DROPLET_DESTROY_TOKEN=` line:

```bash
DO_SSH_KEY_IDS=
DO_SSH_KEY_PATH=
```

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/gpu_env.py tests/test_gpu_env.py .env.example
git commit -m "feat: gpu_env.load_gpu_env (dotenv + fail-loud-all-at-once) + .env.example keys"
```

---

### Task 6: `lifecycle.wait_for_public_ip`

**Files:**
- Create: `src/ml_lab/gpu/lifecycle.py`
- Test: `tests/test_gpu_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_lifecycle.py`:

```python
import pytest

from ml_lab.gpu import do_client, lifecycle
from ml_lab.gpu.remote import RemoteError


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_wait_for_public_ip_returns_once_networks_populate(monkeypatch):
    seq = iter(
        [
            {"id": 1},  # no networks yet
            {"id": 1, "networks": {"v4": [{"type": "public", "ip_address": "5.6.7.8"}]}},
        ]
    )
    monkeypatch.setattr(lifecycle.do_client, "get_droplet", lambda i: next(seq))
    clock = _FakeClock()
    ip = lifecycle.wait_for_public_ip(1, timeout=1000, now=clock.now, sleep=clock.sleep)
    assert ip == "5.6.7.8"


def test_wait_for_public_ip_times_out(monkeypatch):
    monkeypatch.setattr(lifecycle.do_client, "get_droplet", lambda i: {"id": 1})  # never gets IP
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        lifecycle.wait_for_public_ip(1, timeout=30, now=clock.now, sleep=clock.sleep)


def test_wait_for_public_ip_tolerates_transient_do_error(monkeypatch):
    seq = iter([1, 2])  # first call raises, second returns a droplet with an IP

    def flaky(i):
        if next(seq) == 1:
            raise do_client.DOClientError("transient")
        return {"networks": {"v4": [{"type": "public", "ip_address": "9.9.9.9"}]}}

    monkeypatch.setattr(lifecycle.do_client, "get_droplet", flaky)
    clock = _FakeClock()
    ip = lifecycle.wait_for_public_ip(1, timeout=1000, now=clock.now, sleep=clock.sleep)
    assert ip == "9.9.9.9"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_lifecycle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.lifecycle'`.

- [ ] **Step 3: Create `lifecycle.py` with `wait_for_public_ip`**

Create `src/ml_lab/gpu/lifecycle.py`:

```python
"""The gpu-up orchestration: create, verify readiness, hand off — or tear down.

Invariant: once a droplet exists, every non-clean exit routes through
destroy_and_verify; a fully verified success leaves it alive (the debug path).
All seams (do_client, remote, create, teardown) are mocked in tests.
"""

from __future__ import annotations

import time

from ml_lab.gpu import do_client
from ml_lab.gpu.constants import REMOTE_POLL_INTERVAL_SECONDS, SSH_TIMEOUT_SECONDS
from ml_lab.gpu.remote import RemoteError


def wait_for_public_ip(
    droplet_id,
    *,
    timeout=SSH_TIMEOUT_SECONDS,
    interval=REMOTE_POLL_INTERVAL_SECONDS,
    now=None,
    sleep=None,
):
    """Poll get_droplet until a public IPv4 appears; raise RemoteError at the deadline."""
    now = now or time.monotonic
    sleep = sleep or time.sleep
    deadline = now() + timeout
    while True:
        try:
            droplet = do_client.get_droplet(droplet_id)
        except do_client.DOClientError:
            droplet = None  # transient; the poll, not one call, is the truth
        ip = do_client.public_ipv4(droplet) if droplet else None
        if ip:
            return ip
        if now() >= deadline:
            raise RemoteError(f"no public IP for droplet {droplet_id} after {timeout}s")
        sleep(interval)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_lifecycle.py -v`
Expected: PASS — the three `wait_for_public_ip` tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/lifecycle.py tests/test_gpu_lifecycle.py
git commit -m "feat: lifecycle.wait_for_public_ip (deadline poll, transient-tolerant)"
```

---

### Task 7: `lifecycle.gpu_up` — orchestration + teardown discipline

**Files:**
- Modify: `src/ml_lab/gpu/lifecycle.py`
- Test: `tests/test_gpu_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

First, add these two imports to the **top** of `tests/test_gpu_lifecycle.py` (with the existing imports, not mid-file):

```python
from ml_lab.gpu import gpu_env
from ml_lab.gpu.create import LabDropletExistsError
```

Then append the tests to `tests/test_gpu_lifecycle.py`:

```python
def _env():
    return gpu_env.GpuEnv(destroy_token="tok", ssh_key_ids=["k1"], ssh_key_path="/key")


def _stub_waits_ok(monkeypatch):
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")
    monkeypatch.setattr(lifecycle, "wait_for_ssh", lambda ip, **k: None)
    monkeypatch.setattr(
        lifecycle,
        "wait_for_bootstrap",
        lambda ip, **k: {"ready": True, "self_destruct_timer_active": True},
    )


def test_gpu_up_happy_leaves_droplet_alive(monkeypatch, capsys):
    destroyed = {}
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet",
        lambda ud, **k: {"id": 42, "name": "ml-lab-gpu-phase0-r", "run_id": "r"},
    )
    _stub_waits_ok(monkeypatch)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    result = lifecycle.gpu_up(env=_env(), now=1000.0)
    assert result["id"] == 42 and result["ip"] == "1.2.3.4"
    assert "id" not in destroyed  # alive on verified success
    out = capsys.readouterr().out
    assert "verified active" in out
    assert "make gpu-down DROPLET_ID=42" in out


def test_gpu_up_ssh_timeout_destroys_and_reraises(monkeypatch):
    destroyed = {}
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet", lambda ud, **k: {"id": 42, "name": "n", "run_id": "r"}
    )
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")

    def boom(ip, **k):
        raise RemoteError("unreachable")

    monkeypatch.setattr(lifecycle, "wait_for_ssh", boom)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(RemoteError):
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert destroyed["id"] == 42


def test_gpu_up_keyboardinterrupt_after_create_destroys(monkeypatch):
    destroyed = {}
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet", lambda ud, **k: {"id": 42, "name": "n", "run_id": "r"}
    )
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")

    def interrupt(ip, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(lifecycle, "wait_for_ssh", interrupt)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(KeyboardInterrupt):
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert destroyed["id"] == 42


def test_gpu_up_preflight_failure_does_not_destroy(monkeypatch):
    destroyed = {}

    def refuse(ud, **k):
        raise LabDropletExistsError("exists")

    monkeypatch.setattr(lifecycle, "create_lab_droplet", refuse)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(LabDropletExistsError):
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert "id" not in destroyed  # no droplet existed → nothing to destroy


def test_gpu_up_threads_ttl_and_enforce_budget(monkeypatch):
    seen = {}

    def capture(ud, **k):
        seen.update(k)
        return {"id": 42, "name": "n", "run_id": "r"}

    monkeypatch.setattr(lifecycle, "create_lab_droplet", capture)
    _stub_waits_ok(monkeypatch)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: None)
    lifecycle.gpu_up(env=_env(), ttl_seconds=900, enforce_budget=False, now=1000.0)
    assert seen["ttl_seconds"] == 900
    assert seen["enforce_budget"] is False
    assert seen["ssh_key_ids"] == ["k1"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_lifecycle.py -k gpu_up -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.lifecycle' has no attribute 'gpu_up'`.

- [ ] **Step 3: Add imports, `_print_handoff`, and `gpu_up`**

In `src/ml_lab/gpu/lifecycle.py`, extend the imports at the top — add the create/cloud-init/teardown/env seams and the extra constants:

```python
from __future__ import annotations

import time

from ml_lab.gpu import do_client
from ml_lab.gpu.cloud_init import render_cloud_init
from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    DEFAULT_TTL_SECONDS,
    DO_REGION,
    DO_SIZE_SLUG,
    REMOTE_POLL_INTERVAL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SSH_TIMEOUT_SECONDS,
)
from ml_lab.gpu.create import create_lab_droplet, generate_run_id
from ml_lab.gpu.gpu_env import load_gpu_env
from ml_lab.gpu.remote import RemoteError, wait_for_bootstrap, wait_for_ssh
from ml_lab.gpu.teardown import destroy_and_verify
```

Then append `_print_handoff` and `gpu_up`:

```python
def _print_handoff(result, ip, ttl_seconds):
    print(
        "Created GPU droplet:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  region: {DO_REGION}\n"
        f"  size: {DO_SIZE_SLUG}\n"
        f"  public-ip: {ip}\n"
        f"  ttl: {ttl_seconds} seconds\n"
        f"  self-destruct retry: {SELF_DESTRUCT_RETRY_SECONDS} seconds\n"
        f"  destroy: make gpu-down DROPLET_ID={result['id']}\n"
        "Self-destruct timer has been verified active on the droplet.\n"
        "Do not power off this droplet. Destroy it.",
        flush=True,
    )


def gpu_up(*, ttl_seconds=DEFAULT_TTL_SECONDS, enforce_budget=True, env=None, now=None) -> dict:
    """Create a GPU droplet, verify SSH + bootstrap, hand off — or tear down on any failure.

    On a fully verified success the droplet is LEFT ALIVE (debug path). On any
    post-create exception (including KeyboardInterrupt) it is destroyed and the
    error re-raised. A preflight failure creates no droplet, so nothing is destroyed.
    """
    if env is None:
        env = load_gpu_env()
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
            enforce_budget=enforce_budget,
            ssh_key_ids=env.ssh_key_ids,
            run_id=run_id,
            now=run_now,
        )
        droplet_id = result["id"]
        ip = wait_for_public_ip(droplet_id, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_ssh(ip, key_path=env.ssh_key_path, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_bootstrap(ip, key_path=env.ssh_key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        _print_handoff(result, ip, ttl_seconds)
        return {**result, "ip": ip}
    except BaseException:
        if droplet_id is not None:
            destroy_and_verify(droplet_id)
        raise
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_lifecycle.py -v`
Expected: PASS — the five `gpu_up` tests plus the three `wait_for_public_ip` tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/lifecycle.py tests/test_gpu_lifecycle.py
git commit -m "feat: lifecycle.gpu_up (verified handoff or destroy-on-any-failure)"
```

---

### Task 8: `up` CLI command + wiring

**Files:**
- Modify: `src/ml_lab/gpu/cli.py`
- Test: `tests/test_gpu_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_cli.py`:

```python
def test_help_names_up():
    result = CliRunner().invoke(gpu_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "up" in result.output


def test_up_default_calls_gpu_up_with_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(gpu_cli, "gpu_up", lambda **k: seen.setdefault("kwargs", k))
    result = CliRunner().invoke(gpu_cli.app, ["up"])
    assert result.exit_code == 0
    assert seen["kwargs"] == {}  # no --ttl-seconds → gpu_up() defaults


def test_up_short_ttl_bypasses_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr(gpu_cli, "gpu_up", lambda **k: seen.setdefault("kwargs", k))
    result = CliRunner().invoke(gpu_cli.app, ["up", "--ttl-seconds", "900"])
    assert result.exit_code == 0
    assert seen["kwargs"] == {"ttl_seconds": 900, "enforce_budget": False}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_cli.py -k up -v`
Expected: FAIL — the `up` command does not exist (`--help` lacks it; invoking `up` errors).

- [ ] **Step 3: Add the `up` command**

In `src/ml_lab/gpu/cli.py`, add the import of `gpu_up` (next to the existing `destroy_and_verify` import):

```python
from ml_lab.gpu.lifecycle import gpu_up
from ml_lab.gpu.teardown import destroy_and_verify
```

Then append the command:

```python
@app.command("up")
def up_command(
    ttl_seconds: int = typer.Option(
        None,
        "--ttl-seconds",
        help="Short-fuse self-destruct test; bypasses the benchmark-budget check.",
    ),
) -> None:
    """Create a GPU droplet and hand it off once bootstrap is verified (leaves it alive)."""
    if ttl_seconds is None:
        gpu_up()
    else:
        gpu_up(ttl_seconds=ttl_seconds, enforce_budget=False)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_cli.py -v`
Expected: PASS — the three new `up` tests plus the existing `audit`/`down` tests.

- [ ] **Step 5: Confirm the full suite stays green**

Run: `uv run pytest -q`
Expected: PASS — the existing 113 S1/S2/S3a tests plus all new S3b tests.

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/cli.py tests/test_gpu_cli.py
git commit -m "feat: gpu-up CLI command (default + short-ttl self-destruct mode)"
```

---

## Definition of Done

- `uv run pytest` is fully green (existing 113 tests plus the new S3b tests).
- `do_client` exposes `public_ipv4(droplet)`; `audit` reuses it (no duplicate `_public_ip`); `_run_doctl` raises `DOClientError` (not raw `CalledProcessError`) on a non-zero doctl exit.
- `remote.py` exposes `wait_for_ssh` and `wait_for_bootstrap` over a hardened, non-interactive `ssh` subprocess seam with an injectable clock; `RemoteError` on deadline. `scp` is deliberately absent (S3c).
- `gpu_env.load_gpu_env` loads `.env` and returns a `GpuEnv` (token + comma-split key ids + key path), raising `GpuEnvError` that names every missing/empty var at once; `.env.example` lists `DO_SSH_KEY_IDS` and `DO_SSH_KEY_PATH`.
- `lifecycle.gpu_up` runs create → resolve IP → wait ssh → wait bootstrap → verified handoff, threading one `run_id` into both `render_cloud_init` and `create_lab_droplet`; a fully verified success leaves the droplet alive; any post-create exception (including `KeyboardInterrupt`) routes through `destroy_and_verify` and re-raises; a preflight failure destroys nothing.
- `make gpu-up` and `make gpu-up`-with-`--ttl-seconds` invoke `gpu_up` with the right kwargs.
- No S3b test reaches the network, a real droplet, or a real SSH connection (every seam mocked).
- Out of scope and untouched: `_run_scp`, `gpu_benchmark.py`, artifact pull, Spaces upload, `gpu-run` (S3c); live SSH/droplet + failure-path gates (S4).
