# Phase 0 GPU Safety Spine Slice (S1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU safety spine — constants, a mockable `do_client` seam, `audit`, and the SIGINT-hardened `destroy_and_verify` — backing working `make gpu-audit` / `make gpu-down`, before any code in the repo can create a billable droplet.

**Architecture:** New `src/ml_lab/gpu/` package. A thin `do_client` module hides the DigitalOcean transport (doctl for list/get/destroy, REST for the destroy-token probe) behind intent-level functions. `audit` and `teardown` are pure logic over that seam, so every test mocks `do_client` and never touches a real droplet. A Typer `app` in `ml_lab/gpu/cli.py` exposes `audit` / `down`; `scripts/do_gpu.py` is a 2-line shim so the master-plan command `uv run python scripts/do_gpu.py audit` works.

**Tech Stack:** Python 3.11+, Typer, `requests` (new, explicit dep), `doctl` (subprocess), pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Create:** `src/ml_lab/gpu/__init__.py` — empty package marker (Task 1).
- **Create:** `src/ml_lab/gpu/constants.py` — pinned constants + `validate_timeout_budget` (Task 1).
- **Create:** `src/ml_lab/gpu/do_client.py` — the DO seam: `list_lab_droplets`, `get_droplet`, `destroy_droplet`, `probe_destroy_token`, related-resource lists, `DOClientError` (Task 2).
- **Create:** `src/ml_lab/gpu/audit.py` — `DropletInfo`, `AuditReport`, `collect_audit`, `format_report`, `is_clean` (Task 3).
- **Create:** `src/ml_lab/gpu/teardown.py` — `destroy_and_verify`, `TeardownError` (Task 4).
- **Create:** `src/ml_lab/gpu/cli.py` — Typer `app` with `audit` / `down` (Task 5).
- **Create:** `scripts/do_gpu.py` — shim importing `app` (Task 5).
- **Modify:** `pyproject.toml` — add `requests` (Task 2, via `uv add`).
- **Create tests:** `tests/test_gpu_constants.py` (T1), `tests/test_gpu_do_client.py` (T2), `tests/test_gpu_audit.py` (T3), `tests/test_gpu_teardown.py` (T4), `tests/test_gpu_cli.py` (T5).

pytest is configured with `pythonpath = ["src"]` and the package installs editable via hatchling, so `import ml_lab.gpu...` works in tests and subprocesses. `create_droplet`, `run`/`up`, cloud-init, and live gates are OUT of scope (later slices). `DO_IMAGE_SLUG` stays `None` (only `create` needs it).

---

### Task 1: Constants module + timeout-budget validator

**Files:**
- Create: `src/ml_lab/gpu/__init__.py`
- Create: `src/ml_lab/gpu/constants.py`
- Test: `tests/test_gpu_constants.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_constants.py`:

```python
import pytest

from ml_lab.gpu import constants
from ml_lab.gpu.constants import validate_timeout_budget


def test_default_ttl_fits_the_timeout_budget():
    # SSH(600) + BOOTSTRAP(1800) + BENCHMARK(1800) = 4200 < 7200
    validate_timeout_budget(constants.DEFAULT_TTL_SECONDS)  # no raise


def test_short_ttl_below_budget_raises():
    with pytest.raises(ValueError):
        validate_timeout_budget(900)


def test_pinned_constants_present():
    assert constants.DO_REGION == "atl1"
    assert constants.DO_SIZE_SLUG == "gpu-rtx4000x1-20gb"
    assert constants.DROPLET_NAME_PREFIX == "ml-lab-gpu-"
    assert constants.DESTROY_POLL_INTERVAL_SECONDS == 10
    assert "ml-lab" in constants.BASE_TAGS
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_gpu_constants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu'`.

- [ ] **Step 3: Create the package and constants**

Create `src/ml_lab/gpu/__init__.py` (empty file).

Create `src/ml_lab/gpu/constants.py`:

```python
"""Pinned DigitalOcean GPU-lab constants (master plan §3, no fallback)."""

# Toolchain / image / placement
DO_REGION = "atl1"
DO_SIZE_SLUG = "gpu-rtx4000x1-20gb"
DO_GPU_RUNG = "RTX 4000 Ada"
DO_IMAGE_SLUG = None  # DO NVIDIA AI/ML-ready GPU image; resolved in S2 (create-only)
SMOKE_MODEL_ID = "facebook/opt-125m"

# Lifecycle timing (seconds)
DEFAULT_TTL_SECONDS = 7200
SSH_TIMEOUT_SECONDS = 600
BOOTSTRAP_TIMEOUT_SECONDS = 1800
BENCHMARK_TIMEOUT_SECONDS = 1800
DESTROY_POLL_TIMEOUT_SECONDS = 600
SELF_DESTRUCT_RETRY_SECONDS = 300
DESTROY_POLL_INTERVAL_SECONDS = 10  # cadence for absence polling (added; not in plan)

# Identity / audit matching
DROPLET_NAME_PREFIX = "ml-lab-gpu-"
NAME_FORMAT = "ml-lab-gpu-phase0-{run_id}"
BASE_TAGS = ["ml-lab", "ml-pathway", "phase-0", "owner-weldon"]


def validate_timeout_budget(ttl_seconds: int) -> None:
    """Raise ValueError unless the SSH+bootstrap+benchmark budget fits under ttl_seconds."""
    budget = SSH_TIMEOUT_SECONDS + BOOTSTRAP_TIMEOUT_SECONDS + BENCHMARK_TIMEOUT_SECONDS
    if budget >= ttl_seconds:
        raise ValueError(
            f"timeout budget {budget}s does not fit under TTL {ttl_seconds}s"
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_gpu_constants.py -v`
Expected: PASS — all three tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/__init__.py src/ml_lab/gpu/constants.py tests/test_gpu_constants.py
git commit -m "feat: GPU-lab constants + timeout-budget validator"
```

---

### Task 2: `do_client` seam (+ explicit `requests` dep)

**Files:**
- Create: `src/ml_lab/gpu/do_client.py`
- Modify: `pyproject.toml` (via `uv add requests`)
- Test: `tests/test_gpu_do_client.py`

- [ ] **Step 1: Add `requests` as an explicit dependency**

Run: `uv add requests`
Expected: `pyproject.toml` gains `requests` under `[project].dependencies`; `uv.lock` updates; `.venv` syncs. (`requests` was already present transitively via boto3/mlflow, so no new wheels may download.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_gpu_do_client.py`:

```python
import types

import pytest

from ml_lab.gpu import do_client
from ml_lab.gpu.do_client import DOClientError


def _completed(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_is_lab_droplet_matches_tag_or_name_prefix():
    assert do_client._is_lab_droplet({"name": "x", "tags": ["ml-lab"]}) is True
    assert do_client._is_lab_droplet({"name": "ml-lab-gpu-phase0-1", "tags": []}) is True
    assert do_client._is_lab_droplet({"name": "unrelated", "tags": []}) is False


def test_list_lab_droplets_filters(monkeypatch):
    payload = (
        '[{"id":1,"name":"ml-lab-gpu-phase0-a","tags":["ml-lab"]},'
        '{"id":2,"name":"ml-lab-gpu-phase0-b","tags":[]},'
        '{"id":3,"name":"someone-else","tags":["other"]}]'
    )
    monkeypatch.setattr(
        do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload)
    )
    result = do_client.list_lab_droplets()
    assert [d["id"] for d in result] == [1, 2]


def test_get_droplet_returns_dict(monkeypatch):
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(stdout='[{"id":7,"status":"active"}]'),
    )
    assert do_client.get_droplet(7) == {"id": 7, "status": "active"}


def test_get_droplet_absent_returns_none(monkeypatch):
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: GET ... 404 not found"),
    )
    assert do_client.get_droplet(7) is None


def test_get_droplet_other_error_raises(monkeypatch):
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: 500 server error"),
    )
    with pytest.raises(DOClientError):
        do_client.get_droplet(7)


def test_destroy_droplet_status_mapping(monkeypatch):
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(returncode=0))
    assert do_client.destroy_droplet(7) == "accepted"

    monkeypatch.setattr(
        do_client.subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="404 not found")
    )
    assert do_client.destroy_droplet(7) == "gone"

    monkeypatch.setattr(
        do_client.subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="500 boom")
    )
    assert do_client.destroy_droplet(7) == "error"


def test_probe_destroy_token_ok_on_404(monkeypatch):
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")
    monkeypatch.setattr(
        do_client.requests, "delete", lambda *a, **k: types.SimpleNamespace(status_code=404)
    )
    do_client.probe_destroy_token()  # no raise


@pytest.mark.parametrize("code", [401, 403, 204])
def test_probe_destroy_token_bad_status_raises(monkeypatch, code):
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")
    monkeypatch.setattr(
        do_client.requests, "delete", lambda *a, **k: types.SimpleNamespace(status_code=code)
    )
    with pytest.raises(DOClientError):
        do_client.probe_destroy_token()


def test_probe_destroy_token_missing_env_raises(monkeypatch):
    monkeypatch.delenv("DO_DROPLET_DESTROY_TOKEN", raising=False)
    with pytest.raises(DOClientError):
        do_client.probe_destroy_token()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_do_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.do_client'`.

- [ ] **Step 4: Create `do_client.py`**

Create `src/ml_lab/gpu/do_client.py`:

```python
"""The DigitalOcean seam: intent-level calls, mocked in every S1 test.

doctl (subprocess) handles list/get/destroy; the destroy-token probe uses the
REST API directly, per master plan §3.
"""

from __future__ import annotations

import json
import os
import subprocess

import requests

from ml_lab.gpu.constants import DROPLET_NAME_PREFIX

DO_API_BASE = "https://api.digitalocean.com/v2"
_LAB_TAGS = {"ml-lab", "ml-pathway"}


class DOClientError(RuntimeError):
    """A DigitalOcean interaction failed in a way that is not a clean 404."""


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


def _is_lab_droplet(droplet: dict) -> bool:
    tags = droplet.get("tags") or []
    name = droplet.get("name") or ""
    return "ml-lab" in tags or name.startswith(DROPLET_NAME_PREFIX)


def _has_lab_tag(resource: dict) -> bool:
    return bool(_LAB_TAGS & set(resource.get("tags") or []))


def list_lab_droplets() -> list[dict]:
    return [d for d in _run_doctl(["compute", "droplet", "list"]) if _is_lab_droplet(d)]


def get_droplet(droplet_id: int) -> dict | None:
    result = subprocess.run(
        ["doctl", "compute", "droplet", "get", str(droplet_id), "-o", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return data[0] if isinstance(data, list) else data
    if "404" in (result.stderr or ""):
        return None
    raise DOClientError(f"droplet get failed: {(result.stderr or '').strip()}")


def destroy_droplet(droplet_id: int) -> str:
    """Return 'accepted' (2xx), 'gone' (404), or 'error' (anything else)."""
    result = subprocess.run(
        ["doctl", "compute", "droplet", "delete", str(droplet_id), "--force"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return "accepted"
    if "404" in (result.stderr or ""):
        return "gone"
    return "error"


def probe_destroy_token() -> None:
    """Verify the destroy-scoped token authenticates and carries delete scope.

    DELETE /v2/droplets/1 against a known-nonexistent droplet: 404 = good.
    """
    token = os.environ.get("DO_DROPLET_DESTROY_TOKEN")
    if not token:
        raise DOClientError("DO_DROPLET_DESTROY_TOKEN is not set")
    try:
        resp = requests.delete(
            f"{DO_API_BASE}/droplets/1",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise DOClientError(f"destroy-token probe network failure: {exc}") from exc
    if resp.status_code == 404:
        return
    if resp.status_code in (401, 403):
        raise DOClientError(f"destroy token rejected: HTTP {resp.status_code}")
    raise DOClientError(
        f"unexpected destroy-token probe response: HTTP {resp.status_code}"
    )


def list_lab_volumes() -> list[dict]:
    return [v for v in _run_doctl(["compute", "volume", "list"]) if _has_lab_tag(v)]


def list_lab_snapshots() -> list[dict]:
    return [s for s in _run_doctl(["compute", "snapshot", "list"]) if _has_lab_tag(s)]


def list_lab_reserved_ips() -> list[dict]:
    return [r for r in _run_doctl(["compute", "reserved-ip", "list"]) if _has_lab_tag(r)]


def list_lab_load_balancers() -> list[dict]:
    return [lb for lb in _run_doctl(["compute", "load-balancer", "list"]) if _has_lab_tag(lb)]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_do_client.py -v`
Expected: PASS — all `do_client` tests green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ml_lab/gpu/do_client.py tests/test_gpu_do_client.py
git commit -m "feat: do_client DigitalOcean seam + explicit requests dep"
```

---

### Task 3: `audit` — structured report, formatter, `is_clean`

**Files:**
- Create: `src/ml_lab/gpu/audit.py`
- Test: `tests/test_gpu_audit.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_audit.py`:

```python
from ml_lab.gpu import audit, do_client
from ml_lab.gpu.audit import AuditReport

_EMPTY_LISTERS = [
    "list_lab_droplets",
    "list_lab_volumes",
    "list_lab_snapshots",
    "list_lab_reserved_ips",
    "list_lab_load_balancers",
]


def _all_empty(monkeypatch):
    for fn in _EMPTY_LISTERS:
        monkeypatch.setattr(do_client, fn, lambda: [])


def _droplet(**over):
    d = {
        "id": 123456789,
        "name": "ml-lab-gpu-phase0-20260711-abc123",
        "status": "active",
        "region": {"slug": "atl1"},
        "size_slug": "gpu-rtx4000x1-20gb",
        "image": {"slug": "nvidia-ai-ml"},
        "networks": {"v4": [{"type": "public", "ip_address": "143.0.0.1"}]},
        "created_at": "2026-07-11T10:00:00Z",
        "tags": ["ml-lab", "ttl-expiry-1700000000"],
    }
    d.update(over)
    return d


def test_clean_report_is_clean(monkeypatch):
    _all_empty(monkeypatch)
    report = audit.collect_audit(now=1_800_000_000.0)
    assert audit.is_clean(report) is True
    assert "audit clean" in audit.format_report(report)


def test_droplet_reported_with_fields_and_overdue(monkeypatch):
    ttl_epoch = 1_700_000_000
    _all_empty(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_lab_droplets", lambda: [_droplet(tags=["ml-lab", f"ttl-expiry-{ttl_epoch}"])]
    )
    report = audit.collect_audit(now=ttl_epoch + 3600)  # one hour past expiry
    assert audit.is_clean(report) is False
    info = report.droplets[0]
    assert info.id == 123456789
    assert info.region == "atl1"
    assert info.size == "gpu-rtx4000x1-20gb"
    assert info.public_ip == "143.0.0.1"
    assert info.overdue is True
    assert info.destroy_command == "make gpu-down DROPLET_ID=123456789"
    text = audit.format_report(report)
    assert "DIRTY" in text
    assert "make gpu-down DROPLET_ID=123456789" in text


def test_not_overdue_when_before_expiry(monkeypatch):
    ttl_epoch = 1_700_000_000
    _all_empty(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_lab_droplets", lambda: [_droplet(tags=["ml-lab", f"ttl-expiry-{ttl_epoch}"])]
    )
    report = audit.collect_audit(now=ttl_epoch - 3600)  # before expiry
    assert report.droplets[0].overdue is False


def test_tagged_volume_makes_report_dirty(monkeypatch):
    _all_empty(monkeypatch)
    monkeypatch.setattr(do_client, "list_lab_volumes", lambda: [{"id": "v1", "tags": ["ml-lab"]}])
    report = audit.collect_audit(now=1.0)
    assert audit.is_clean(report) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.audit'`.

- [ ] **Step 3: Create `audit.py`**

Create `src/ml_lab/gpu/audit.py`:

```python
"""Find lab GPU resources and report whether the account is billing-clean."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ml_lab.gpu import do_client


@dataclass
class DropletInfo:
    id: int
    name: str
    status: str
    region: str
    size: str
    image: str
    public_ip: str
    age: str
    ttl_expiry: str | None
    overdue: bool
    destroy_command: str


@dataclass
class AuditReport:
    droplets: list[DropletInfo] = field(default_factory=list)
    volumes: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    reserved_ips: list[dict] = field(default_factory=list)
    load_balancers: list[dict] = field(default_factory=list)


def _region_slug(d: dict) -> str:
    region = d.get("region")
    if isinstance(region, dict):
        return region.get("slug", "")
    return region or ""


def _image_slug(d: dict) -> str:
    image = d.get("image") or {}
    return image.get("slug") or image.get("name") or ""


def _public_ip(d: dict) -> str:
    for net in (d.get("networks") or {}).get("v4") or []:
        if net.get("type") == "public":
            return net.get("ip_address", "")
    return ""


def _ttl_expiry_tag(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith("ttl-expiry-"):
            return tag[len("ttl-expiry-"):]
    return None


def _created_epoch(created_at: str | None) -> float | None:
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fmt_age(created_at: str | None, now: float) -> str:
    epoch = _created_epoch(created_at)
    if epoch is None:
        return "unknown"
    secs = max(0, int(now - epoch))
    hours, rem = divmod(secs, 3600)
    return f"{hours}h{rem // 60:02d}m"


def _fmt_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def _to_info(d: dict, now: float) -> DropletInfo:
    tags = d.get("tags") or []
    ttl_raw = _ttl_expiry_tag(tags)
    overdue = False
    ttl_display: str | None = None
    if ttl_raw is not None:
        try:
            ttl_epoch = int(ttl_raw)
            overdue = now > ttl_epoch
            ttl_display = _fmt_epoch(ttl_epoch)
        except ValueError:
            ttl_display = ttl_raw
    droplet_id = int(d["id"])
    return DropletInfo(
        id=droplet_id,
        name=d.get("name", ""),
        status=d.get("status", ""),
        region=_region_slug(d),
        size=d.get("size_slug", ""),
        image=_image_slug(d),
        public_ip=_public_ip(d),
        age=_fmt_age(d.get("created_at"), now),
        ttl_expiry=ttl_display,
        overdue=overdue,
        destroy_command=f"make gpu-down DROPLET_ID={droplet_id}",
    )


def collect_audit(now: float | None = None) -> AuditReport:
    if now is None:
        now = time.time()
    return AuditReport(
        droplets=[_to_info(d, now) for d in do_client.list_lab_droplets()],
        volumes=do_client.list_lab_volumes(),
        snapshots=do_client.list_lab_snapshots(),
        reserved_ips=do_client.list_lab_reserved_ips(),
        load_balancers=do_client.list_lab_load_balancers(),
    )


def is_clean(report: AuditReport) -> bool:
    return not (
        report.droplets
        or report.volumes
        or report.snapshots
        or report.reserved_ips
        or report.load_balancers
    )


def format_report(report: AuditReport) -> str:
    lines = ["GPU AUDIT · phase-0"]
    if is_clean(report):
        lines += [
            "  droplets (tag ml-lab | name ml-lab-gpu-*): none",
            "  volumes: none  snapshots: none  reserved-ips: none  load-balancers: none",
            "",
            "✓ audit clean — no billable lab resources",
        ]
        return "\n".join(lines)

    lines.append(f"  ✗ {len(report.droplets)} lab GPU droplet(s):")
    for d in report.droplets:
        overdue = "  → OVERDUE" if d.overdue else ""
        lines += [
            f"    id: {d.id}  name: {d.name}",
            f"    status: {d.status}  region: {d.region}  size: {d.size}",
            f"    image: {d.image}  public-ip: {d.public_ip}  age: {d.age}",
            f"    ttl-expiry: {d.ttl_expiry}{overdue}",
            f"    destroy: {d.destroy_command}",
        ]
    lines += [
        f"  related — volumes: {len(report.volumes)}  snapshots: {len(report.snapshots)}"
        f"  reserved-ips: {len(report.reserved_ips)}  load-balancers: {len(report.load_balancers)}",
        "",
        "✗ audit DIRTY — billable lab resources exist",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_audit.py -v`
Expected: PASS — all audit tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/audit.py tests/test_gpu_audit.py
git commit -m "feat: GPU audit report (tags + name-prefix, overdue, formatter)"
```

---

### Task 4: `destroy_and_verify` — SIGINT-hardened teardown

**Files:**
- Create: `src/ml_lab/gpu/teardown.py`
- Test: `tests/test_gpu_teardown.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_teardown.py`:

```python
import os
import signal
import subprocess
import sys
import time

import pytest

from ml_lab.gpu import audit, do_client, teardown
from ml_lab.gpu.audit import AuditReport, DropletInfo


def _info():
    return DropletInfo(
        id=1,
        name="ml-lab-gpu-phase0-x",
        status="active",
        region="atl1",
        size="gpu-rtx4000x1-20gb",
        image="img",
        public_ip="143.0.0.1",
        age="0h01m",
        ttl_expiry=None,
        overdue=False,
        destroy_command="make gpu-down DROPLET_ID=1",
    )


def _clean(monkeypatch):
    monkeypatch.setattr(audit, "collect_audit", lambda now=None: AuditReport())


def test_happy_path(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)  # no raise


def test_404_on_destroy_is_success(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "gone")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)  # no raise


def test_poll_then_absent(monkeypatch):
    seq = iter([{"id": 1}, {"id": 1}, None])
    calls = []

    def fake_get(i):
        calls.append(i)
        return next(seq)

    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", fake_get)
    monkeypatch.setattr(teardown.time, "sleep", lambda s: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)
    assert len(calls) == 3


def test_poll_timeout_raises(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: {"id": 1})
    monkeypatch.setattr(teardown.time, "sleep", lambda s: None)
    ticks = iter([0.0, 1.0, 700.0, 1400.0])
    monkeypatch.setattr(teardown.time, "monotonic", lambda: next(ticks))
    _clean(monkeypatch)
    with pytest.raises(teardown.TeardownError):
        teardown.destroy_and_verify(1)


def test_audit_dirty_after_absence_raises(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    monkeypatch.setattr(audit, "collect_audit", lambda now=None: AuditReport(droplets=[_info()]))
    with pytest.raises(teardown.TeardownError):
        teardown.destroy_and_verify(1)


def test_transient_get_error_then_absent_succeeds(monkeypatch):
    seq = iter([do_client.DOClientError("transient"), None])

    def fake_get(i):
        value = next(seq)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "error")
    monkeypatch.setattr(do_client, "get_droplet", fake_get)
    monkeypatch.setattr(teardown.time, "sleep", lambda s: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)  # no raise


def test_sigint_ignored_during_poll_and_restored(monkeypatch):
    observed = {}

    def fake_get(i):
        observed["handler"] = signal.getsignal(signal.SIGINT)
        return None

    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", fake_get)
    _clean(monkeypatch)
    before = signal.getsignal(signal.SIGINT)
    teardown.destroy_and_verify(1)
    assert observed["handler"] == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == before


def test_sigint_restored_after_failure(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    monkeypatch.setattr(audit, "collect_audit", lambda now=None: AuditReport(droplets=[_info()]))
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(teardown.TeardownError):
        teardown.destroy_and_verify(1)
    assert signal.getsignal(signal.SIGINT) == before


# --- Heavier: a real subprocess proves SIGINT cannot abort an armed teardown ---

_SIGINT_RUNNER = '''\
from ml_lab.gpu import teardown, do_client, audit

teardown.DESTROY_POLL_INTERVAL_SECONDS = 0.2

_calls = {"n": 0}


def _fake_get(droplet_id):
    _calls["n"] += 1
    if _calls["n"] > 5:
        return None
    return {"id": droplet_id, "status": "active"}


do_client.get_droplet = _fake_get
do_client.destroy_droplet = lambda droplet_id: "accepted"
audit.collect_audit = lambda now=None: audit.AuditReport()

teardown.destroy_and_verify(12345)
print("DONE", flush=True)
'''


def test_sigint_does_not_abort_teardown_subprocess(tmp_path):
    runner = tmp_path / "sigint_runner.py"
    runner.write_text(_SIGINT_RUNNER)
    env = {**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.Popen(
        [sys.executable, str(runner)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Wait for the arming banner so SIGINT lands AFTER SIG_IGN is installed.
    armed = False
    for _ in range(200):
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if "Teardown in progress" in line:
            armed = True
            break
    assert armed, "teardown never armed (banner not seen)"

    # Hammer SIGINT while the poll loop is running.
    for _ in range(3):
        proc.send_signal(signal.SIGINT)
        time.sleep(0.1)

    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"process died (rc={proc.returncode}); output:\n{out}"
    assert "DONE" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_teardown.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.teardown'`.

- [ ] **Step 3: Create `teardown.py`**

Create `src/ml_lab/gpu/teardown.py`:

```python
"""The one function every teardown path routes through.

Success = the droplet does not exist AND audit is clean. Once armed, SIGINT is
ignored so operator impatience cannot strand a billable droplet mid-teardown.
"""

from __future__ import annotations

import signal
import time

from ml_lab.gpu import audit as audit_mod
from ml_lab.gpu import do_client
from ml_lab.gpu.constants import (
    DESTROY_POLL_INTERVAL_SECONDS,
    DESTROY_POLL_TIMEOUT_SECONDS,
)


class TeardownError(RuntimeError):
    """Teardown could not confirm the droplet is gone and the account is clean."""


def _poll_until_absent(droplet_id: int) -> None:
    deadline = time.monotonic() + DESTROY_POLL_TIMEOUT_SECONDS
    while True:
        try:
            droplet = do_client.get_droplet(droplet_id)
        except do_client.DOClientError:
            droplet = "unknown"  # transient; the poll, not the call, is the truth
        if droplet is None:
            return
        if time.monotonic() >= deadline:
            raise TeardownError(
                f"droplet {droplet_id} still present after {DESTROY_POLL_TIMEOUT_SECONDS}s"
            )
        time.sleep(DESTROY_POLL_INTERVAL_SECONDS)


def destroy_and_verify(droplet_id: int) -> None:
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(
        f"Teardown in progress — droplet {droplet_id} is being destroyed. Do not interrupt.",
        flush=True,
    )
    try:
        do_client.destroy_droplet(droplet_id)  # 404 or transient error → poll anyway
        _poll_until_absent(droplet_id)
        report = audit_mod.collect_audit()
        if not audit_mod.is_clean(report):
            raise TeardownError(
                "audit dirty after destroy:\n" + audit_mod.format_report(report)
            )
    finally:
        signal.signal(signal.SIGINT, previous)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_teardown.py -v`
Expected: PASS — all unit tests plus the subprocess SIGINT test green. (The subprocess test takes ~1–2s.)

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/teardown.py tests/test_gpu_teardown.py
git commit -m "feat: SIGINT-hardened destroy_and_verify (404-idempotent, audit-gated)"
```

---

### Task 5: Typer `app` + `scripts/do_gpu.py` shim

**Files:**
- Create: `src/ml_lab/gpu/cli.py`
- Create: `scripts/do_gpu.py`
- Test: `tests/test_gpu_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_cli.py`:

```python
from typer.testing import CliRunner

from ml_lab.gpu import audit as audit_mod
from ml_lab.gpu import cli as gpu_cli
from ml_lab.gpu.audit import AuditReport, DropletInfo


def _info():
    return DropletInfo(
        id=555,
        name="ml-lab-gpu-phase0-x",
        status="active",
        region="atl1",
        size="gpu-rtx4000x1-20gb",
        image="img",
        public_ip="143.0.0.1",
        age="0h01m",
        ttl_expiry=None,
        overdue=False,
        destroy_command="make gpu-down DROPLET_ID=555",
    )


def test_help_names_audit_and_down():
    result = CliRunner().invoke(gpu_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output
    assert "down" in result.output


def test_down_calls_destroy_and_verify(monkeypatch):
    called = {}
    monkeypatch.setattr(gpu_cli, "destroy_and_verify", lambda did: called.setdefault("id", did))
    result = CliRunner().invoke(gpu_cli.app, ["down", "--droplet-id", "555"])
    assert result.exit_code == 0
    assert called["id"] == 555


def test_audit_exits_zero_when_clean(monkeypatch):
    monkeypatch.setattr(audit_mod, "collect_audit", lambda: AuditReport())
    result = CliRunner().invoke(gpu_cli.app, ["audit"])
    assert result.exit_code == 0
    assert "audit clean" in result.output


def test_audit_exits_nonzero_when_dirty(monkeypatch):
    monkeypatch.setattr(audit_mod, "collect_audit", lambda: AuditReport(droplets=[_info()]))
    result = CliRunner().invoke(gpu_cli.app, ["audit"])
    assert result.exit_code == 1
    assert "DIRTY" in result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.cli'`.

- [ ] **Step 3: Create `cli.py` and the shim**

Create `src/ml_lab/gpu/cli.py`:

```python
"""Typer surface for the GPU safety spine: `audit` and `down`.

`run` / `up` are added in a later slice. `scripts/do_gpu.py` imports this `app`.
"""

import typer

from ml_lab.gpu import audit as audit_mod
from ml_lab.gpu.teardown import destroy_and_verify

app = typer.Typer(help="DigitalOcean GPU-lab lifecycle (Phase 0 safety spine).")


@app.command("audit")
def audit_command() -> None:
    """Report any billable lab GPU resources; exit nonzero if the account is dirty."""
    report = audit_mod.collect_audit()
    typer.echo(audit_mod.format_report(report))
    if not audit_mod.is_clean(report):
        raise typer.Exit(code=1)


@app.command("down")
def down_command(
    droplet_id: int = typer.Option(..., "--droplet-id", help="Droplet id to destroy."),
) -> None:
    """Destroy a droplet and verify it is gone and audit is clean."""
    destroy_and_verify(droplet_id)
```

Create `scripts/do_gpu.py`:

```python
"""Entry point: `uv run python scripts/do_gpu.py {audit,down}` (Makefile wraps this)."""

from ml_lab.gpu.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_cli.py -v`
Expected: PASS — all four CLI tests green.

- [ ] **Step 5: Confirm the full suite stays green**

Run: `uv run pytest -q`
Expected: PASS — the 53 prior Phase 0 tests plus the new GPU S1 tests.

- [ ] **Step 6: Sanity-check the wiring (no cloud calls needed for `--help`)**

Run: `uv run python scripts/do_gpu.py --help`
Expected: exit 0; help text lists `audit` and `down`.

(Do NOT run `make gpu-audit` here unless `doctl` is authenticated — it makes real
DigitalOcean list calls. `make gpu-run` / `make gpu-up` remain undefined commands
until the S3 slice; that is expected.)

- [ ] **Step 7: Commit**

```bash
git add src/ml_lab/gpu/cli.py scripts/do_gpu.py tests/test_gpu_cli.py
git commit -m "feat: gpu-audit/gpu-down Typer commands + do_gpu.py entry"
```

---

## Definition of Done

- `uv run pytest` is fully green (prior 53 tests + new S1 tests).
- `src/ml_lab/gpu/` exports: `constants` (pinned values + `validate_timeout_budget`), `do_client` (list/get/destroy/probe + related-resource lists + `DOClientError`), `audit` (`collect_audit`/`format_report`/`is_clean`, `AuditReport`/`DropletInfo`), `teardown` (`destroy_and_verify`, `TeardownError`), `cli` (`app`).
- `scripts/do_gpu.py --help` lists `audit` and `down`.
- `destroy_and_verify` treats a 404/absent droplet as success, polls to absence, gates on a clean audit, ignores SIGINT once armed, and always restores the handler — proven by both a unit assertion and a real-subprocess signal test.
- `audit` finds droplets by tag **and** name-prefix, flags overdue via the `ttl-expiry` tag, and scans related resources; any match → nonzero exit.
- `requests` is an explicit dependency.
- No code path in S1 can create or reach a real droplet (the `do_client` seam is mocked in every test).
- Out of scope and untouched: `create_droplet`, `run`/`up`, cloud-init, `gpu_benchmark.py`, Spaces upload, live failure-path gates (S2–S4); `DO_IMAGE_SLUG` stays `None`.
