# Phase 0 GPU Create Path Slice (S2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `create_lab_droplet` engine — preflight-gated (duplicate refusal, constant validation, destroy-token probe, timeout budget), atomically tagged, with an immediate id/name/pid print — that `gpu-run`/`gpu-up` (S3) will drive, and correct the three account-reality constants (`nyc2`, `gpu-4000adax1-20gb`, `gpu-h100x1-base`).

**Architecture:** New `src/ml_lab/gpu/create.py` holds the create logic over the existing `do_client` seam (extended here with `create_droplet` + slug-availability queries). All preflight gates live *inside* `create_lab_droplet` so both S3 commands inherit the identical safety sequence. `user_data` (cloud-init with the self-destruct timer) is a required parameter; `run_id`/`now` are injectable for deterministic tests. No new `make` target — S2 is a library slice exercised entirely by mocked tests.

**Tech Stack:** Python 3.11+, `doctl` (subprocess, mocked in tests), pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Modify:** `src/ml_lab/gpu/constants.py` — correct `DO_REGION`, `DO_SIZE_SLUG`, `DO_IMAGE_SLUG` (Task 1).
- **Modify:** `src/ml_lab/gpu/do_client.py` — add `list_region_slugs`, `list_sizes`, `list_image_slugs` (Task 2) and `create_droplet` (Task 3).
- **Create:** `src/ml_lab/gpu/create.py` — `ConstantsError`/`validate_constants` (Task 4), `generate_run_id`/`build_tags` (Task 5), `LabDropletExistsError`/`create_lab_droplet` (Task 6). Imports are added incrementally per task.
- **Modify:** `tests/test_gpu_constants.py` (Task 1), `tests/test_gpu_do_client.py` (Tasks 2–3).
- **Create:** `tests/test_gpu_create.py` — extended across Tasks 4–6.

Depends on (already committed S1, do not modify): `do_client.list_lab_droplets`, `do_client.probe_destroy_token`, `constants.validate_timeout_budget`, `constants.BASE_TAGS`/`NAME_FORMAT`/`DEFAULT_TTL_SECONDS`. pytest is configured with `pythonpath = ["src"]`. `run`/`up` commands, cloud-init rendering, and SSH-key sourcing are OUT of scope (S3).

---

### Task 1: Correct the three account-reality constants

**Files:**
- Modify: `src/ml_lab/gpu/constants.py`
- Test: `tests/test_gpu_constants.py`

- [ ] **Step 1: Update the failing assertions**

In `tests/test_gpu_constants.py`, replace the body of `test_pinned_constants_present`. Change:

```python
def test_pinned_constants_present():
    assert constants.DO_REGION == "atl1"
    assert constants.DO_SIZE_SLUG == "gpu-rtx4000x1-20gb"
    assert constants.DROPLET_NAME_PREFIX == "ml-lab-gpu-"
    assert constants.DESTROY_POLL_INTERVAL_SECONDS == 10
    assert "ml-lab" in constants.BASE_TAGS
```

to:

```python
def test_pinned_constants_present():
    assert constants.DO_REGION == "nyc2"
    assert constants.DO_SIZE_SLUG == "gpu-4000adax1-20gb"
    assert constants.DO_IMAGE_SLUG == "gpu-h100x1-base"
    assert constants.DROPLET_NAME_PREFIX == "ml-lab-gpu-"
    assert constants.DESTROY_POLL_INTERVAL_SECONDS == 10
    assert "ml-lab" in constants.BASE_TAGS
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_gpu_constants.py::test_pinned_constants_present -v`
Expected: FAIL — asserts `nyc2` but the code still says `atl1`.

- [ ] **Step 3: Correct the constants**

In `src/ml_lab/gpu/constants.py`, change:

```python
DO_REGION = "atl1"
DO_SIZE_SLUG = "gpu-rtx4000x1-20gb"
DO_GPU_RUNG = "RTX 4000 Ada"
DO_IMAGE_SLUG = None  # DO NVIDIA AI/ML-ready GPU image; resolved in S2 (create-only)
```

to:

```python
DO_REGION = "nyc2"  # atl1 has no GPU capacity in-account; deliberate change (master plan §3)
DO_SIZE_SLUG = "gpu-4000adax1-20gb"  # RTX 4000 Ada single-GPU (type nvidia_rtx4000_ada)
DO_GPU_RUNG = "RTX 4000 Ada"
DO_IMAGE_SLUG = "gpu-h100x1-base"  # NVIDIA AI/ML Ready Image; slug legacy-named, verify RTX-4000 boot at S3 live
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_constants.py -v`
Expected: PASS — all constant tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/constants.py tests/test_gpu_constants.py
git commit -m "fix: correct DO region/size/image slugs to account reality (nyc2, 4000ada, h100x1-base)"
```

---

### Task 2: `do_client` slug-availability queries

**Files:**
- Modify: `src/ml_lab/gpu/do_client.py`
- Test: `tests/test_gpu_do_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_do_client.py` (the `_completed` helper and `import types` already exist at the top of the file from S1):

```python
def test_list_region_slugs_returns_available_only(monkeypatch):
    payload = '[{"slug":"nyc2","available":true},{"slug":"sfo1","available":false}]'
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload))
    assert do_client.list_region_slugs() == ["nyc2"]


def test_list_sizes_returns_raw_dicts(monkeypatch):
    payload = '[{"slug":"gpu-4000adax1-20gb","regions":["nyc2"]}]'
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload))
    sizes = do_client.list_sizes()
    assert sizes[0]["slug"] == "gpu-4000adax1-20gb"
    assert sizes[0]["regions"] == ["nyc2"]


def test_list_image_slugs_filters_null_slugs(monkeypatch):
    payload = '[{"slug":"gpu-h100x1-base"},{"slug":null},{"slug":"other"}]'
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload))
    assert do_client.list_image_slugs() == ["gpu-h100x1-base", "other"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_do_client.py -k "region_slugs or list_sizes or image_slugs" -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.do_client' has no attribute 'list_region_slugs'`.

- [ ] **Step 3: Add the query functions**

Append to `src/ml_lab/gpu/do_client.py`:

```python
def list_region_slugs() -> list[str]:
    return [r["slug"] for r in _run_doctl(["compute", "region", "list"]) if r.get("available")]


def list_sizes() -> list[dict]:
    return _run_doctl(["compute", "size", "list"])


def list_image_slugs() -> list[str]:
    return [
        i["slug"]
        for i in _run_doctl(["compute", "image", "list", "--public"])
        if i.get("slug")
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_do_client.py -v`
Expected: PASS — the three new tests plus all existing S1 `do_client` tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/do_client.py tests/test_gpu_do_client.py
git commit -m "feat: do_client slug-availability queries (regions/sizes/images)"
```

---

### Task 3: `do_client.create_droplet`

**Files:**
- Modify: `src/ml_lab/gpu/do_client.py`
- Test: `tests/test_gpu_do_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_do_client.py`:

```python
def test_create_droplet_builds_argv_and_parses_id(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(stdout='[{"id": 42, "name": "ml-lab-gpu-phase0-x", "status": "new"}]')

    monkeypatch.setattr(do_client.subprocess, "run", fake_run)
    droplet = do_client.create_droplet(
        name="ml-lab-gpu-phase0-x",
        region="nyc2",
        size="gpu-4000adax1-20gb",
        image="gpu-h100x1-base",
        tags=["ml-lab", "run-x"],
        user_data="#cloud-config\n",
        ssh_key_ids=["aa:bb"],
    )
    assert droplet["id"] == 42
    argv = captured["argv"]
    assert argv[:4] == ["doctl", "compute", "droplet", "create"]
    assert "ml-lab-gpu-phase0-x" in argv
    for flag, val in [
        ("--region", "nyc2"),
        ("--size", "gpu-4000adax1-20gb"),
        ("--image", "gpu-h100x1-base"),
    ]:
        assert flag in argv and argv[argv.index(flag) + 1] == val
    assert "--tag-names" in argv and argv[argv.index("--tag-names") + 1] == "ml-lab,run-x"
    assert "--user-data-file" in argv
    assert "--ssh-keys" in argv and argv[argv.index("--ssh-keys") + 1] == "aa:bb"
    assert "--wait" not in argv
    assert argv[-2:] == ["-o", "json"]


def test_create_droplet_omits_ssh_keys_when_none(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(stdout='[{"id": 7, "name": "n", "status": "new"}]')

    monkeypatch.setattr(do_client.subprocess, "run", fake_run)
    do_client.create_droplet(
        name="n", region="nyc2", size="s", image="i", tags=["ml-lab"], user_data="#cloud-config\n"
    )
    assert "--ssh-keys" not in captured["argv"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_do_client.py -k create_droplet -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.do_client' has no attribute 'create_droplet'`.

- [ ] **Step 3: Add `create_droplet` and the tempfile import**

In `src/ml_lab/gpu/do_client.py`, add `import tempfile` to the imports (with the other stdlib imports near `import subprocess`):

```python
import subprocess
import tempfile
```

Then append the function:

```python
def create_droplet(
    *, name, region, size, image, tags, user_data, ssh_key_ids=None
) -> dict:
    """Create a tagged GPU droplet with cloud-init; return the created droplet dict.

    Tags are applied atomically via --tag-names (never create-then-tag). cloud-init
    is passed by file. NO --wait, so the id returns immediately for the create-time
    identity print. Runs through _run_doctl, which appends -o json and check=True.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(user_data)
        user_data_path = handle.name
    try:
        args = [
            "compute",
            "droplet",
            "create",
            name,
            "--region",
            region,
            "--size",
            size,
            "--image",
            image,
            "--tag-names",
            ",".join(tags),
            "--user-data-file",
            user_data_path,
        ]
        if ssh_key_ids:
            args += ["--ssh-keys", ",".join(ssh_key_ids)]
        result = _run_doctl(args)
    finally:
        os.unlink(user_data_path)
    return result[0] if isinstance(result, list) else result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_do_client.py -v`
Expected: PASS — both `create_droplet` tests plus all existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/do_client.py tests/test_gpu_do_client.py
git commit -m "feat: do_client.create_droplet (atomic tags, cloud-init file, no --wait)"
```

---

### Task 4: `create.py` — `validate_constants`

**Files:**
- Create: `src/ml_lab/gpu/create.py`
- Test: `tests/test_gpu_create.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_create.py`:

```python
import pytest

from ml_lab.gpu import create, do_client
from ml_lab.gpu.create import ConstantsError


def _valid(monkeypatch):
    """Mock do_client so validate_constants passes for the corrected constants."""
    monkeypatch.setattr(do_client, "list_region_slugs", lambda: ["nyc2"])
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-4000adax1-20gb", "regions": ["nyc2"]}]
    )
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["gpu-h100x1-base"])


def test_validate_constants_passes(monkeypatch):
    _valid(monkeypatch)
    create.validate_constants()  # no raise


def test_validate_constants_region_missing(monkeypatch):
    _valid(monkeypatch)
    monkeypatch.setattr(do_client, "list_region_slugs", lambda: ["sfo3"])
    with pytest.raises(ConstantsError):
        create.validate_constants()


def test_validate_constants_size_missing(monkeypatch):
    _valid(monkeypatch)
    monkeypatch.setattr(do_client, "list_sizes", lambda: [{"slug": "other", "regions": ["nyc2"]}])
    with pytest.raises(ConstantsError):
        create.validate_constants()


def test_validate_constants_size_not_in_region(monkeypatch):
    _valid(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-4000adax1-20gb", "regions": ["sfo3"]}]
    )
    with pytest.raises(ConstantsError):
        create.validate_constants()


def test_validate_constants_image_missing(monkeypatch):
    _valid(monkeypatch)
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["other"])
    with pytest.raises(ConstantsError):
        create.validate_constants()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_create.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.create'`.

- [ ] **Step 3: Create `create.py` with `validate_constants`**

Create `src/ml_lab/gpu/create.py`:

```python
"""The shared GPU-droplet create path: preflight-gated, atomic-tagged create.

Both gpu-run and gpu-up (S3) call create_lab_droplet, so every safety gate lives
inside it. All DigitalOcean access goes through the mocked do_client seam.
"""

from __future__ import annotations

from ml_lab.gpu import do_client
from ml_lab.gpu.constants import DO_IMAGE_SLUG, DO_REGION, DO_SIZE_SLUG


class ConstantsError(RuntimeError):
    """A pinned region/size/image slug is not available in the account."""


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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_create.py -v`
Expected: PASS — all five `validate_constants` tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/create.py tests/test_gpu_create.py
git commit -m "feat: validate_constants (region/size/image present + size-in-region)"
```

---

### Task 5: `create.py` — `generate_run_id` and `build_tags`

**Files:**
- Modify: `src/ml_lab/gpu/create.py`
- Test: `tests/test_gpu_create.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_create.py`:

```python
def test_generate_run_id_format():
    from datetime import datetime, timezone

    now = 1783728000.0  # a fixed instant
    expected_date = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d")
    run_id = create.generate_run_id(now)
    assert run_id.startswith(expected_date + "-")
    date, suffix = run_id.split("-")
    assert len(date) == 8 and date.isdigit()
    assert len(suffix) == 6 and all(c in "0123456789abcdef" for c in suffix)


def test_build_tags():
    tags = create.build_tags("20260711-abc123", 1783735200)
    assert tags[:4] == ["ml-lab", "ml-pathway", "phase-0", "owner-weldon"]
    assert tags[-2:] == ["run-20260711-abc123", "ttl-expiry-1783735200"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_create.py -k "run_id or build_tags" -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.create' has no attribute 'generate_run_id'`.

- [ ] **Step 3: Add the helpers**

In `src/ml_lab/gpu/create.py`, extend the imports — add `secrets`, `datetime`/`timezone`, and `BASE_TAGS`:

```python
from __future__ import annotations

import secrets
from datetime import datetime, timezone

from ml_lab.gpu import do_client
from ml_lab.gpu.constants import BASE_TAGS, DO_IMAGE_SLUG, DO_REGION, DO_SIZE_SLUG
```

Then append after `validate_constants`:

```python
def generate_run_id(now: float) -> str:
    """Return 'YYYYMMDD-<6 hex>' — date from now (UTC) + random suffix for uniqueness."""
    date = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d")
    return f"{date}-{secrets.token_hex(3)}"


def build_tags(run_id: str, ttl_epoch: int) -> list[str]:
    return [*BASE_TAGS, f"run-{run_id}", f"ttl-expiry-{ttl_epoch}"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_create.py -v`
Expected: PASS — the two new tests plus the `validate_constants` tests.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/gpu/create.py tests/test_gpu_create.py
git commit -m "feat: generate_run_id + build_tags for create path"
```

---

### Task 6: `create.py` — `create_lab_droplet` (the shared engine)

**Files:**
- Modify: `src/ml_lab/gpu/create.py`
- Test: `tests/test_gpu_create.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_create.py`:

```python
def _happy(monkeypatch):
    """Full happy-path environment; returns a dict capturing create_droplet kwargs."""
    monkeypatch.setattr(do_client, "list_lab_droplets", lambda: [])
    monkeypatch.setattr(do_client, "probe_destroy_token", lambda: None)
    _valid(monkeypatch)  # so real validate_constants passes
    calls = {}

    def fake_create(**kwargs):
        calls["kwargs"] = kwargs
        return {"id": 42, "name": kwargs["name"], "status": "new"}

    monkeypatch.setattr(do_client, "create_droplet", fake_create)
    return calls


def test_create_lab_droplet_happy(monkeypatch, capsys):
    calls = _happy(monkeypatch)
    result = create.create_lab_droplet(
        "#cloud-config\n", run_id="20260711-abc123", now=1783728000.0
    )
    assert result == {
        "id": 42,
        "name": "ml-lab-gpu-phase0-20260711-abc123",
        "run_id": "20260711-abc123",
    }
    kw = calls["kwargs"]
    assert kw["name"] == "ml-lab-gpu-phase0-20260711-abc123"
    assert kw["region"] == "nyc2"
    assert kw["size"] == "gpu-4000adax1-20gb"
    assert kw["image"] == "gpu-h100x1-base"
    assert kw["user_data"] == "#cloud-config\n"
    assert "run-20260711-abc123" in kw["tags"]
    assert f"ttl-expiry-{1783728000 + 7200}" in kw["tags"]
    out = capsys.readouterr().out
    assert "id: 42" in out
    assert "ml-lab-gpu-phase0-20260711-abc123" in out
    assert "local_pid:" in out


def test_create_lab_droplet_refuses_when_exists(monkeypatch):
    _happy(monkeypatch)
    monkeypatch.setattr(do_client, "list_lab_droplets", lambda: [{"id": 1, "tags": ["ml-lab"]}])

    def boom(**k):
        raise AssertionError("create_droplet must not be called when a droplet exists")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(create.LabDropletExistsError):
        create.create_lab_droplet("#cloud-config\n")


def test_create_lab_droplet_ttl_expiry_tag(monkeypatch):
    calls = _happy(monkeypatch)
    create.create_lab_droplet("#cloud-config\n", ttl_seconds=3600, run_id="r", now=1000.0)
    assert "ttl-expiry-4600" in calls["kwargs"]["tags"]


def test_create_lab_droplet_enforce_budget_true_raises(monkeypatch):
    _happy(monkeypatch)
    with pytest.raises(ValueError):
        create.create_lab_droplet("#cloud-config\n", ttl_seconds=900)


def test_create_lab_droplet_enforce_budget_false_allows_short_ttl(monkeypatch):
    _happy(monkeypatch)
    result = create.create_lab_droplet(
        "#cloud-config\n", ttl_seconds=900, enforce_budget=False, run_id="r", now=1000.0
    )
    assert result["id"] == 42


def test_create_lab_droplet_passes_ssh_keys(monkeypatch):
    calls = _happy(monkeypatch)
    create.create_lab_droplet(
        "#cloud-config\n", ssh_key_ids=["aa:bb"], run_id="r", now=1000.0
    )
    assert calls["kwargs"]["ssh_key_ids"] == ["aa:bb"]


def test_create_lab_droplet_validates_before_create(monkeypatch):
    _happy(monkeypatch)
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["other"])  # image now missing

    def boom(**k):
        raise AssertionError("create_droplet must not be called when constants invalid")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(create.ConstantsError):
        create.create_lab_droplet("#cloud-config\n")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_create.py -k create_lab_droplet -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.create' has no attribute 'create_lab_droplet'`.

- [ ] **Step 3: Add `create_lab_droplet` and `LabDropletExistsError`**

In `src/ml_lab/gpu/create.py`, extend the imports — add `os`, `time`, and the extra constants (`DEFAULT_TTL_SECONDS`, `NAME_FORMAT`, `validate_timeout_budget`):

```python
from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timezone

from ml_lab.gpu import do_client
from ml_lab.gpu.constants import (
    BASE_TAGS,
    DEFAULT_TTL_SECONDS,
    DO_IMAGE_SLUG,
    DO_REGION,
    DO_SIZE_SLUG,
    NAME_FORMAT,
    validate_timeout_budget,
)
```

Then append:

```python
class LabDropletExistsError(RuntimeError):
    """A matching lab GPU droplet already exists; refuse to create another."""


def create_lab_droplet(
    user_data: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    enforce_budget: bool = True,
    ssh_key_ids: list[str] | None = None,
    run_id: str | None = None,
    now: float | None = None,
) -> dict:
    """Preflight-gated, atomically tagged create shared by gpu-run and gpu-up (S3).

    On any preflight failure, nothing is created. user_data (the cloud-init that
    carries the self-destruct timer) is required. Returns {"id", "name", "run_id"}.
    """
    if do_client.list_lab_droplets():
        raise LabDropletExistsError(
            "a matching lab GPU droplet already exists; destroy it before creating another"
        )
    validate_constants()
    do_client.probe_destroy_token()
    if enforce_budget:
        validate_timeout_budget(ttl_seconds)

    if now is None:
        now = time.time()
    if run_id is None:
        run_id = generate_run_id(now)
    name = NAME_FORMAT.format(run_id=run_id)
    ttl_epoch = int(now) + ttl_seconds
    tags = build_tags(run_id, ttl_epoch)

    droplet = do_client.create_droplet(
        name=name,
        region=DO_REGION,
        size=DO_SIZE_SLUG,
        image=DO_IMAGE_SLUG,
        tags=tags,
        user_data=user_data,
        ssh_key_ids=ssh_key_ids,
    )
    print(
        f"Created droplet:\n  id: {droplet['id']}\n  name: {name}\n  local_pid: {os.getpid()}",
        flush=True,
    )
    return {"id": droplet["id"], "name": name, "run_id": run_id}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_create.py -v`
Expected: PASS — all create-path tests green.

- [ ] **Step 5: Confirm the full suite stays green**

Run: `uv run pytest -q`
Expected: PASS — the 85 S1 tests plus the new S2 tests.

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/create.py tests/test_gpu_create.py
git commit -m "feat: create_lab_droplet shared engine (preflight-gated, atomic tags, id print)"
```

---

## Definition of Done

- `uv run pytest` is fully green (85 S1 tests + new S2 tests).
- Constants corrected: `DO_REGION="nyc2"`, `DO_SIZE_SLUG="gpu-4000adax1-20gb"`, `DO_IMAGE_SLUG="gpu-h100x1-base"`.
- `do_client` exposes `list_region_slugs`, `list_sizes`, `list_image_slugs`, and `create_droplet` (atomic `--tag-names`, cloud-init via `--user-data-file`, no `--wait`, optional `--ssh-keys`).
- `create.py` exposes `validate_constants` (region + size-in-region + image), `generate_run_id`, `build_tags`, `create_lab_droplet`, `ConstantsError`, `LabDropletExistsError`.
- `create_lab_droplet` runs the master-plan preflight order (refuse-existing → validate_constants → probe_destroy_token → timeout budget) entirely before any create; `user_data` is required; tags are atomic; id/name/pid print immediately; `run_id`/`now` are injectable.
- No S2 code path creates or reaches a real droplet (the `do_client` seam is mocked in every test).
- Out of scope and untouched: `gpu-run`/`gpu-up` commands, cloud-init rendering, SSH/bootstrap/benchmark, Spaces upload (S3/S4).
