# Phase 0 GPU Cloud-init Renderer Slice (S3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `render_cloud_init(...)` — a pure function that renders the self-destruct cloud-config (from a collision-proof `@@PLACEHOLDER@@` template) into the `user_data` string that S2's `create_lab_droplet` already requires — plus the `scripts/cloud-init-gpu.yaml.tmpl` template and its self-destruct service/timer units.

**Architecture:** New `src/ml_lab/gpu/cloud_init.py` reads `scripts/cloud-init-gpu.yaml.tmpl` and does one `str.replace` per placeholder, leaving the template's literal bash `${...}` untouched (that is why we use `@@NAME@@` markers, not `str.format`/`string.Template`). A fail-loud guard raises `ValueError` on an empty token or any unfilled placeholder, so a broken self-destruct becomes a pre-create hard stop. A library slice — nothing calls the renderer yet (S3b is the first caller); the entire slice is offline-testable, no network, no droplet.

**Tech Stack:** Python 3.11+, pytest, uv. PyYAML is added as a **dev-only** dependency for a structural YAML-parse check in tests (the renderer itself does no YAML parsing). Run tests with `uv run pytest`.

---

## File Structure

- **Modify:** `pyproject.toml` — add `pyyaml` to the `dev` dependency group (Task 1).
- **Create:** `scripts/cloud-init-gpu.yaml.tmpl` — the master-plan cloud-config with `@@...@@` placeholders and the self-destruct units (Task 2).
- **Create:** `src/ml_lab/gpu/cloud_init.py` — `render_cloud_init` (substitution in Task 2; fail-loud guards in Task 3).
- **Create:** `tests/test_gpu_cloud_init.py` — happy-path/invariant tests (Task 2), guard tests (Task 3).

Depends on (already committed, do not modify): `constants.DEFAULT_TTL_SECONDS`, `constants.SELF_DESTRUCT_RETRY_SECONDS`, `constants.SMOKE_MODEL_ID`. pytest is configured with `pythonpath = ["src"]`. Reading `DO_DROPLET_DESTROY_TOKEN` from the environment, calling `create_lab_droplet`, SSH/bootstrap orchestration, `gpu-up`/`gpu-run`, and `gpu_benchmark.py` are all OUT of scope (S3b/S3c).

---

### Task 1: Add PyYAML as a dev dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `pyyaml` to the dev group**

In `pyproject.toml`, change:

```toml
[dependency-groups]
dev = [
  "pytest"
]
```

to:

```toml
[dependency-groups]
dev = [
  "pytest",
  "pyyaml"
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync --dev`
Expected: resolves and installs `pyyaml` (and updates `uv.lock`).

- [ ] **Step 3: Verify the import is available**

Run: `uv run python -c "import yaml; print(yaml.__version__)"`
Expected: prints a version string (e.g. `6.0.2`) with no `ModuleNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pyyaml dev dependency for cloud-init parse tests"
```

---

### Task 2: Cloud-init template + `render_cloud_init` (substitution)

**Files:**
- Create: `scripts/cloud-init-gpu.yaml.tmpl`
- Create: `src/ml_lab/gpu/cloud_init.py`
- Test: `tests/test_gpu_cloud_init.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_cloud_init.py`:

```python
import yaml

from ml_lab.gpu import cloud_init
from ml_lab.gpu.constants import (
    DEFAULT_TTL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SMOKE_MODEL_ID,
)

FAKE_TOKEN = "dop_v1_FAKEfake0123456789"


def _write_file(doc, path):
    for wf in doc["write_files"]:
        if wf["path"] == path:
            return wf["content"]
    raise AssertionError(f"{path} not in write_files")


def _run_env(doc):
    return _write_file(doc, "/etc/ml-lab/run.env")


def test_render_substitutes_all_placeholders_and_parses():
    out = cloud_init.render_cloud_init(run_id="20260712-abc123", destroy_token=FAKE_TOKEN)
    assert "@@" not in out  # every placeholder filled
    doc = yaml.safe_load(out)  # structural check: valid cloud-config mapping
    assert isinstance(doc, dict)
    assert doc["package_update"] is False
    assert "packages" not in doc
    run_env = _run_env(doc)
    assert "RUN_ID=20260712-abc123" in run_env
    assert f"TTL_SECONDS={DEFAULT_TTL_SECONDS}" in run_env
    assert f"SMOKE_MODEL_ID={SMOKE_MODEL_ID}" in run_env
    assert f"DO_DROPLET_DESTROY_TOKEN={FAKE_TOKEN}" in run_env


def test_render_wires_timer_from_params():
    out = cloud_init.render_cloud_init(
        run_id="r", destroy_token=FAKE_TOKEN, ttl_seconds=900, self_destruct_retry_seconds=120
    )
    timer = _write_file(yaml.safe_load(out), "/etc/systemd/system/ml-lab-self-destruct.timer")
    assert "OnBootSec=900" in timer
    assert "OnUnitActiveSec=120" in timer


def test_render_defaults_timer_from_constants():
    out = cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
    timer = _write_file(yaml.safe_load(out), "/etc/systemd/system/ml-lab-self-destruct.timer")
    assert f"OnBootSec={DEFAULT_TTL_SECONDS}" in timer
    assert f"OnUnitActiveSec={SELF_DESTRUCT_RETRY_SECONDS}" in timer


def test_render_preserves_bash_expansions():
    out = cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
    # bash ${...} in self_destruct.sh must survive the @@...@@ substitution untouched
    assert "${DROPLET_ID}" in out
    assert "${DO_DROPLET_DESTROY_TOKEN}" in out


def test_token_appears_only_once_and_in_run_env():
    out = cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
    assert out.count(FAKE_TOKEN) == 1  # secret is not duplicated across the file
    assert FAKE_TOKEN in _run_env(yaml.safe_load(out))


def test_runcmd_arms_timer_before_install_and_marks_ready_last():
    doc = yaml.safe_load(cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN))
    cmds = [str(c) for c in doc["runcmd"]]
    enable_idx = next(i for i, c in enumerate(cmds) if "enable --now ml-lab-self-destruct.timer" in c)
    pip_idx = next(i for i, c in enumerate(cmds) if "pip install vllm" in c)
    ready_idx = next(i for i, c in enumerate(cmds) if "bootstrap-ready.json" in c)
    assert enable_idx < pip_idx < ready_idx  # arm before slow install; mark ready last
    active_idxs = [i for i, c in enumerate(cmds) if "is-active --quiet ml-lab-self-destruct.timer" in c]
    assert any(i < enable_idx + 2 for i in active_idxs)  # verified right after enable
    assert any(i > pip_idx for i in active_idxs)  # re-verified before the ready marker
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_cloud_init.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.cloud_init'`.

- [ ] **Step 3: Create the cloud-init template**

Create `scripts/cloud-init-gpu.yaml.tmpl` with exactly this content:

```yaml
#cloud-config
package_update: false
write_files:
  - path: /etc/ml-lab/run.env
    permissions: "0600"
    content: |
      RUN_ID=@@RUN_ID@@
      TTL_SECONDS=@@TTL_SECONDS@@
      SELF_DESTRUCT_RETRY_SECONDS=@@SELF_DESTRUCT_RETRY_SECONDS@@
      SMOKE_MODEL_ID=@@SMOKE_MODEL_ID@@
      DO_DROPLET_DESTROY_TOKEN=@@DESTROY_TOKEN@@
  - path: /opt/ml-lab/self_destruct.sh
    permissions: "0700"
    content: |
      #!/usr/bin/env bash
      set -euo pipefail
      source /etc/ml-lab/run.env
      DROPLET_ID="$(curl -fsS http://169.254.169.254/metadata/v1/id)"
      curl -fsS -X DELETE \
        -H "Authorization: Bearer ${DO_DROPLET_DESTROY_TOKEN}" \
        "https://api.digitalocean.com/v2/droplets/${DROPLET_ID}"
  - path: /etc/systemd/system/ml-lab-self-destruct.service
    permissions: "0644"
    content: |
      [Unit]
      Description=Destroy ML lab GPU droplet after TTL
      [Service]
      Type=oneshot
      ExecStart=/opt/ml-lab/self_destruct.sh
  - path: /etc/systemd/system/ml-lab-self-destruct.timer
    permissions: "0644"
    content: |
      [Unit]
      Description=Recurring self-destruct timer for ML lab GPU droplet
      [Timer]
      OnBootSec=@@TTL_SECONDS@@
      OnUnitActiveSec=@@SELF_DESTRUCT_RETRY_SECONDS@@
      [Install]
      WantedBy=timers.target
runcmd:
  - systemctl daemon-reload
  - systemctl enable --now ml-lab-self-destruct.timer
  - systemctl is-active --quiet ml-lab-self-destruct.timer
  - mkdir -p /opt/ml-lab/artifacts
  - nvidia-smi > /opt/ml-lab/artifacts/nvidia-smi.txt 2>&1 || true
  - python3 -m venv /opt/ml-lab/venv
  - /opt/ml-lab/venv/bin/python -m pip install --upgrade pip
  - /opt/ml-lab/venv/bin/python -m pip install vllm
  - systemctl is-active --quiet ml-lab-self-destruct.timer
  - echo '{"ready":true,"self_destruct_timer_active":true}' > /opt/ml-lab/bootstrap-ready.json
```

> **Note — no spaces after the JSON colons.** The bootstrap marker is written as `{"ready":true,"self_destruct_timer_active":true}` (no space after `:`). A `:` followed by a space would make YAML read the `runcmd` entry as a mapping and the whole document fails `yaml.safe_load`. The space-free form is a legal YAML plain scalar and is semantically identical JSON (`json.loads` yields `{"ready": True, "self_destruct_timer_active": True}`), so S3b's `wait_for_bootstrap` — which will parse the file as JSON, not string-match it — is unaffected.

- [ ] **Step 4: Create `render_cloud_init` (substitution only)**

Create `src/ml_lab/gpu/cloud_init.py`:

```python
"""Render the self-destruct cloud-init into the user_data string create_lab_droplet requires.

Pure string templating over scripts/cloud-init-gpu.yaml.tmpl using collision-proof
@@PLACEHOLDER@@ markers, so the template's literal bash ${...} survives untouched.
No network, no droplet — S3b supplies the caller (env token + create_lab_droplet).
"""

from __future__ import annotations

from pathlib import Path

from ml_lab.gpu.constants import (
    DEFAULT_TTL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SMOKE_MODEL_ID,
)

# src/ml_lab/gpu/cloud_init.py -> parents[3] is the repo root; scripts/ lives there.
_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "cloud-init-gpu.yaml.tmpl"


def render_cloud_init(
    *,
    run_id: str,
    destroy_token: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    self_destruct_retry_seconds: int = SELF_DESTRUCT_RETRY_SECONDS,
    smoke_model_id: str = SMOKE_MODEL_ID,
) -> str:
    """Render the cloud-init user_data string. destroy_token lands only in run.env (0600)."""
    template = _TEMPLATE_PATH.read_text()
    return (
        template.replace("@@RUN_ID@@", run_id)
        .replace("@@TTL_SECONDS@@", str(ttl_seconds))
        .replace("@@SELF_DESTRUCT_RETRY_SECONDS@@", str(self_destruct_retry_seconds))
        .replace("@@SMOKE_MODEL_ID@@", smoke_model_id)
        .replace("@@DESTROY_TOKEN@@", destroy_token)
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_cloud_init.py -v`
Expected: PASS — all six render tests green.

- [ ] **Step 6: Commit**

```bash
git add scripts/cloud-init-gpu.yaml.tmpl src/ml_lab/gpu/cloud_init.py tests/test_gpu_cloud_init.py
git commit -m "feat: render_cloud_init + self-destruct cloud-init template (placeholder substitution)"
```

---

### Task 3: Fail-loud guards (empty token, unfilled placeholder)

**Files:**
- Modify: `src/ml_lab/gpu/cloud_init.py`
- Test: `tests/test_gpu_cloud_init.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_cloud_init.py`:

```python
import pytest


def test_render_raises_on_empty_token():
    with pytest.raises(ValueError, match="destroy_token"):
        cloud_init.render_cloud_init(run_id="r", destroy_token="")


def test_render_raises_on_leftover_placeholder(monkeypatch, tmp_path):
    bad = tmp_path / "bad.tmpl"
    bad.write_text("RUN_ID=@@RUN_ID@@\nMYSTERY=@@MYSTERY@@\n")
    monkeypatch.setattr(cloud_init, "_TEMPLATE_PATH", bad)
    with pytest.raises(ValueError, match="placeholder"):
        cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_cloud_init.py -k "empty_token or leftover" -v`
Expected: FAIL — the empty-token render returns a string instead of raising, and the leftover-placeholder render returns text containing `@@MYSTERY@@` instead of raising.

- [ ] **Step 3: Add the guards**

In `src/ml_lab/gpu/cloud_init.py`, add the `re` import and a placeholder pattern below the existing imports:

```python
from __future__ import annotations

import re
from pathlib import Path
```

Add, just under `_TEMPLATE_PATH`:

```python
_PLACEHOLDER_RE = re.compile(r"@@[A-Z_]+@@")
```

Then change the body of `render_cloud_init` — add the empty-token check at the top and the leftover check before the return:

```python
    if not destroy_token:
        raise ValueError("destroy_token is required (self-destruct cannot arm without it)")
    template = _TEMPLATE_PATH.read_text()
    rendered = (
        template.replace("@@RUN_ID@@", run_id)
        .replace("@@TTL_SECONDS@@", str(ttl_seconds))
        .replace("@@SELF_DESTRUCT_RETRY_SECONDS@@", str(self_destruct_retry_seconds))
        .replace("@@SMOKE_MODEL_ID@@", smoke_model_id)
        .replace("@@DESTROY_TOKEN@@", destroy_token)
    )
    leftover = _PLACEHOLDER_RE.findall(rendered)
    if leftover:
        raise ValueError(f"unfilled cloud-init placeholders: {sorted(set(leftover))}")
    return rendered
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_cloud_init.py -v`
Expected: PASS — the two guard tests plus all Task 2 render tests.

- [ ] **Step 5: Confirm the full suite stays green**

Run: `uv run pytest -q`
Expected: PASS — the existing 105 S1+S2 tests plus the new S3a tests.

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/cloud_init.py tests/test_gpu_cloud_init.py
git commit -m "feat: fail-loud cloud-init guards (empty token, unfilled placeholder)"
```

---

## Definition of Done

- `uv run pytest` is fully green (existing S1+S2 tests plus the new S3a tests).
- `pyyaml` is a dev-only dependency; the renderer itself does no YAML parsing.
- `scripts/cloud-init-gpu.yaml.tmpl` exists with `@@...@@` placeholders only (no real token in git), `package_update: false`, no `packages:` block, the self-destruct service/timer units, and the `runcmd` that arms the timer first and writes `bootstrap-ready.json` last.
- `src/ml_lab/gpu/cloud_init.py` exposes `render_cloud_init(*, run_id, destroy_token, ttl_seconds=…, self_destruct_retry_seconds=…, smoke_model_id=…) -> str`.
- Render substitutes every placeholder, leaves bash `${...}` untouched, wires `OnBootSec`/`OnUnitActiveSec` from the params, and places the token only in `run.env` (once).
- `render_cloud_init` raises `ValueError` on an empty `destroy_token` or any unfilled `@@...@@` placeholder.
- Nothing calls the renderer yet (library slice); no S3a code path reaches the network or creates a droplet.
- Out of scope and untouched: env-token reading, `create_lab_droplet` wiring, SSH/bootstrap orchestration, `gpu-up`/`gpu-run`, `gpu_benchmark.py`, Spaces upload (S3b/S3c/S4).
```