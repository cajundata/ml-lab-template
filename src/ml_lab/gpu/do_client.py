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
