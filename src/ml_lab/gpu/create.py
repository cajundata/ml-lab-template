"""The shared GPU-droplet create path: preflight-gated, atomic-tagged create.

Both gpu-run and gpu-up (S3) call create_lab_droplet, so every safety gate lives
inside it. All DigitalOcean access goes through the mocked do_client seam.
"""

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


def generate_run_id(now: float) -> str:
    """Return 'YYYYMMDD-<6 hex>' — date from now (UTC) + random suffix for uniqueness."""
    date = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d")
    return f"{date}-{secrets.token_hex(3)}"


def build_tags(run_id: str, ttl_epoch: int) -> list[str]:
    return [*BASE_TAGS, f"run-{run_id}", f"ttl-expiry-{ttl_epoch}"]


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
