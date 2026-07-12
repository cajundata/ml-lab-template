"""The shared GPU-droplet create path: preflight-gated, atomic-tagged create.

Both gpu-run and gpu-up (S3) call create_lab_droplet, so every safety gate lives
inside it. All DigitalOcean access goes through the mocked do_client seam.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from ml_lab.gpu import do_client
from ml_lab.gpu.constants import BASE_TAGS, DO_IMAGE_SLUG, DO_REGION, DO_SIZE_SLUG


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
