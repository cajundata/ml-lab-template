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
    CREATE_CAPACITY_POLL_INTERVAL_SECONDS,
    CREATE_CAPACITY_WAIT_SECONDS,
    DEFAULT_TTL_SECONDS,
    DO_IMAGE_SLUG,
    DO_REGION,
    DO_SIZE_SLUG,
    NAME_FORMAT,
    validate_timeout_budget,
)

_SIZE_UNAVAILABLE_IN_REGION = "not available in this region"


class ConstantsError(RuntimeError):
    """A pinned region/size/image slug is not available in the account."""


def validate_constants() -> None:
    """Preflight the size + image exist in the account.

    Region is NOT gated here: DO GPU capacity shifts between regions within minutes,
    so the create region is resolved live at create time (resolve_region) instead of
    being statically pinned. Where a size exists but is not actually creatable, the
    create call's DO 422 (pre-billing) is the authority.
    """
    if next((s for s in do_client.list_sizes() if s["slug"] == DO_SIZE_SLUG), None) is None:
        raise ConstantsError(f"size {DO_SIZE_SLUG} not found")
    if DO_IMAGE_SLUG not in do_client.list_image_slugs():
        raise ConstantsError(f"image {DO_IMAGE_SLUG} not available in account")


def resolve_region(size_slug: str = DO_SIZE_SLUG, exclude: "set[str] | frozenset[str] | tuple" = ()) -> str | None:
    """Return the region to create `size_slug` in, from DO's live per-size regions.

    GPU capacity on DO shifts between regions within minutes, so the create region is
    resolved live rather than statically pinned. Prefer the configured DO_REGION when
    the size is currently available there; otherwise take the first region DO reports
    for the size. When DO reports no regions (null/empty — common for GPU SKUs), fall
    back to DO_REGION and let the create call (DO 422, pre-billing) be the authority.

    `exclude` names regions already tried and 422'd this create; they are skipped so
    the caller can retry across the volatility. Returns None when nothing is left to
    try (all reported regions excluded, and DO_REGION excluded / no fallback).
    """
    exclude = set(exclude)
    size = next((s for s in do_client.list_sizes() if s["slug"] == size_slug), None)
    if size is None:
        raise ConstantsError(f"size {size_slug} not found")
    regions = [r for r in (size.get("regions") or []) if r not in exclude]
    if regions:
        return DO_REGION if DO_REGION in regions else regions[0]
    if DO_REGION not in exclude:
        return DO_REGION
    return None


def generate_run_id(now: float) -> str:
    """Return 'YYYYMMDD-<6 hex>' — date from now (UTC) + random suffix for uniqueness."""
    date = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d")
    return f"{date}-{secrets.token_hex(3)}"


def build_tags(run_id: str, ttl_epoch: int) -> list[str]:
    return [*BASE_TAGS, f"run-{run_id}", f"ttl-expiry-{ttl_epoch}"]


class LabDropletExistsError(RuntimeError):
    """A matching lab GPU droplet already exists; refuse to create another."""


def _create_with_region_retry(name, tags, user_data, ssh_key_ids):
    """Create the droplet, retrying across regions as GPU capacity flips.

    DO reports a size's live regions, but capacity can vanish between the resolve and
    the create POST (a 422 "not available in this region"). On that 422 we exclude the
    region and try the next one DO reports; when no region has capacity we re-probe on
    a bounded poll (capacity windows reopen within minutes). A non-capacity error is
    re-raised immediately. Returns (region, droplet). All 422s are pre-billing.
    """
    tried: set[str] = set()
    start = time.monotonic()
    while True:
        region = resolve_region(DO_SIZE_SLUG, exclude=tried)
        if region is not None:
            try:
                droplet = do_client.create_droplet(
                    name=name,
                    region=region,
                    size=DO_SIZE_SLUG,
                    image=DO_IMAGE_SLUG,
                    tags=tags,
                    user_data=user_data,
                    ssh_key_ids=ssh_key_ids,
                )
                return region, droplet
            except do_client.DOClientError as exc:
                if _SIZE_UNAVAILABLE_IN_REGION not in str(exc):
                    raise
                print(f"  region {region} has no capacity; retrying another region…", flush=True)
                tried.add(region)
                continue
        # No region currently has capacity — re-probe until a window opens or we time out.
        if time.monotonic() - start >= CREATE_CAPACITY_WAIT_SECONDS:
            raise ConstantsError(
                f"no region has capacity for {DO_SIZE_SLUG} after "
                f"{CREATE_CAPACITY_WAIT_SECONDS}s (tried {sorted(tried) or 'none'})"
            )
        tried.clear()
        time.sleep(CREATE_CAPACITY_POLL_INTERVAL_SECONDS)


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
    region, droplet = _create_with_region_retry(name, tags, user_data, ssh_key_ids)
    print(
        f"Created droplet:\n  id: {droplet['id']}\n  name: {name}\n"
        f"  region: {region}\n  local_pid: {os.getpid()}",
        flush=True,
    )
    return {"id": droplet["id"], "name": name, "run_id": run_id, "region": region}
