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
    ACCEPTABLE_SIZE_SLUGS,
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
    """Preflight that the acceptable size slugs + image exist in the account.

    Region/capacity is NOT gated here: DO GPU capacity shifts between regions AND SKUs
    within minutes, so the create SKU+region are resolved live at create time (the
    multi-SKU retry loop) instead of being statically pinned. Where a size exists but
    is not actually creatable, the create call's DO 422 (pre-billing) is the authority.
    """
    known = {s["slug"] for s in do_client.list_sizes()}
    missing = [slug for slug in ACCEPTABLE_SIZE_SLUGS if slug not in known]
    if missing:
        raise ConstantsError(f"size(s) not found in account: {missing}")
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
    """Create the droplet, retrying across SKUs AND regions as GPU capacity flips.

    Single-GPU capacity on DO flaps across both regions and SKUs within minutes, so we
    try each interchangeable SKU in ACCEPTABLE_SIZE_SLUGS (Hopper first), and for each
    the region DO currently reports. Capacity can also vanish between the resolve and
    the create POST (a 422 "not available in this region"); on that 422 we exclude that
    (SKU, region) pair and move on. When nothing has capacity we re-probe on a bounded
    poll (windows reopen within minutes). A non-capacity error is re-raised immediately.
    Returns (size, region, droplet). All 422s are pre-billing.
    """
    tried: set[tuple[str, str]] = set()  # (size_slug, region) pairs already 422'd
    start = time.monotonic()
    while True:
        candidate = None
        for size_slug in ACCEPTABLE_SIZE_SLUGS:
            excluded = {r for (s, r) in tried if s == size_slug}
            region = resolve_region(size_slug, exclude=excluded)
            if region is not None:
                candidate = (size_slug, region)
                break
        if candidate is None:
            # Nothing has capacity right now — re-probe until a window opens or timeout.
            if time.monotonic() - start >= CREATE_CAPACITY_WAIT_SECONDS:
                raise ConstantsError(
                    f"no capacity for any of {ACCEPTABLE_SIZE_SLUGS} after "
                    f"{CREATE_CAPACITY_WAIT_SECONDS}s (tried {sorted(tried) or 'none'})"
                )
            tried.clear()
            time.sleep(CREATE_CAPACITY_POLL_INTERVAL_SECONDS)
            continue
        size_slug, region = candidate
        try:
            droplet = do_client.create_droplet(
                name=name,
                region=region,
                size=size_slug,
                image=DO_IMAGE_SLUG,
                tags=tags,
                user_data=user_data,
                ssh_key_ids=ssh_key_ids,
            )
            return size_slug, region, droplet
        except do_client.DOClientError as exc:
            if _SIZE_UNAVAILABLE_IN_REGION not in str(exc):
                raise
            print(f"  {size_slug} in {region} has no capacity; trying next…", flush=True)
            tried.add((size_slug, region))


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
    size, region, droplet = _create_with_region_retry(name, tags, user_data, ssh_key_ids)
    print(
        f"Created droplet:\n  id: {droplet['id']}\n  name: {name}\n"
        f"  size: {size}\n  region: {region}\n  local_pid: {os.getpid()}",
        flush=True,
    )
    return {"id": droplet["id"], "name": name, "run_id": run_id, "size": size, "region": region}
