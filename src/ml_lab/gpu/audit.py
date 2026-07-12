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
