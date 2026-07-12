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
