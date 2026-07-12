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
    create.create_lab_droplet(
        "#cloud-config\n", ttl_seconds=3600, enforce_budget=False, run_id="r", now=1000.0
    )
    assert "ttl-expiry-4600" in calls["kwargs"]["tags"]


def test_create_lab_droplet_enforce_budget_true_raises(monkeypatch):
    _happy(monkeypatch)

    def boom(**k):
        raise AssertionError("create_droplet must not be called when the budget check fails")

    monkeypatch.setattr(do_client, "create_droplet", boom)
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
