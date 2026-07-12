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
