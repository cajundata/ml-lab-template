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
