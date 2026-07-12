import types

import pytest

from ml_lab.gpu import do_client
from ml_lab.gpu.do_client import DOClientError


def _completed(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_is_lab_droplet_matches_tag_or_name_prefix():
    assert do_client._is_lab_droplet({"name": "x", "tags": ["ml-lab"]}) is True
    assert do_client._is_lab_droplet({"name": "ml-lab-gpu-phase0-1", "tags": []}) is True
    assert do_client._is_lab_droplet({"name": "unrelated", "tags": []}) is False


def test_list_lab_droplets_filters(monkeypatch):
    payload = (
        '[{"id":1,"name":"ml-lab-gpu-phase0-a","tags":["ml-lab"]},'
        '{"id":2,"name":"ml-lab-gpu-phase0-b","tags":[]},'
        '{"id":3,"name":"someone-else","tags":["other"]}]'
    )
    monkeypatch.setattr(
        do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload)
    )
    result = do_client.list_lab_droplets()
    assert [d["id"] for d in result] == [1, 2]


def test_get_droplet_returns_dict(monkeypatch):
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(stdout='[{"id":7,"status":"active"}]'),
    )
    assert do_client.get_droplet(7) == {"id": 7, "status": "active"}


def test_get_droplet_absent_returns_none(monkeypatch):
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: GET ... 404 not found"),
    )
    assert do_client.get_droplet(7) is None


def test_get_droplet_other_error_raises(monkeypatch):
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stderr="Error: 500 server error"),
    )
    with pytest.raises(DOClientError):
        do_client.get_droplet(7)


def test_destroy_droplet_status_mapping(monkeypatch):
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(returncode=0))
    assert do_client.destroy_droplet(7) == "accepted"

    monkeypatch.setattr(
        do_client.subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="404 not found")
    )
    assert do_client.destroy_droplet(7) == "gone"

    monkeypatch.setattr(
        do_client.subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="500 boom")
    )
    assert do_client.destroy_droplet(7) == "error"


def test_probe_destroy_token_ok_on_404(monkeypatch):
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")
    monkeypatch.setattr(
        do_client.requests, "delete", lambda *a, **k: types.SimpleNamespace(status_code=404)
    )
    do_client.probe_destroy_token()  # no raise


@pytest.mark.parametrize("code", [401, 403, 204])
def test_probe_destroy_token_bad_status_raises(monkeypatch, code):
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")
    monkeypatch.setattr(
        do_client.requests, "delete", lambda *a, **k: types.SimpleNamespace(status_code=code)
    )
    with pytest.raises(DOClientError):
        do_client.probe_destroy_token()


def test_probe_destroy_token_missing_env_raises(monkeypatch):
    monkeypatch.delenv("DO_DROPLET_DESTROY_TOKEN", raising=False)
    with pytest.raises(DOClientError):
        do_client.probe_destroy_token()
