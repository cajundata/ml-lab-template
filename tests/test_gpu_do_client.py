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


def test_get_droplet_absent_via_stdout_json_returns_none(monkeypatch):
    # Real doctl behavior with `-o json`: a 404 error is emitted as JSON on STDOUT
    # (exit 1), stderr empty. Absence detection must look at stdout, not just stderr.
    payload = (
        '{"errors":[{"detail":"GET https://api.digitalocean.com/v2/droplets/7: '
        '404 (request \\"x\\") The resource you were accessing could not be found."}]}'
    )
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(returncode=1, stdout=payload, stderr=""),
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

    # Real doctl `-o json`: the 404 is on STDOUT (exit 1), stderr empty.
    monkeypatch.setattr(
        do_client.subprocess,
        "run",
        lambda *a, **k: _completed(
            returncode=1, stdout='{"errors":[{"detail":"... 404 ... not be found."}]}', stderr=""
        ),
    )
    assert do_client.destroy_droplet(7) == "gone"


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


def test_list_region_slugs_returns_available_only(monkeypatch):
    payload = '[{"slug":"nyc2","available":true},{"slug":"sfo1","available":false}]'
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload))
    assert do_client.list_region_slugs() == ["nyc2"]


def test_list_sizes_returns_raw_dicts(monkeypatch):
    payload = '[{"slug":"gpu-4000adax1-20gb","regions":["nyc2"]}]'
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload))
    sizes = do_client.list_sizes()
    assert sizes[0]["slug"] == "gpu-4000adax1-20gb"
    assert sizes[0]["regions"] == ["nyc2"]


def test_list_image_slugs_filters_null_slugs(monkeypatch):
    payload = '[{"slug":"gpu-h100x1-base"},{"slug":null},{"slug":"other"}]'
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=payload))
    assert do_client.list_image_slugs() == ["gpu-h100x1-base", "other"]


def test_create_droplet_builds_argv_and_parses_id(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(stdout='[{"id": 42, "name": "ml-lab-gpu-phase0-x", "status": "new"}]')

    monkeypatch.setattr(do_client.subprocess, "run", fake_run)
    droplet = do_client.create_droplet(
        name="ml-lab-gpu-phase0-x",
        region="nyc2",
        size="gpu-4000adax1-20gb",
        image="gpu-h100x1-base",
        tags=["ml-lab", "run-x"],
        user_data="#cloud-config\n",
        ssh_key_ids=["aa:bb"],
    )
    assert droplet["id"] == 42
    argv = captured["argv"]
    assert argv[:4] == ["doctl", "compute", "droplet", "create"]
    assert "ml-lab-gpu-phase0-x" in argv
    for flag, val in [
        ("--region", "nyc2"),
        ("--size", "gpu-4000adax1-20gb"),
        ("--image", "gpu-h100x1-base"),
    ]:
        assert flag in argv and argv[argv.index(flag) + 1] == val
    assert "--tag-names" in argv and argv[argv.index("--tag-names") + 1] == "ml-lab,run-x"
    assert "--user-data-file" in argv
    assert "--ssh-keys" in argv and argv[argv.index("--ssh-keys") + 1] == "aa:bb"
    assert "--wait" not in argv
    assert argv[-2:] == ["-o", "json"]


def test_create_droplet_omits_ssh_keys_when_none(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(stdout='[{"id": 7, "name": "n", "status": "new"}]')

    monkeypatch.setattr(do_client.subprocess, "run", fake_run)
    do_client.create_droplet(
        name="n", region="nyc2", size="s", image="i", tags=["ml-lab"], user_data="#cloud-config\n"
    )
    assert "--ssh-keys" not in captured["argv"]


def test_create_droplet_raises_when_doctl_returns_nothing(monkeypatch):
    monkeypatch.setattr(do_client.subprocess, "run", lambda *a, **k: _completed(stdout=""))
    with pytest.raises(do_client.DOClientError):
        do_client.create_droplet(
            name="n", region="r", size="s", image="i", tags=["ml-lab"], user_data="#cloud-config\n"
        )


def test_public_ipv4_extracts_public_address():
    droplet = {
        "networks": {
            "v4": [
                {"type": "private", "ip_address": "10.0.0.1"},
                {"type": "public", "ip_address": "1.2.3.4"},
            ]
        }
    }
    assert do_client.public_ipv4(droplet) == "1.2.3.4"


def test_public_ipv4_none_when_no_public():
    private_only = {"networks": {"v4": [{"type": "private", "ip_address": "10.0.0.1"}]}}
    assert do_client.public_ipv4(private_only) is None
    assert do_client.public_ipv4({}) is None


def test_run_doctl_wraps_calledprocesserror(monkeypatch):
    def boom(*a, **k):
        raise do_client.subprocess.CalledProcessError(1, "doctl", stderr="unauthorized")

    monkeypatch.setattr(do_client.subprocess, "run", boom)
    with pytest.raises(DOClientError):
        do_client._run_doctl(["compute", "droplet", "list"])
