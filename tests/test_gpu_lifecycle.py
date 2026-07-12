import pytest

from ml_lab.gpu import do_client, gpu_env, lifecycle
from ml_lab.gpu.create import LabDropletExistsError
from ml_lab.gpu.remote import RemoteError
from ml_lab.gpu.teardown import TeardownError


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_wait_for_public_ip_returns_once_networks_populate(monkeypatch):
    seq = iter(
        [
            {"id": 1},  # no networks yet
            {"id": 1, "networks": {"v4": [{"type": "public", "ip_address": "5.6.7.8"}]}},
        ]
    )
    monkeypatch.setattr(lifecycle.do_client, "get_droplet", lambda i: next(seq))
    clock = _FakeClock()
    ip = lifecycle.wait_for_public_ip(1, timeout=1000, now=clock.now, sleep=clock.sleep)
    assert ip == "5.6.7.8"


def test_wait_for_public_ip_times_out(monkeypatch):
    monkeypatch.setattr(lifecycle.do_client, "get_droplet", lambda i: {"id": 1})  # never gets IP
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        lifecycle.wait_for_public_ip(1, timeout=30, now=clock.now, sleep=clock.sleep)


def test_wait_for_public_ip_tolerates_transient_do_error(monkeypatch):
    seq = iter([1, 2])  # first call raises, second returns a droplet with an IP

    def flaky(i):
        if next(seq) == 1:
            raise do_client.DOClientError("transient")
        return {"networks": {"v4": [{"type": "public", "ip_address": "9.9.9.9"}]}}

    monkeypatch.setattr(lifecycle.do_client, "get_droplet", flaky)
    clock = _FakeClock()
    ip = lifecycle.wait_for_public_ip(1, timeout=1000, now=clock.now, sleep=clock.sleep)
    assert ip == "9.9.9.9"


def _env():
    return gpu_env.GpuEnv(destroy_token="tok", ssh_key_ids=["k1"], ssh_key_path="/key")


def _stub_waits_ok(monkeypatch):
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")
    monkeypatch.setattr(lifecycle, "wait_for_ssh", lambda ip, **k: None)
    monkeypatch.setattr(
        lifecycle,
        "wait_for_bootstrap",
        lambda ip, **k: {"ready": True, "self_destruct_timer_active": True},
    )


def test_gpu_up_happy_leaves_droplet_alive(monkeypatch, capsys):
    destroyed = {}
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet",
        lambda ud, **k: {"id": 42, "name": "ml-lab-gpu-phase0-r", "run_id": "r"},
    )
    _stub_waits_ok(monkeypatch)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    result = lifecycle.gpu_up(env=_env(), now=1000.0)
    assert result["id"] == 42 and result["ip"] == "1.2.3.4"
    assert "id" not in destroyed  # alive on verified success
    out = capsys.readouterr().out
    assert "verified active" in out
    assert "make gpu-down DROPLET_ID=42" in out


def test_gpu_up_ssh_timeout_destroys_and_reraises(monkeypatch):
    destroyed = {}
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet", lambda ud, **k: {"id": 42, "name": "n", "run_id": "r"}
    )
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")

    def boom(ip, **k):
        raise RemoteError("unreachable")

    monkeypatch.setattr(lifecycle, "wait_for_ssh", boom)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(RemoteError):
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert destroyed["id"] == 42


def test_gpu_up_keyboardinterrupt_after_create_destroys(monkeypatch):
    destroyed = {}
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet", lambda ud, **k: {"id": 42, "name": "n", "run_id": "r"}
    )
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")

    def interrupt(ip, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(lifecycle, "wait_for_ssh", interrupt)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(KeyboardInterrupt):
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert destroyed["id"] == 42


def test_gpu_up_preflight_failure_does_not_destroy(monkeypatch):
    destroyed = {}

    def refuse(ud, **k):
        raise LabDropletExistsError("exists")

    monkeypatch.setattr(lifecycle, "create_lab_droplet", refuse)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: destroyed.setdefault("id", did))
    with pytest.raises(LabDropletExistsError):
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert "id" not in destroyed  # no droplet existed → nothing to destroy


def test_gpu_up_threads_ttl_and_enforce_budget(monkeypatch):
    seen = {}

    def capture(ud, **k):
        seen.update(k)
        return {"id": 42, "name": "n", "run_id": "r"}

    monkeypatch.setattr(lifecycle, "create_lab_droplet", capture)
    _stub_waits_ok(monkeypatch)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", lambda did: None)
    lifecycle.gpu_up(env=_env(), ttl_seconds=900, enforce_budget=False, now=1000.0)
    assert seen["ttl_seconds"] == 900
    assert seen["enforce_budget"] is False
    assert seen["ssh_key_ids"] == ["k1"]


def test_gpu_up_teardown_failure_propagates_and_chains_original(monkeypatch):
    monkeypatch.setattr(
        lifecycle, "create_lab_droplet", lambda ud, **k: {"id": 42, "name": "n", "run_id": "r"}
    )
    monkeypatch.setattr(lifecycle, "wait_for_public_ip", lambda did, **k: "1.2.3.4")

    def ssh_boom(ip, **k):
        raise RemoteError("unreachable")

    def teardown_boom(did):
        raise TeardownError("droplet still present")

    monkeypatch.setattr(lifecycle, "wait_for_ssh", ssh_boom)
    monkeypatch.setattr(lifecycle, "destroy_and_verify", teardown_boom)
    # A failed teardown is the louder alarm: TeardownError propagates, original chained.
    with pytest.raises(TeardownError) as exc:
        lifecycle.gpu_up(env=_env(), now=1000.0)
    assert isinstance(exc.value.__context__, RemoteError)
