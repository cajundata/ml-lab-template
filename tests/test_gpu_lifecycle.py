import pytest

from ml_lab.gpu import do_client, lifecycle
from ml_lab.gpu.remote import RemoteError


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
