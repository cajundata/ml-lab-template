import subprocess
import types

import pytest

from ml_lab.gpu import remote
from ml_lab.gpu.remote import RemoteError


def _completed(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeClock:
    """Deterministic monotonic clock: sleep() advances virtual time."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_wait_for_ssh_returns_once_reachable(monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, *a, **k):
        calls["n"] += 1
        return _completed(returncode=0 if calls["n"] >= 3 else 255)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    clock = _FakeClock()
    remote.wait_for_ssh("1.2.3.4", key_path="/key", timeout=1000, now=clock.now, sleep=clock.sleep)
    assert calls["n"] == 3  # two failures, third attempt succeeds


def test_wait_for_ssh_times_out(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(returncode=255))
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_ssh("1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep)


def test_wait_for_ssh_treats_attempt_timeout_as_not_ready(monkeypatch):
    def timeout_then_never(argv, *a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(remote.subprocess, "run", timeout_then_never)
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_ssh("1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep)


def test_run_ssh_builds_argv(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(returncode=0)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    remote._run_ssh("1.2.3.4", ["true"], key_path="/key", timeout=30)
    argv = captured["argv"]
    assert argv[0] == "ssh"
    assert "-i" in argv and argv[argv.index("-i") + 1] == "/key"
    assert "root@1.2.3.4" in argv
    assert argv[-1] == "true"
    assert "BatchMode=yes" in argv
    assert "IdentitiesOnly=yes" in argv  # only the -i key authenticates, not agent keys
    assert "StrictHostKeyChecking=accept-new" in argv


READY_JSON = '{"ready":true,"self_destruct_timer_active":true}'


def test_wait_for_bootstrap_returns_marker_when_both_flags_true(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(stdout=READY_JSON))
    clock = _FakeClock()
    marker = remote.wait_for_bootstrap(
        "1.2.3.4", key_path="/key", timeout=1000, now=clock.now, sleep=clock.sleep
    )
    assert marker == {"ready": True, "self_destruct_timer_active": True}


def test_wait_for_bootstrap_polls_until_ready(monkeypatch):
    seq = iter(
        [
            _completed(returncode=1, stdout=""),  # file not there yet
            _completed(stdout='{"ready":true,"self_destruct_timer_active":false}'),  # timer not up
            _completed(stdout=READY_JSON),  # ready
        ]
    )
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: next(seq))
    clock = _FakeClock()
    marker = remote.wait_for_bootstrap(
        "1.2.3.4", key_path="/key", timeout=1000, now=clock.now, sleep=clock.sleep
    )
    assert marker["self_destruct_timer_active"] is True


def test_wait_for_bootstrap_times_out_when_timer_never_active(monkeypatch):
    stale = '{"ready":true,"self_destruct_timer_active":false}'
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(stdout=stale))
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_bootstrap(
            "1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep
        )


def test_wait_for_bootstrap_ignores_malformed_json(monkeypatch):
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: _completed(stdout="not json"))
    clock = _FakeClock()
    with pytest.raises(RemoteError):
        remote.wait_for_bootstrap(
            "1.2.3.4", key_path="/key", timeout=30, now=clock.now, sleep=clock.sleep
        )
