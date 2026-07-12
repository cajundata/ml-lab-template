import os
import signal
import subprocess
import sys
import time

import pytest

from ml_lab.gpu import audit, do_client, teardown
from ml_lab.gpu.audit import AuditReport, DropletInfo


def _info():
    return DropletInfo(
        id=1,
        name="ml-lab-gpu-phase0-x",
        status="active",
        region="atl1",
        size="gpu-rtx4000x1-20gb",
        image="img",
        public_ip="143.0.0.1",
        age="0h01m",
        ttl_expiry=None,
        overdue=False,
        destroy_command="make gpu-down DROPLET_ID=1",
    )


def _clean(monkeypatch):
    monkeypatch.setattr(audit, "collect_audit", lambda now=None: AuditReport())


def test_happy_path(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)  # no raise


def test_404_on_destroy_is_success(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "gone")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)  # no raise


def test_poll_then_absent(monkeypatch):
    seq = iter([{"id": 1}, {"id": 1}, None])
    calls = []

    def fake_get(i):
        calls.append(i)
        return next(seq)

    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", fake_get)
    monkeypatch.setattr(teardown.time, "sleep", lambda s: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)
    assert len(calls) == 3


def test_poll_timeout_raises(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: {"id": 1})
    monkeypatch.setattr(teardown.time, "sleep", lambda s: None)
    ticks = iter([0.0, 1.0, 700.0, 1400.0])
    monkeypatch.setattr(teardown.time, "monotonic", lambda: next(ticks))
    _clean(monkeypatch)
    with pytest.raises(teardown.TeardownError):
        teardown.destroy_and_verify(1)


def test_audit_dirty_after_absence_raises(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    monkeypatch.setattr(audit, "collect_audit", lambda now=None: AuditReport(droplets=[_info()]))
    with pytest.raises(teardown.TeardownError):
        teardown.destroy_and_verify(1)


def test_transient_get_error_then_absent_succeeds(monkeypatch):
    seq = iter([do_client.DOClientError("transient"), None])

    def fake_get(i):
        value = next(seq)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "error")
    monkeypatch.setattr(do_client, "get_droplet", fake_get)
    monkeypatch.setattr(teardown.time, "sleep", lambda s: None)
    _clean(monkeypatch)
    teardown.destroy_and_verify(1)  # no raise


def test_sigint_ignored_during_poll_and_restored(monkeypatch):
    observed = {}

    def fake_get(i):
        observed["handler"] = signal.getsignal(signal.SIGINT)
        return None

    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", fake_get)
    _clean(monkeypatch)
    before = signal.getsignal(signal.SIGINT)
    teardown.destroy_and_verify(1)
    assert observed["handler"] == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == before


def test_sigint_restored_after_failure(monkeypatch):
    monkeypatch.setattr(do_client, "destroy_droplet", lambda i: "accepted")
    monkeypatch.setattr(do_client, "get_droplet", lambda i: None)
    monkeypatch.setattr(audit, "collect_audit", lambda now=None: AuditReport(droplets=[_info()]))
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(teardown.TeardownError):
        teardown.destroy_and_verify(1)
    assert signal.getsignal(signal.SIGINT) == before


# --- Heavier: a real subprocess proves SIGINT cannot abort an armed teardown ---

_SIGINT_RUNNER = '''\
from ml_lab.gpu import teardown, do_client, audit

teardown.DESTROY_POLL_INTERVAL_SECONDS = 0.2

_calls = {"n": 0}


def _fake_get(droplet_id):
    _calls["n"] += 1
    if _calls["n"] > 5:
        return None
    return {"id": droplet_id, "status": "active"}


do_client.get_droplet = _fake_get
do_client.destroy_droplet = lambda droplet_id: "accepted"
audit.collect_audit = lambda now=None: audit.AuditReport()

teardown.destroy_and_verify(12345)
print("DONE", flush=True)
'''


def test_sigint_does_not_abort_teardown_subprocess(tmp_path):
    runner = tmp_path / "sigint_runner.py"
    runner.write_text(_SIGINT_RUNNER)
    env = {**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.Popen(
        [sys.executable, str(runner)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Wait for the arming banner so SIGINT lands AFTER SIG_IGN is installed.
    armed = False
    for _ in range(200):
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if "Teardown in progress" in line:
            armed = True
            break
    assert armed, "teardown never armed (banner not seen)"

    # Hammer SIGINT while the poll loop is running.
    for _ in range(3):
        proc.send_signal(signal.SIGINT)
        time.sleep(0.1)

    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"process died (rc={proc.returncode}); output:\n{out}"
    assert "DONE" in out
