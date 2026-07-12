import subprocess
import types

import pytest

from ml_lab.gpu import benchmark
from ml_lab.gpu.constants import BENCHMARK_TIMEOUT_SECONDS
from ml_lab.gpu.remote import RemoteError


def _completed(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_deliver_and_run_scps_script_then_runs_it(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        benchmark.remote, "scp_up",
        lambda host, local, remote, **k: calls.update(scp=(host, local, remote)),
    )

    def fake_run_ssh(host, argv, *, key_path, timeout):
        calls["ssh"] = {"host": host, "argv": argv, "timeout": timeout}
        return _completed(returncode=0)

    monkeypatch.setattr(benchmark.remote, "_run_ssh", fake_run_ssh)

    code = benchmark.deliver_and_run_benchmark(
        "1.2.3.4", "R1", key_path="/key", smoke_model_id="my/model"
    )
    assert code == 0
    # Script was scp'd up to the remote benchmark path.
    assert calls["scp"] == ("1.2.3.4", str(benchmark.BENCHMARK_SCRIPT_PATH),
                            benchmark.REMOTE_BENCHMARK_PATH)
    # Run command uses the venv python, the remote script, and the two args.
    argv = calls["ssh"]["argv"]
    assert argv[0] == benchmark.REMOTE_VENV_PYTHON
    assert argv[1] == benchmark.REMOTE_BENCHMARK_PATH
    assert argv[argv.index("--run-id") + 1] == "R1"
    assert argv[argv.index("--smoke-model-id") + 1] == "my/model"
    assert argv[argv.index("--artifacts-root") + 1] == benchmark.REMOTE_ARTIFACTS_ROOT
    assert calls["ssh"]["timeout"] == BENCHMARK_TIMEOUT_SECONDS


def test_deliver_and_run_returns_nonzero_code_without_raising(monkeypatch):
    monkeypatch.setattr(benchmark.remote, "scp_up", lambda *a, **k: None)
    monkeypatch.setattr(benchmark.remote, "_run_ssh", lambda *a, **k: _completed(returncode=1))
    # A completed-but-failed benchmark is a returned code, not an exception.
    assert benchmark.deliver_and_run_benchmark("1.2.3.4", "R1", key_path="/key") == 1


def test_deliver_and_run_raises_on_ssh_timeout(monkeypatch):
    monkeypatch.setattr(benchmark.remote, "scp_up", lambda *a, **k: None)

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=BENCHMARK_TIMEOUT_SECONDS)

    monkeypatch.setattr(benchmark.remote, "_run_ssh", boom)
    with pytest.raises(RemoteError):
        benchmark.deliver_and_run_benchmark("1.2.3.4", "R1", key_path="/key")


def test_deliver_and_run_propagates_scp_failure(monkeypatch):
    def boom(*a, **k):
        raise RemoteError("scp up failed")

    monkeypatch.setattr(benchmark.remote, "scp_up", boom)
    monkeypatch.setattr(benchmark.remote, "_run_ssh", lambda *a, **k: _completed(returncode=0))
    with pytest.raises(RemoteError):
        benchmark.deliver_and_run_benchmark("1.2.3.4", "R1", key_path="/key")


def test_pull_artifacts_pulls_bundle_and_cloud_init_log(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        benchmark.remote, "scp_down",
        lambda host, remote, local, **k: calls.append((remote, local, k.get("recursive", False))),
    )
    dest = benchmark.pull_artifacts("1.2.3.4", "R1", key_path="/key", dest_root=str(tmp_path))
    assert dest == tmp_path / "R1"
    # First call: recursive pull of the remote bundle into dest_root.
    assert calls[0] == (f"{benchmark.REMOTE_ARTIFACTS_ROOT}/R1", str(tmp_path), True)
    # Second call: the cloud-init log renamed to bootstrap.log inside the run dir.
    assert calls[1] == (benchmark.CLOUD_INIT_LOG, str(tmp_path / "R1" / "bootstrap.log"), False)


def test_pull_artifacts_raises_on_scp_failure(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RemoteError("scp down failed")

    monkeypatch.setattr(benchmark.remote, "scp_down", boom)
    with pytest.raises(RemoteError):
        benchmark.pull_artifacts("1.2.3.4", "R1", key_path="/key", dest_root=str(tmp_path))
