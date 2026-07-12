"""The local benchmark driver: deliver the script, run it over SSH, pull the bundle.

Transport failures and the benchmark timeout raise RemoteError (S3c-2's gpu_run
finally then destroys, no pull). A completed-but-failed benchmark returns a nonzero
exit code so gpu_run can still pull the partial bundle before surfacing failure.
The remote/scp seam is mocked in every test — nothing here touches a live droplet.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ml_lab.gpu import remote
from ml_lab.gpu.constants import (
    BENCHMARK_TIMEOUT_SECONDS,
    SCP_TIMEOUT_SECONDS,
    SMOKE_MODEL_ID,
)
from ml_lab.gpu.remote import RemoteError

# Module-anchored local paths (parents[3] == repo root, same trick as cloud_init.py).
BENCHMARK_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "gpu_benchmark.py"
LOCAL_ARTIFACTS_ROOT = Path(__file__).resolve().parents[3] / "artifacts"

REMOTE_BENCHMARK_PATH = "/opt/ml-lab/gpu_benchmark.py"
REMOTE_ARTIFACTS_ROOT = "/opt/ml-lab/artifacts"
REMOTE_VENV_PYTHON = "/opt/ml-lab/venv/bin/python"
CLOUD_INIT_LOG = "/var/log/cloud-init-output.log"


def deliver_and_run_benchmark(
    host, run_id, *, key_path,
    smoke_model_id=SMOKE_MODEL_ID,
    timeout=BENCHMARK_TIMEOUT_SECONDS,
) -> int:
    """scp the benchmark up, run it over SSH, return its exit code.

    Raises RemoteError only on transport failure (scp) or the benchmark timing out;
    a benchmark that runs to completion but fails returns a nonzero code.
    """
    remote.scp_up(host, str(BENCHMARK_SCRIPT_PATH), REMOTE_BENCHMARK_PATH,
                  key_path=key_path, timeout=SCP_TIMEOUT_SECONDS)
    argv = [
        REMOTE_VENV_PYTHON, REMOTE_BENCHMARK_PATH,
        "--run-id", run_id,
        "--smoke-model-id", smoke_model_id,
        "--artifacts-root", REMOTE_ARTIFACTS_ROOT,
    ]
    try:
        result = remote._run_ssh(host, argv, key_path=key_path, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RemoteError(f"benchmark on {host} timed out after {timeout}s") from e
    return result.returncode


def pull_artifacts(
    host, run_id, *, key_path,
    dest_root=LOCAL_ARTIFACTS_ROOT,
    timeout=SCP_TIMEOUT_SECONDS,
) -> Path:
    """Pull the remote bundle to dest_root/<run-id>/ plus the cloud-init log as bootstrap.log."""
    dest_root = Path(dest_root)
    dest = dest_root / run_id
    dest.mkdir(parents=True, exist_ok=True)
    remote.scp_down(host, f"{REMOTE_ARTIFACTS_ROOT}/{run_id}", str(dest_root),
                    key_path=key_path, timeout=timeout, recursive=True)
    remote.scp_down(host, CLOUD_INIT_LOG, str(dest / "bootstrap.log"),
                    key_path=key_path, timeout=timeout)
    return dest
