# GPU Benchmark + Local Drive/Pull Slice (S3c-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the scp seam, a self-contained on-droplet `gpu_benchmark.py`, and the local driver that delivers/runs it and pulls the artifact bundle — a library slice with no live droplet, SSH, or GPU touched.

**Architecture:** scp primitives extend the existing `remote.py` SSH seam (same `SSH_OPTS`, same subprocess-mock test pattern, but one-shot: failure raises `RemoteError`). `scripts/gpu_benchmark.py` is a stdlib-only-at-import file with a testable `run_benchmark` core plus lazy-importing GPU probes; each probe returns `{"ok": bool, ...}`, the core always writes a partial bundle, and the exit code is `0` iff both *required* probes (`torch_cuda`, `vllm_smoke`) pass. A new `ml_lab/gpu/benchmark.py` holds the local driver: `deliver_and_run_benchmark` (returns the benchmark exit code; raises only on transport/timeout) and `pull_artifacts`.

**Tech Stack:** Python 3, `subprocess` (ssh/scp seam), `argparse`/`json` (benchmark script), pytest + `monkeypatch` (all seams mocked). Reuses `constants.py`, `remote.py`, `SMOKE_MODEL_ID`.

---

## File structure

- Modify `src/ml_lab/gpu/constants.py` — add `SCP_TIMEOUT_SECONDS`.
- Modify `src/ml_lab/gpu/remote.py` — add `_run_scp`, `_checked_scp`, `scp_up`, `scp_down`.
- Create `scripts/gpu_benchmark.py` — self-contained on-droplet benchmark (core + probes + `main`).
- Create `src/ml_lab/gpu/benchmark.py` — local driver (`deliver_and_run_benchmark`, `pull_artifacts`).
- Modify `pyproject.toml` — add `"scripts"` to pytest `pythonpath` so tests can `import gpu_benchmark`.
- Modify `tests/test_gpu_remote.py` — scp seam tests.
- Create `tests/test_gpu_benchmark.py` — benchmark core/probe/main tests.
- Create `tests/test_gpu_benchmark_driver.py` — driver tests.

---

## Task 1: scp seam in `remote.py`

**Files:**
- Modify: `src/ml_lab/gpu/constants.py`
- Modify: `src/ml_lab/gpu/remote.py`
- Test: `tests/test_gpu_remote.py`

- [ ] **Step 1: Add the scp timeout constant**

In `src/ml_lab/gpu/constants.py`, under the "Lifecycle timing (seconds)" block (after `SSH_ATTEMPT_TIMEOUT_SECONDS`), add:

```python
SCP_TIMEOUT_SECONDS = 120  # bounds a single scp transfer (small script up / bundle down)
```

- [ ] **Step 2: Write the failing scp tests**

Append to `tests/test_gpu_remote.py`:

```python
def test_run_scp_builds_argv(monkeypatch):
    captured = {}

    def fake_run(argv, *a, **k):
        captured["argv"] = argv
        return _completed(returncode=0)

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    remote._run_scp(["/tmp/x", "root@1.2.3.4:/remote/x"], key_path="/key", timeout=120)
    argv = captured["argv"]
    assert argv[0] == "scp"
    assert "-i" in argv and argv[argv.index("-i") + 1] == "/key"
    assert "BatchMode=yes" in argv  # reuses hardened SSH_OPTS
    assert argv[-2:] == ["/tmp/x", "root@1.2.3.4:/remote/x"]


def test_scp_up_sends_local_to_remote(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        remote, "_run_scp",
        lambda argv, **k: captured.setdefault("argv", argv) or _completed(returncode=0),
    )
    remote.scp_up("1.2.3.4", "/local/f", "/remote/f", key_path="/key")
    assert captured["argv"] == ["/local/f", "root@1.2.3.4:/remote/f"]


def test_scp_down_pulls_remote_to_local_with_recursive(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        remote, "_run_scp",
        lambda argv, **k: captured.setdefault("argv", argv) or _completed(returncode=0),
    )
    remote.scp_down("1.2.3.4", "/remote/dir", "/local/dir", key_path="/key", recursive=True)
    assert captured["argv"] == ["-r", "root@1.2.3.4:/remote/dir", "/local/dir"]


def test_scp_down_without_recursive_has_no_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        remote, "_run_scp",
        lambda argv, **k: captured.setdefault("argv", argv) or _completed(returncode=0),
    )
    remote.scp_down("1.2.3.4", "/remote/log", "/local/log", key_path="/key")
    assert captured["argv"] == ["root@1.2.3.4:/remote/log", "/local/log"]


def test_scp_raises_remote_error_on_nonzero(monkeypatch):
    monkeypatch.setattr(remote, "_run_scp", lambda argv, **k: _completed(returncode=1, stderr="nope"))
    with pytest.raises(RemoteError):
        remote.scp_up("1.2.3.4", "/local/f", "/remote/f", key_path="/key")


def test_scp_raises_remote_error_on_timeout(monkeypatch):
    def boom(argv, **k):
        raise subprocess.TimeoutExpired(cmd="scp", timeout=120)

    monkeypatch.setattr(remote, "_run_scp", boom)
    with pytest.raises(RemoteError):
        remote.scp_down("1.2.3.4", "/remote/dir", "/local/dir", key_path="/key", recursive=True)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_remote.py -k scp -v`
Expected: FAIL — `AttributeError: module 'ml_lab.gpu.remote' has no attribute '_run_scp'`.

- [ ] **Step 4: Implement the scp seam**

In `src/ml_lab/gpu/remote.py`, add `SCP_TIMEOUT_SECONDS` to the constants import:

```python
from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    REMOTE_POLL_INTERVAL_SECONDS,
    SCP_TIMEOUT_SECONDS,
    SSH_ATTEMPT_TIMEOUT_SECONDS,
    SSH_TIMEOUT_SECONDS,
)
```

Then append these functions to the module:

```python
def _run_scp(argv, *, key_path, timeout):
    """Run `scp -i <key> <SSH_OPTS...> <argv...>`; return the CompletedProcess."""
    return subprocess.run(
        ["scp", "-i", key_path, *SSH_OPTS, *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _checked_scp(argv, *, key_path, timeout, what):
    """Run a one-shot scp; raise RemoteError on timeout or nonzero exit."""
    try:
        result = _run_scp(argv, key_path=key_path, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RemoteError(f"scp {what} timed out after {timeout}s") from e
    if result.returncode != 0:
        raise RemoteError(f"scp {what} failed ({result.returncode}): {result.stderr.strip()}")


def scp_up(host, local_path, remote_path, *, key_path, timeout=SCP_TIMEOUT_SECONDS):
    """Copy a local file up to root@<host>:<remote_path>; RemoteError on failure."""
    _checked_scp(
        [local_path, f"root@{host}:{remote_path}"],
        key_path=key_path, timeout=timeout, what=f"up {local_path}",
    )


def scp_down(host, remote_path, local_path, *, key_path, timeout=SCP_TIMEOUT_SECONDS, recursive=False):
    """Pull root@<host>:<remote_path> down to local_path; RemoteError on failure."""
    argv = (["-r"] if recursive else []) + [f"root@{host}:{remote_path}", local_path]
    _checked_scp(argv, key_path=key_path, timeout=timeout, what=f"down {remote_path}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_remote.py -v`
Expected: PASS (all existing + 6 new scp tests).

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/constants.py src/ml_lab/gpu/remote.py tests/test_gpu_remote.py
git commit -m "feat: remote scp seam (scp_up/scp_down, one-shot RemoteError)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `gpu_benchmark.py` testable core

**Files:**
- Create: `scripts/gpu_benchmark.py`
- Modify: `pyproject.toml`
- Test: `tests/test_gpu_benchmark.py`

- [ ] **Step 1: Put `scripts` on the pytest path**

In `pyproject.toml`, change the pytest `pythonpath` line from:

```toml
pythonpath = ["src"]
```

to:

```toml
pythonpath = ["src", "scripts"]
```

This lets tests `import gpu_benchmark` (the on-droplet script is not a package).

- [ ] **Step 2: Write the failing core tests**

Create `tests/test_gpu_benchmark.py`:

```python
import json

import pytest

import gpu_benchmark


def _read_json(path):
    return json.loads(path.read_text())


def test_module_imports_without_gpu_libraries():
    # The whole point: torch / vllm are NOT installed in CI, yet import must succeed
    # because they are imported lazily inside the probes, never at module top.
    assert hasattr(gpu_benchmark, "run_benchmark")


def test_all_probes_pass_writes_full_bundle_and_exit_zero(tmp_path):
    probes = {
        "torch_cuda": lambda: {"ok": True, "cuda_available": True},
        "vllm_smoke": lambda: {"ok": True, "generate_ok": True},
        "system": lambda: {"ok": True, "os": "linux"},
        "nvidia_smi": lambda: "GPU 0: fine",
    }
    code = gpu_benchmark.run_benchmark("R1", str(tmp_path), probes)
    assert code == 0
    run_dir = tmp_path / "R1"
    for name in ("benchmark.json", "system.json", "torch_cuda.json",
                 "vllm_smoke.json", "nvidia-smi.txt", "benchmark.log"):
        assert (run_dir / name).exists()
    assert _read_json(run_dir / "benchmark.json")["required_ok"] is True


def test_required_probe_failure_records_error_still_writes_others_exit_one(tmp_path):
    def boom():
        raise RuntimeError("cuda exploded")

    probes = {
        "torch_cuda": boom,  # required
        "vllm_smoke": lambda: {"ok": True},
        "system": lambda: {"ok": True},
        "nvidia_smi": lambda: "smi text",
    }
    code = gpu_benchmark.run_benchmark("R2", str(tmp_path), probes)
    assert code == 1
    run_dir = tmp_path / "R2"
    # The failing probe's artifact records the error...
    torch_art = _read_json(run_dir / "torch_cuda.json")
    assert torch_art["ok"] is False
    assert "cuda exploded" in torch_art["error"]
    # ...and the other artifacts were still written (partial bundle guaranteed).
    assert (run_dir / "vllm_smoke.json").exists()
    assert (run_dir / "nvidia-smi.txt").exists()
    assert _read_json(run_dir / "benchmark.json")["required_ok"] is False


def test_optional_probe_failure_does_not_fail_the_run(tmp_path):
    def boom():
        raise RuntimeError("no nvidia-smi")

    probes = {
        "torch_cuda": lambda: {"ok": True},
        "vllm_smoke": lambda: {"ok": True},
        "system": lambda: {"ok": True},
        "nvidia_smi": boom,  # optional
    }
    code = gpu_benchmark.run_benchmark("R3", str(tmp_path), probes)
    assert code == 0
    assert "no nvidia-smi" in (tmp_path / "R3" / "nvidia-smi.txt").read_text()


def test_probe_returning_ok_false_fails_required(tmp_path):
    probes = {
        "torch_cuda": lambda: {"ok": True},
        "vllm_smoke": lambda: {"ok": False, "error": "model did not load"},
        "system": lambda: {"ok": True},
        "nvidia_smi": lambda: "smi",
    }
    code = gpu_benchmark.run_benchmark("R4", str(tmp_path), probes)
    assert code == 1


def test_run_benchmark_creates_run_dir(tmp_path):
    probes = {
        "torch_cuda": lambda: {"ok": True},
        "vllm_smoke": lambda: {"ok": True},
        "system": lambda: {"ok": True},
        "nvidia_smi": lambda: "smi",
    }
    gpu_benchmark.run_benchmark("R5", str(tmp_path), probes)
    assert (tmp_path / "R5").is_dir()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpu_benchmark'`.

- [ ] **Step 4: Implement the core (probe registry, writers, `run_benchmark`)**

Create `scripts/gpu_benchmark.py` with **stdlib-only top-level imports** and the testable core. (Real probes + `main` land in Task 3; a temporary `DEFAULT_PROBES = {}` placeholder is added here only so the module imports.)

```python
#!/usr/bin/env python3
"""Self-contained GPU benchmark run ON the droplet (scp'd there; no ml_lab import).

Top-level imports are stdlib only so the module is importable in CI; torch / vllm are
imported lazily inside each probe. Each probe returns {"ok": bool, ...} (or raw text
for nvidia-smi). run_benchmark always writes a partial bundle and exits 0 iff both
required probes (torch_cuda, vllm_smoke) pass.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

# (filename, required, is_text)
PROBE_SPEC = {
    "torch_cuda": ("torch_cuda.json", True, False),
    "vllm_smoke": ("vllm_smoke.json", True, False),
    "system": ("system.json", False, False),
    "nvidia_smi": ("nvidia-smi.txt", False, True),
}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def _write_text(path, value):
    path.write_text(value if isinstance(value, str) else str(value))


def run_benchmark(run_id, artifacts_root, probes) -> int:
    """Run every probe defensively; write the bundle; return 0 iff required probes pass.

    `probes` is a name->zero-arg-callable mapping (injected in tests). A probe that
    raises, or a JSON probe returning {"ok": False, ...}, counts as a failure for that
    probe; the remaining artifacts are still written.
    """
    run_dir = Path(artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, (filename, _required, is_text) in PROBE_SPEC.items():
        try:
            value = probes[name]()
            if is_text:
                _write_text(run_dir / filename, value)
                ok = True
            else:
                _write_json(run_dir / filename, value)
                ok = bool(value.get("ok", True))
        except Exception:
            tb = traceback.format_exc()
            if is_text:
                _write_text(run_dir / filename, tb)
            else:
                _write_json(run_dir / filename, {"ok": False, "error": tb})
            ok = False
        results[name] = ok

    required_ok = all(ok for name, ok in results.items() if PROBE_SPEC[name][1])
    _write_json(run_dir / "benchmark.json",
                {"run_id": run_id, "results": results, "required_ok": required_ok})
    _write_text(run_dir / "benchmark.log",
                "\n".join(f"{n}: {'ok' if results[n] else 'FAILED'}" for n in PROBE_SPEC))
    return 0 if required_ok else 1


DEFAULT_PROBES = {}  # real probes wired in Task 3
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_benchmark.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml scripts/gpu_benchmark.py tests/test_gpu_benchmark.py
git commit -m "feat: gpu_benchmark run_benchmark core (defensive bundle, required-probe exit code)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `gpu_benchmark.py` real probes + `main`

**Files:**
- Modify: `scripts/gpu_benchmark.py`
- Test: `tests/test_gpu_benchmark.py`

- [ ] **Step 1: Write the failing `main` tests**

Append to `tests/test_gpu_benchmark.py`:

```python
def test_default_probes_names_match_spec():
    probes = gpu_benchmark._default_probes("facebook/opt-125m")
    assert set(probes) == set(gpu_benchmark.PROBE_SPEC)


def test_main_exits_with_run_benchmark_code(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gpu_benchmark, "_default_probes",
        lambda model_id: {
            "torch_cuda": lambda: {"ok": True},
            "vllm_smoke": lambda: {"ok": True},
            "system": lambda: {"ok": True},
            "nvidia_smi": lambda: "smi",
        },
    )
    with pytest.raises(SystemExit) as exc:
        gpu_benchmark.main(["--run-id", "R9", "--artifacts-root", str(tmp_path)])
    assert exc.value.code == 0
    assert (tmp_path / "R9" / "benchmark.json").exists()


def test_main_propagates_smoke_model_id(monkeypatch, tmp_path):
    captured = {}

    def fake_defaults(model_id):
        captured["model_id"] = model_id
        return {name: (lambda: {"ok": True}) for name in gpu_benchmark.PROBE_SPEC}

    monkeypatch.setattr(gpu_benchmark, "_default_probes", fake_defaults)
    with pytest.raises(SystemExit):
        gpu_benchmark.main(
            ["--run-id", "R10", "--smoke-model-id", "my/model", "--artifacts-root", str(tmp_path)]
        )
    assert captured["model_id"] == "my/model"
```

Note: `_default_probes` here builds a JSON probe for every name including `nvidia_smi`; `run_benchmark` writes that dict to `nvidia-smi.txt` via `_write_text(str(value))`, which is fine for this exit-code test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_benchmark.py -k "default_probes or main" -v`
Expected: FAIL — `AttributeError: module 'gpu_benchmark' has no attribute '_default_probes'`.

- [ ] **Step 3: Implement the real probes, `_default_probes`, and `main`**

In `scripts/gpu_benchmark.py`, replace the `DEFAULT_PROBES = {}` placeholder line with the probes, the probe builder, and `main`:

```python
def probe_torch_cuda() -> dict:
    """CUDA visibility + a small matmul, using the torch installed by vLLM."""
    import torch  # lazy: not importable in CI

    start = time.perf_counter()
    cuda = bool(torch.cuda.is_available())
    matmul_ok = False
    if cuda:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        (a @ b).sum().item()
        matmul_ok = True
    return {
        "ok": cuda and matmul_ok,
        "cuda_available": cuda,
        "matmul_ok": matmul_ok,
        "elapsed_s": round(time.perf_counter() - start, 4),
    }


def probe_vllm_smoke(model_id) -> dict:
    """Import vLLM, load the pinned smoke model, generate a few tokens."""
    from vllm import LLM, SamplingParams  # lazy: not importable in CI

    start = time.perf_counter()
    result = {
        "ok": False, "model_id": model_id, "load_ok": False,
        "generate_ok": False, "token_count": 0, "elapsed_s": 0.0, "error": None,
    }
    try:
        llm = LLM(model=model_id)
        result["load_ok"] = True
        out = llm.generate(["Hello from the ML lab"], SamplingParams(max_tokens=8))
        tokens = out[0].outputs[0].token_ids
        result["token_count"] = len(tokens)
        result["generate_ok"] = len(tokens) > 0
        result["ok"] = result["load_ok"] and result["generate_ok"]
    except Exception:
        result["error"] = traceback.format_exc()
    result["elapsed_s"] = round(time.perf_counter() - start, 4)
    return result


def probe_system() -> dict:
    """OS / Python / GPU name+memory / driver / CUDA visibility (informational)."""
    result = {
        "ok": True, "os": platform.platform(), "python": platform.python_version(),
        "gpu_name": None, "gpu_memory": None, "cuda_visible": False, "error": None,
    }
    try:
        import torch  # lazy

        result["cuda_visible"] = bool(torch.cuda.is_available())
        if result["cuda_visible"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def probe_nvidia_smi() -> str:
    """Raw nvidia-smi output (best-effort)."""
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


def _default_probes(smoke_model_id):
    return {
        "torch_cuda": probe_torch_cuda,
        "vllm_smoke": lambda: probe_vllm_smoke(smoke_model_id),
        "system": probe_system,
        "nvidia_smi": probe_nvidia_smi,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="ML lab GPU benchmark (runs on the droplet).")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-model-id", default="facebook/opt-125m")
    parser.add_argument("--artifacts-root", default="/opt/ml-lab/artifacts")
    args = parser.parse_args(argv)
    sys.exit(run_benchmark(args.run_id, args.artifacts_root, _default_probes(args.smoke_model_id)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_benchmark.py -v`
Expected: PASS (9 tests), including `test_module_imports_without_gpu_libraries` (torch/vllm still not installed — lazy imports keep the module importable).

- [ ] **Step 5: Commit**

```bash
git add scripts/gpu_benchmark.py tests/test_gpu_benchmark.py
git commit -m "feat: gpu_benchmark real probes (lazy torch/vllm) + argparse main

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: local driver `benchmark.py`

**Files:**
- Create: `src/ml_lab/gpu/benchmark.py`
- Test: `tests/test_gpu_benchmark_driver.py`

- [ ] **Step 1: Write the failing driver tests**

Create `tests/test_gpu_benchmark_driver.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpu_benchmark_driver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_lab.gpu.benchmark'`.

- [ ] **Step 3: Implement the local driver**

Create `src/ml_lab/gpu/benchmark.py`:

```python
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
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / run_id
    remote.scp_down(host, f"{REMOTE_ARTIFACTS_ROOT}/{run_id}", str(dest_root),
                    key_path=key_path, timeout=timeout, recursive=True)
    remote.scp_down(host, CLOUD_INIT_LOG, str(dest / "bootstrap.log"),
                    key_path=key_path, timeout=timeout)
    return dest
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpu_benchmark_driver.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: PASS — the S3b baseline (140) plus the new scp (6), benchmark core/probe/main (9), and driver (6) tests.

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/gpu/benchmark.py tests/test_gpu_benchmark_driver.py
git commit -m "feat: local benchmark driver (deliver_and_run + pull_artifacts)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:**
- scp seam (`_run_scp`/`scp_up`/`scp_down`, one-shot RemoteError, `-r`) → Task 1. ✓
- `gpu_benchmark.py` stdlib-only import, testable core, defensive partial bundle, required-probe exit code → Tasks 2–3. ✓
- Four probes (torch_cuda/vllm_smoke required; system/nvidia-smi informational) with lazy GPU imports → Task 3. ✓
- Six-file bundle (`benchmark/system/torch_cuda/vllm_smoke.json`, `nvidia-smi.txt`, `benchmark.log`) → Task 2 core writes all six. ✓
- Local driver: `deliver_and_run_benchmark` returns exit code, raises on transport/timeout; `pull_artifacts` pulls bundle + cloud-init log→`bootstrap.log` → Task 4. ✓
- Module-anchored paths, `SMOKE_MODEL_ID` passed explicitly, `BENCHMARK_TIMEOUT_SECONDS` on the run → Task 4. ✓
- All offline, every seam mocked → all test steps use `monkeypatch`. ✓
- No new `make` target, no Spaces/`gpu_run`/`run` (S3c-2), no live GPU (S4) → nothing in the plan adds them. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The only intentional placeholder (`DEFAULT_PROBES = {}` in Task 2) is explicitly replaced in Task 3 Step 3. ✓

**Type/name consistency:** `run_benchmark(run_id, artifacts_root, probes)`, `PROBE_SPEC`, `_default_probes`, `_write_json`/`_write_text`, `_run_scp`/`_checked_scp`/`scp_up`/`scp_down`, `deliver_and_run_benchmark`/`pull_artifacts`, `REMOTE_BENCHMARK_PATH`/`REMOTE_VENV_PYTHON`/`REMOTE_ARTIFACTS_ROOT`/`CLOUD_INIT_LOG`/`BENCHMARK_SCRIPT_PATH` are used identically across tasks and tests. `SCP_TIMEOUT_SECONDS` is defined in Task 1 before use in Tasks 1 & 4. ✓
