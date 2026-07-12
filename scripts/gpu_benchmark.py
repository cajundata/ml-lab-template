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
    """Raw nvidia-smi output (best-effort); include stderr so a failing call is diagnosable."""
    proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return proc.stdout + proc.stderr


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
