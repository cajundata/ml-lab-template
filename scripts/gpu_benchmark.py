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
