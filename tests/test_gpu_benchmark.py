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
        "transformers_smoke": lambda: {"ok": True, "generate_ok": True},
        "vllm_smoke": lambda: {"ok": True, "generate_ok": True},
        "system": lambda: {"ok": True, "os": "linux"},
        "nvidia_smi": lambda: "GPU 0: fine",
    }
    code = gpu_benchmark.run_benchmark("R1", str(tmp_path), probes)
    assert code == 0
    run_dir = tmp_path / "R1"
    for name in ("benchmark.json", "system.json", "torch_cuda.json", "transformers_smoke.json",
                 "vllm_smoke.json", "nvidia-smi.txt", "benchmark.log"):
        assert (run_dir / name).exists()
    assert _read_json(run_dir / "benchmark.json")["required_ok"] is True


def test_required_probe_failure_records_error_still_writes_others_exit_one(tmp_path):
    def boom():
        raise RuntimeError("cuda exploded")

    probes = {
        "torch_cuda": boom,  # required
        "transformers_smoke": lambda: {"ok": True},
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
        "transformers_smoke": lambda: {"ok": True},
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
        "transformers_smoke": lambda: {"ok": False, "error": "model did not load"},
        "vllm_smoke": lambda: {"ok": True},
        "system": lambda: {"ok": True},
        "nvidia_smi": lambda: "smi",
    }
    code = gpu_benchmark.run_benchmark("R4", str(tmp_path), probes)
    assert code == 1


def test_vllm_smoke_is_informational_not_required(tmp_path):
    # vLLM is a Phase-5 concern; a failed vllm_smoke must NOT fail the Phase-0 run
    # as long as the required probes (torch_cuda, transformers_smoke) pass.
    probes = {
        "torch_cuda": lambda: {"ok": True},
        "transformers_smoke": lambda: {"ok": True},
        "vllm_smoke": lambda: {"ok": False, "error": "engine core init failed"},
        "system": lambda: {"ok": True},
        "nvidia_smi": lambda: "smi",
    }
    code = gpu_benchmark.run_benchmark("R4b", str(tmp_path), probes)
    assert code == 0
    assert _read_json(tmp_path / "R4b" / "benchmark.json")["required_ok"] is True
    assert _read_json(tmp_path / "R4b" / "vllm_smoke.json")["ok"] is False


def test_run_benchmark_creates_run_dir(tmp_path):
    probes = {
        "torch_cuda": lambda: {"ok": True},
        "transformers_smoke": lambda: {"ok": True},
        "vllm_smoke": lambda: {"ok": True},
        "system": lambda: {"ok": True},
        "nvidia_smi": lambda: "smi",
    }
    gpu_benchmark.run_benchmark("R5", str(tmp_path), probes)
    assert (tmp_path / "R5").is_dir()


def test_default_probes_names_match_spec():
    probes = gpu_benchmark._default_probes("facebook/opt-125m")
    assert set(probes) == set(gpu_benchmark.PROBE_SPEC)


def test_main_exits_with_run_benchmark_code(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gpu_benchmark, "_default_probes",
        lambda model_id: {
            "torch_cuda": lambda: {"ok": True},
            "transformers_smoke": lambda: {"ok": True},
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
