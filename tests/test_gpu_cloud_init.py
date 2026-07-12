import yaml

from ml_lab.gpu import cloud_init
from ml_lab.gpu.constants import (
    DEFAULT_TTL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SMOKE_MODEL_ID,
)

FAKE_TOKEN = "dop_v1_FAKEfake0123456789"


def _write_file(doc, path):
    for wf in doc["write_files"]:
        if wf["path"] == path:
            return wf["content"]
    raise AssertionError(f"{path} not in write_files")


def _run_env(doc):
    return _write_file(doc, "/etc/ml-lab/run.env")


def test_render_substitutes_all_placeholders_and_parses():
    out = cloud_init.render_cloud_init(run_id="20260712-abc123", destroy_token=FAKE_TOKEN)
    assert "@@" not in out  # every placeholder filled
    doc = yaml.safe_load(out)  # structural check: valid cloud-config mapping
    assert isinstance(doc, dict)
    assert doc["package_update"] is False
    assert "packages" not in doc
    run_env = _run_env(doc)
    assert "RUN_ID=20260712-abc123" in run_env
    assert f"TTL_SECONDS={DEFAULT_TTL_SECONDS}" in run_env
    assert f"SMOKE_MODEL_ID={SMOKE_MODEL_ID}" in run_env
    assert f"DO_DROPLET_DESTROY_TOKEN={FAKE_TOKEN}" in run_env


def test_render_wires_timer_from_params():
    out = cloud_init.render_cloud_init(
        run_id="r", destroy_token=FAKE_TOKEN, ttl_seconds=900, self_destruct_retry_seconds=120
    )
    timer = _write_file(yaml.safe_load(out), "/etc/systemd/system/ml-lab-self-destruct.timer")
    assert "OnBootSec=900" in timer
    assert "OnUnitActiveSec=120" in timer


def test_render_defaults_timer_from_constants():
    out = cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
    timer = _write_file(yaml.safe_load(out), "/etc/systemd/system/ml-lab-self-destruct.timer")
    assert f"OnBootSec={DEFAULT_TTL_SECONDS}" in timer
    assert f"OnUnitActiveSec={SELF_DESTRUCT_RETRY_SECONDS}" in timer


def test_render_preserves_bash_expansions():
    out = cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
    # bash ${...} in self_destruct.sh must survive the @@...@@ substitution untouched
    assert "${DROPLET_ID}" in out
    assert "${DO_DROPLET_DESTROY_TOKEN}" in out


def test_token_appears_only_once_and_in_run_env():
    out = cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
    assert out.count(FAKE_TOKEN) == 1  # secret is not duplicated across the file
    assert FAKE_TOKEN in _run_env(yaml.safe_load(out))


def test_runcmd_arms_timer_before_install_and_marks_ready_last():
    doc = yaml.safe_load(cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN))
    cmds = [str(c) for c in doc["runcmd"]]
    enable_idx = next(i for i, c in enumerate(cmds) if "enable --now ml-lab-self-destruct.timer" in c)
    pip_idx = next(i for i, c in enumerate(cmds) if "pip install vllm" in c)
    ready_idx = next(i for i, c in enumerate(cmds) if "bootstrap-ready.json" in c)
    assert enable_idx < pip_idx < ready_idx  # arm before slow install; mark ready last
    active_idxs = [i for i, c in enumerate(cmds) if "is-active --quiet ml-lab-self-destruct.timer" in c]
    assert any(i < enable_idx + 2 for i in active_idxs)  # verified right after enable
    assert any(i > pip_idx for i in active_idxs)  # re-verified before the ready marker


import pytest


def test_render_raises_on_empty_token():
    with pytest.raises(ValueError, match="destroy_token"):
        cloud_init.render_cloud_init(run_id="r", destroy_token="")


def test_render_raises_on_leftover_placeholder(monkeypatch, tmp_path):
    bad = tmp_path / "bad.tmpl"
    bad.write_text("RUN_ID=@@RUN_ID@@\nMYSTERY=@@MYSTERY@@\n")
    monkeypatch.setattr(cloud_init, "_TEMPLATE_PATH", bad)
    with pytest.raises(ValueError, match="placeholder"):
        cloud_init.render_cloud_init(run_id="r", destroy_token=FAKE_TOKEN)
