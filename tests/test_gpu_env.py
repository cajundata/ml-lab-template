import pytest

from ml_lab.gpu import gpu_env


def test_load_gpu_env_happy(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")
    monkeypatch.setenv("DO_SSH_KEY_IDS", "a1:b2, c3:d4")
    monkeypatch.setenv("DO_SSH_KEY_PATH", "/home/me/.ssh/id_ed25519")
    env = gpu_env.load_gpu_env()
    assert env.destroy_token == "tok"
    assert env.ssh_key_ids == ["a1:b2", "c3:d4"]  # comma-split, whitespace trimmed
    assert env.ssh_key_path == "/home/me/.ssh/id_ed25519"


def test_load_gpu_env_missing_lists_every_var(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.delenv("DO_DROPLET_DESTROY_TOKEN", raising=False)
    monkeypatch.delenv("DO_SSH_KEY_IDS", raising=False)
    monkeypatch.delenv("DO_SSH_KEY_PATH", raising=False)
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_gpu_env()
    msg = str(exc.value)
    assert "DO_DROPLET_DESTROY_TOKEN" in msg
    assert "DO_SSH_KEY_IDS" in msg
    assert "DO_SSH_KEY_PATH" in msg
