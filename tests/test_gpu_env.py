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


def test_load_gpu_env_partial_missing_names_only_the_absent(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("DO_DROPLET_DESTROY_TOKEN", "tok")  # present
    monkeypatch.setenv("DO_SSH_KEY_IDS", "k1")  # present
    monkeypatch.delenv("DO_SSH_KEY_PATH", raising=False)  # missing
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_gpu_env()
    msg = str(exc.value)
    assert "DO_SSH_KEY_PATH" in msg
    assert "DO_DROPLET_DESTROY_TOKEN" not in msg  # present vars are not named
    assert "DO_SSH_KEY_IDS" not in msg


def test_load_spaces_env_happy(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("SPACES_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("SPACES_SECRET_ACCESS_KEY", "SK")
    monkeypatch.setenv("SPACES_BUCKET", "ml-lab-artifacts")
    env = gpu_env.load_spaces_env()
    assert env.access_key == "AK"
    assert env.secret_key == "SK"
    assert env.bucket == "ml-lab-artifacts"


def test_load_spaces_env_missing_lists_every_var(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.delenv("SPACES_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("SPACES_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("SPACES_BUCKET", raising=False)
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_spaces_env()
    msg = str(exc.value)
    assert "SPACES_ACCESS_KEY_ID" in msg
    assert "SPACES_SECRET_ACCESS_KEY" in msg
    assert "SPACES_BUCKET" in msg


def test_load_spaces_env_partial_missing_names_only_the_absent(monkeypatch):
    monkeypatch.setattr(gpu_env, "load_dotenv", lambda: None)
    monkeypatch.setenv("SPACES_ACCESS_KEY_ID", "AK")  # present
    monkeypatch.setenv("SPACES_SECRET_ACCESS_KEY", "SK")  # present
    monkeypatch.delenv("SPACES_BUCKET", raising=False)  # missing
    with pytest.raises(gpu_env.GpuEnvError) as exc:
        gpu_env.load_spaces_env()
    msg = str(exc.value)
    assert "SPACES_BUCKET" in msg
    assert "SPACES_ACCESS_KEY_ID" not in msg
    assert "SPACES_SECRET_ACCESS_KEY" not in msg
