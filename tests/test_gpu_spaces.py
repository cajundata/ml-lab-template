import pytest

from ml_lab.gpu import gpu_env, spaces


class _RecordingClient:
    """Captures upload_file(Filename, Bucket, Key) calls; optionally raises."""

    def __init__(self, boom=False):
        self.calls = []
        self.boom = boom

    def upload_file(self, filename, bucket, key):
        if self.boom:
            raise RuntimeError("boto3 exploded")
        self.calls.append((filename, bucket, key))


def _env():
    return gpu_env.SpacesEnv(access_key="AK", secret_key="SK", bucket="ml-lab-artifacts")


def _bundle(tmp_path):
    d = tmp_path / "R1"
    d.mkdir()
    (d / "benchmark.json").write_text("{}")
    (d / "nvidia-smi.txt").write_text("gpu")
    nested = d / "logs"
    nested.mkdir()
    (nested / "bootstrap.log").write_text("boot")
    return d


def test_upload_bundle_uploads_every_file_with_prefixed_keys(tmp_path):
    client = _RecordingClient()
    uri = spaces.upload_bundle(_bundle(tmp_path), "R1", env=_env(), client=client)
    keys = {key for (_f, _b, key) in client.calls}
    assert keys == {
        "ml-pathway/phase0/R1/benchmark.json",
        "ml-pathway/phase0/R1/nvidia-smi.txt",
        "ml-pathway/phase0/R1/logs/bootstrap.log",  # nested → POSIX relpath
    }
    assert all(bucket == "ml-lab-artifacts" for (_f, bucket, _k) in client.calls)
    assert uri == "s3://ml-lab-artifacts/ml-pathway/phase0/R1/"


def test_upload_bundle_missing_dir_raises_spaces_error(tmp_path):
    with pytest.raises(spaces.SpacesError):
        spaces.upload_bundle(tmp_path / "nope", "R1", env=_env(), client=_RecordingClient())


def test_upload_bundle_empty_dir_raises_spaces_error(tmp_path):
    empty = tmp_path / "R1"
    empty.mkdir()
    with pytest.raises(spaces.SpacesError):
        spaces.upload_bundle(empty, "R1", env=_env(), client=_RecordingClient())


def test_upload_bundle_wraps_client_failure_as_spaces_error(tmp_path):
    with pytest.raises(spaces.SpacesError):
        spaces.upload_bundle(_bundle(tmp_path), "R1", env=_env(), client=_RecordingClient(boom=True))


def test_default_client_uses_spaces_endpoint_and_credentials(monkeypatch):
    import boto3

    seen = {}

    def fake_client(service, **kwargs):
        seen["service"] = service
        seen.update(kwargs)
        return "CLIENT"

    monkeypatch.setattr(boto3, "client", fake_client)
    client = spaces._make_client(_env())
    assert client == "CLIENT"
    assert seen["service"] == "s3"
    assert seen["endpoint_url"] == spaces.SPACES_ENDPOINT
    assert seen["aws_access_key_id"] == "AK"
    assert seen["aws_secret_access_key"] == "SK"
