"""The DO Spaces seam: boto3 upload of a pulled artifact bundle, mocked in every test.

Isolated like do_client.py / remote.py — one external dependency (boto3), one public
function, an injectable client factory so no test touches boto3's wire. boto3 is
imported lazily inside the factory so importing this module (and collecting tests that
inject a client) never needs the SDK loaded.
"""

from __future__ import annotations

from pathlib import Path

from ml_lab.gpu.constants import SPACES_ENDPOINT, SPACES_KEY_PREFIX


class SpacesError(RuntimeError):
    """A DO Spaces upload failed, or the bundle to upload was missing/empty."""


def _make_client(env):
    """Build a boto3 S3 client against the DO Spaces endpoint with env credentials."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=env.access_key,
        aws_secret_access_key=env.secret_key,
    )


def upload_bundle(local_dir, run_id, *, env, client=None) -> str:
    """Upload every file under local_dir to s3://<bucket>/<prefix>/<run-id>/<relpath>.

    Returns the s3:// URI of the run's prefix. Raises SpacesError on a missing/empty
    bundle or any upload failure (the boto3 error is chained). `client` is injected in
    every test; in production it defaults to a real DO Spaces S3 client.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise SpacesError(f"bundle directory not found: {local_dir}")
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise SpacesError(f"bundle directory is empty: {local_dir}")

    if client is None:
        client = _make_client(env)

    prefix = f"{SPACES_KEY_PREFIX}/{run_id}"
    for path in files:
        rel = path.relative_to(local_dir).as_posix()  # POSIX keys even on Windows-ish paths
        key = f"{prefix}/{rel}"
        try:
            client.upload_file(str(path), env.bucket, key)
        except Exception as exc:  # boto3 ClientError/BotoCoreError — any failure is a Spaces failure
            raise SpacesError(f"upload of {rel} to {env.bucket} failed: {exc}") from exc

    return f"s3://{env.bucket}/{prefix}/"
