"""Read and validate the operator's local GPU env vars (fail loud, all at once)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class GpuEnvError(RuntimeError):
    """One or more required GPU env vars are missing or empty."""


@dataclass(frozen=True)
class GpuEnv:
    destroy_token: str      # DO_DROPLET_DESTROY_TOKEN
    ssh_key_ids: list[str]  # DO_SSH_KEY_IDS, comma-split
    ssh_key_path: str       # DO_SSH_KEY_PATH


def load_gpu_env() -> GpuEnv:
    """Load .env, then read the three GPU vars. Raise GpuEnvError naming all missing ones."""
    load_dotenv()
    destroy_token = os.environ.get("DO_DROPLET_DESTROY_TOKEN", "").strip()
    ssh_key_ids_raw = os.environ.get("DO_SSH_KEY_IDS", "").strip()
    ssh_key_path = os.environ.get("DO_SSH_KEY_PATH", "").strip()

    missing = [
        name
        for name, value in (
            ("DO_DROPLET_DESTROY_TOKEN", destroy_token),
            ("DO_SSH_KEY_IDS", ssh_key_ids_raw),
            ("DO_SSH_KEY_PATH", ssh_key_path),
        )
        if not value
    ]
    if missing:
        raise GpuEnvError(f"missing/empty GPU env vars: {', '.join(missing)}")

    ssh_key_ids = [k.strip() for k in ssh_key_ids_raw.split(",") if k.strip()]
    return GpuEnv(
        destroy_token=destroy_token, ssh_key_ids=ssh_key_ids, ssh_key_path=ssh_key_path
    )


@dataclass(frozen=True)
class SpacesEnv:
    access_key: str  # SPACES_ACCESS_KEY_ID
    secret_key: str  # SPACES_SECRET_ACCESS_KEY
    bucket: str      # SPACES_BUCKET


def load_spaces_env() -> SpacesEnv:
    """Load .env, then read the three DO Spaces vars. Raise GpuEnvError naming all missing.

    Separate from load_gpu_env: only gpu-run uploads artifacts, so gpu-up must not
    fail on unset Spaces credentials.
    """
    load_dotenv()
    access_key = os.environ.get("SPACES_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("SPACES_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("SPACES_BUCKET", "").strip()

    missing = [
        name
        for name, value in (
            ("SPACES_ACCESS_KEY_ID", access_key),
            ("SPACES_SECRET_ACCESS_KEY", secret_key),
            ("SPACES_BUCKET", bucket),
        )
        if not value
    ]
    if missing:
        raise GpuEnvError(f"missing/empty Spaces env vars: {', '.join(missing)}")

    return SpacesEnv(access_key=access_key, secret_key=secret_key, bucket=bucket)
