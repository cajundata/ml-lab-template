"""Render the self-destruct cloud-init into the user_data string create_lab_droplet requires.

Pure string templating over scripts/cloud-init-gpu.yaml.tmpl using collision-proof
@@PLACEHOLDER@@ markers, so the template's literal bash ${...} survives untouched.
No network, no droplet — S3b supplies the caller (env token + create_lab_droplet).
"""

from __future__ import annotations

import re
from pathlib import Path

from ml_lab.gpu.constants import (
    DEFAULT_TTL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SMOKE_MODEL_ID,
)

# src/ml_lab/gpu/cloud_init.py -> parents[3] is the repo root; scripts/ lives there.
_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "cloud-init-gpu.yaml.tmpl"

_PLACEHOLDER_RE = re.compile(r"@@[A-Z_]+@@")


def render_cloud_init(
    *,
    run_id: str,
    destroy_token: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    self_destruct_retry_seconds: int = SELF_DESTRUCT_RETRY_SECONDS,
    smoke_model_id: str = SMOKE_MODEL_ID,
) -> str:
    """Render the cloud-init user_data string. destroy_token lands only in run.env (0600)."""
    if not destroy_token:
        raise ValueError("destroy_token is required (self-destruct cannot arm without it)")
    template = _TEMPLATE_PATH.read_text()
    rendered = (
        template.replace("@@RUN_ID@@", run_id)
        .replace("@@TTL_SECONDS@@", str(ttl_seconds))
        .replace("@@SELF_DESTRUCT_RETRY_SECONDS@@", str(self_destruct_retry_seconds))
        .replace("@@SMOKE_MODEL_ID@@", smoke_model_id)
        .replace("@@DESTROY_TOKEN@@", destroy_token)
    )
    leftover = _PLACEHOLDER_RE.findall(rendered)
    if leftover:
        raise ValueError(f"unfilled cloud-init placeholders: {sorted(set(leftover))}")
    return rendered
