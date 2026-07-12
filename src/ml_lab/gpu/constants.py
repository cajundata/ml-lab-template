"""Pinned DigitalOcean GPU-lab constants (master plan §3, no fallback)."""

# Toolchain / image / placement
DO_REGION = "atl1"
DO_SIZE_SLUG = "gpu-rtx4000x1-20gb"
DO_GPU_RUNG = "RTX 4000 Ada"
DO_IMAGE_SLUG = None  # DO NVIDIA AI/ML-ready GPU image; resolved in S2 (create-only)
SMOKE_MODEL_ID = "facebook/opt-125m"

# Lifecycle timing (seconds)
DEFAULT_TTL_SECONDS = 7200
SSH_TIMEOUT_SECONDS = 600
BOOTSTRAP_TIMEOUT_SECONDS = 1800
BENCHMARK_TIMEOUT_SECONDS = 1800
DESTROY_POLL_TIMEOUT_SECONDS = 600
SELF_DESTRUCT_RETRY_SECONDS = 300
DESTROY_POLL_INTERVAL_SECONDS = 10  # cadence for absence polling (added; not in plan)

# Identity / audit matching
DROPLET_NAME_PREFIX = "ml-lab-gpu-"
NAME_FORMAT = "ml-lab-gpu-phase0-{run_id}"
BASE_TAGS = ["ml-lab", "ml-pathway", "phase-0", "owner-weldon"]


def validate_timeout_budget(ttl_seconds: int) -> None:
    """Raise ValueError unless the SSH+bootstrap+benchmark budget fits under ttl_seconds."""
    budget = SSH_TIMEOUT_SECONDS + BOOTSTRAP_TIMEOUT_SECONDS + BENCHMARK_TIMEOUT_SECONDS
    if budget >= ttl_seconds:
        raise ValueError(
            f"timeout budget {budget}s does not fit under TTL {ttl_seconds}s"
        )
