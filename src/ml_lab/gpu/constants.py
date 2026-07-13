"""Pinned DigitalOcean GPU-lab constants (master plan §3, no fallback)."""

# Toolchain / image / placement
DO_REGION = "nyc2"  # PREFERRED region only; GPU capacity flips regions per-minute, so create.resolve_region picks the live region for the size at create time
DO_SIZE_SLUG = "gpu-h100x1-80gb"  # Nvidia H100 80GB single-GPU (type nvidia_h100); H200 (gpu-h200x1-141gb) had sustained no capacity at S4 live
DO_GPU_RUNG = "H100"
DO_IMAGE_SLUG = "gpu-h100x1-base"  # NVIDIA AI/ML Ready Image; native target for H100 (boots; proven on Hopper H200 at S4 live)
SMOKE_MODEL_ID = "facebook/opt-125m"

# DO Spaces (artifact upload) — region pinned like DO_REGION; change deliberately, no fallback.
SPACES_REGION = "nyc3"  # nyc2 has no Spaces; nyc3 is the nearest Spaces region
SPACES_ENDPOINT = f"https://{SPACES_REGION}.digitaloceanspaces.com"
SPACES_KEY_PREFIX = "ml-pathway/phase0"  # s3://<bucket>/ml-pathway/phase0/<run-id>/ (artifacts/README.md)

# Lifecycle timing (seconds)
DEFAULT_TTL_SECONDS = 7200
SSH_TIMEOUT_SECONDS = 600
BOOTSTRAP_TIMEOUT_SECONDS = 1800
BENCHMARK_TIMEOUT_SECONDS = 1800
DESTROY_POLL_TIMEOUT_SECONDS = 600
SELF_DESTRUCT_RETRY_SECONDS = 300
DESTROY_POLL_INTERVAL_SECONDS = 10  # cadence for absence polling (added; not in plan)
REMOTE_POLL_INTERVAL_SECONDS = 15  # cadence for ssh-readiness / bootstrap-marker polling
SSH_ATTEMPT_TIMEOUT_SECONDS = 30  # subprocess timeout bounding a single ssh attempt
SCP_TIMEOUT_SECONDS = 120  # bounds a single scp transfer (small script up / bundle down)
CREATE_CAPACITY_WAIT_SECONDS = 180  # bounded poll for a GPU capacity window before giving up
CREATE_CAPACITY_POLL_INTERVAL_SECONDS = 20  # re-probe cadence while no region has capacity

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
