import pytest

from ml_lab.gpu import constants
from ml_lab.gpu.constants import validate_timeout_budget


def test_default_ttl_fits_the_timeout_budget():
    # SSH(600) + BOOTSTRAP(1800) + BENCHMARK(1800) = 4200 < 7200
    validate_timeout_budget(constants.DEFAULT_TTL_SECONDS)  # no raise


def test_short_ttl_below_budget_raises():
    with pytest.raises(ValueError):
        validate_timeout_budget(900)


def test_pinned_constants_present():
    assert constants.DO_REGION == "nyc2"
    assert constants.DO_SIZE_SLUG == "gpu-h100x1-80gb"
    assert constants.DO_IMAGE_SLUG == "gpu-h100x1-base"
    assert constants.DROPLET_NAME_PREFIX == "ml-lab-gpu-"
    assert constants.DESTROY_POLL_INTERVAL_SECONDS == 10
    assert "ml-lab" in constants.BASE_TAGS


def test_pinned_spaces_constants_present():
    assert constants.SPACES_REGION == "nyc3"
    assert constants.SPACES_ENDPOINT == "https://nyc3.digitaloceanspaces.com"
    assert constants.SPACES_KEY_PREFIX == "ml-pathway/phase0"
