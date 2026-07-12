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
    assert constants.DO_REGION == "atl1"
    assert constants.DO_SIZE_SLUG == "gpu-rtx4000x1-20gb"
    assert constants.DROPLET_NAME_PREFIX == "ml-lab-gpu-"
    assert constants.DESTROY_POLL_INTERVAL_SECONDS == 10
    assert "ml-lab" in constants.BASE_TAGS
