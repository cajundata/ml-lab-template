import pandas as pd

from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    TARGET_NAME,
)
from ml_lab.data.synthetic import generate_smoke_data


def test_generate_is_deterministic_for_fixed_seed():
    a = generate_smoke_data(seed=SMOKE_SEED)
    b = generate_smoke_data(seed=SMOKE_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_generate_has_expected_columns_in_order():
    df = generate_smoke_data(seed=SMOKE_SEED)
    assert list(df.columns) == [*FEATURE_NAMES, TARGET_NAME]


def test_generate_has_expected_row_count():
    df = generate_smoke_data(seed=SMOKE_SEED)
    assert len(df) == SMOKE_ROWS


def test_generate_target_is_binary():
    df = generate_smoke_data(seed=SMOKE_SEED)
    assert set(df[TARGET_NAME].unique()).issubset({0, 1})


def test_generate_has_no_null_features():
    df = generate_smoke_data(seed=SMOKE_SEED)
    assert not df[FEATURE_NAMES].isnull().any().any()


def test_generate_label_rule_matches_hyperplane():
    df = generate_smoke_data(seed=SMOKE_SEED)
    score = (
        2.0 * df["x0"]
        - 1.25 * df["x1"]
        + 0.75 * df["x2"]
        + 0.25 * df["x3"]
    )
    expected = (score > 0).astype(int)
    assert (df[TARGET_NAME] == expected).all()
