import pandas as pd

from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    SMOKE_TEST_ROWS,
    SMOKE_TRAIN_ROWS,
    TARGET_NAME,
)
from ml_lab.data.synthetic import generate_smoke_data, split_smoke_data


def test_generate_is_deterministic_for_fixed_seed():
    a = generate_smoke_data(seed=SMOKE_SEED)
    b = generate_smoke_data(seed=SMOKE_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_generate_differs_for_different_seed():
    a = generate_smoke_data(seed=SMOKE_SEED)
    b = generate_smoke_data(seed=0)
    assert not a.equals(b)


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


def test_split_train_row_count():
    train_df, _ = split_smoke_data(generate_smoke_data(seed=SMOKE_SEED))
    assert len(train_df) == SMOKE_TRAIN_ROWS


def test_split_test_row_count():
    _, test_df = split_smoke_data(generate_smoke_data(seed=SMOKE_SEED))
    assert len(test_df) == SMOKE_TEST_ROWS


def test_split_preserves_order_no_shuffle():
    df = generate_smoke_data(seed=SMOKE_SEED)
    train_df, test_df = split_smoke_data(df)
    pd.testing.assert_frame_equal(
        train_df, df.iloc[:SMOKE_TRAIN_ROWS].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        test_df, df.iloc[SMOKE_TRAIN_ROWS:].reset_index(drop=True)
    )


def test_split_columns_match_source():
    df = generate_smoke_data(seed=SMOKE_SEED)
    train_df, test_df = split_smoke_data(df)
    assert list(train_df.columns) == [*FEATURE_NAMES, TARGET_NAME]
    assert list(test_df.columns) == [*FEATURE_NAMES, TARGET_NAME]
