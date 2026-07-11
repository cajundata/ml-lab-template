import json

import pandas as pd

from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    SMOKE_TEST_ROWS,
    SMOKE_TRAIN_ROWS,
    TARGET_NAME,
)
from ml_lab.data.synthetic import (
    generate_smoke_data,
    split_smoke_data,
    write_smoke_data,
)


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
    pd.testing.assert_series_equal(
        train_df.iloc[0], df.iloc[0], check_names=False
    )
    pd.testing.assert_series_equal(
        train_df.iloc[-1], df.iloc[SMOKE_TRAIN_ROWS - 1], check_names=False
    )
    pd.testing.assert_series_equal(
        test_df.iloc[0], df.iloc[SMOKE_TRAIN_ROWS], check_names=False
    )
    pd.testing.assert_series_equal(
        test_df.iloc[-1], df.iloc[-1], check_names=False
    )


def test_split_columns_match_source():
    df = generate_smoke_data(seed=SMOKE_SEED)
    train_df, test_df = split_smoke_data(df)
    assert list(train_df.columns) == [*FEATURE_NAMES, TARGET_NAME]
    assert list(test_df.columns) == [*FEATURE_NAMES, TARGET_NAME]


def test_write_creates_expected_files(tmp_path):
    write_smoke_data(dest=tmp_path)
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "split_manifest.json").exists()


def test_write_returns_manifest_with_expected_keys(tmp_path):
    manifest = write_smoke_data(dest=tmp_path)
    assert manifest["seed"] == SMOKE_SEED
    assert manifest["rows"] == SMOKE_ROWS
    assert manifest["train_rows"] == SMOKE_TRAIN_ROWS
    assert manifest["test_rows"] == SMOKE_TEST_ROWS
    assert manifest["feature_names"] == list(FEATURE_NAMES)
    assert manifest["target_name"] == TARGET_NAME
    assert "train_sha256" in manifest
    assert "test_sha256" in manifest


def test_manifest_row_counts_match_csv_files(tmp_path):
    manifest = write_smoke_data(dest=tmp_path)
    train_df = pd.read_csv(tmp_path / "train.csv")
    test_df = pd.read_csv(tmp_path / "test.csv")
    assert manifest["train_rows"] == len(train_df)
    assert manifest["test_rows"] == len(test_df)


def test_manifest_feature_order_matches_csv_columns(tmp_path):
    manifest = write_smoke_data(dest=tmp_path)
    train_df = pd.read_csv(tmp_path / "train.csv")
    assert manifest["feature_names"] == [
        c for c in train_df.columns if c != TARGET_NAME
    ]


def test_manifest_on_disk_matches_returned_manifest(tmp_path):
    manifest = write_smoke_data(dest=tmp_path)
    on_disk = json.loads((tmp_path / "split_manifest.json").read_text())
    assert on_disk == manifest


def test_checksums_are_stable_across_regeneration(tmp_path):
    first = write_smoke_data(dest=tmp_path / "run_a")
    second = write_smoke_data(dest=tmp_path / "run_b")
    assert first["train_sha256"] == second["train_sha256"]
    assert first["test_sha256"] == second["test_sha256"]
