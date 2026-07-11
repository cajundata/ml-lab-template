import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_lab.config import (
    DATA_PROCESSED_SMOKE,
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    SMOKE_TRAIN_ROWS,
    TARGET_NAME,
)


def generate_smoke_data(
    seed: int = SMOKE_SEED, rows: int = SMOKE_ROWS
) -> pd.DataFrame:
    """Deterministic, linearly separable binary classification data."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, len(FEATURE_NAMES)))
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    score = (
        2.0 * df["x0"]
        - 1.25 * df["x1"]
        + 0.75 * df["x2"]
        + 0.25 * df["x3"]
    )
    df[TARGET_NAME] = (score > 0).astype(int)
    return df


def split_smoke_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First SMOKE_TRAIN_ROWS rows train, remaining rows test. No shuffle."""
    train_df = df.iloc[:SMOKE_TRAIN_ROWS].reset_index(drop=True)
    test_df = df.iloc[SMOKE_TRAIN_ROWS:].reset_index(drop=True)
    return train_df, test_df


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_smoke_data(dest: Path = DATA_PROCESSED_SMOKE) -> dict:
    """Generate, split, persist CSVs + checksummed manifest. Returns manifest."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    df = generate_smoke_data()
    train_df, test_df = split_smoke_data(df)

    train_path = dest / "train.csv"
    test_path = dest / "test.csv"
    manifest_path = dest / "split_manifest.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest = {
        "seed": SMOKE_SEED,
        "rows": SMOKE_ROWS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_names": list(FEATURE_NAMES),
        "target_name": TARGET_NAME,
        "train_sha256": _sha256_of_file(train_path),
        "test_sha256": _sha256_of_file(test_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
