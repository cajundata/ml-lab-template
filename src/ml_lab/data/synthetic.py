import numpy as np
import pandas as pd

from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
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
