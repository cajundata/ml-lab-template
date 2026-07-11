# Phase 0 Synthetic Data Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver deterministic synthetic smoke-data generation, splitting, and persistence (with a checksummed split manifest) plus its full test suite, as the first Phase 0 checkpoint.

**Architecture:** A single module `src/ml_lab/data/synthetic.py` exposes three pure-ish functions — `generate_smoke_data` (seeded features + deterministic linear label rule), `split_smoke_data` (first 180 train / last 60 test, no shuffle), and `write_smoke_data` (generate → split → write CSVs + `split_manifest.json` with SHA-256 checksums). All constants come from the existing `src/ml_lab/config.py`. Tests write into temp dirs so they never touch the repo's real `data/processed/smoke/`.

**Tech Stack:** Python 3.11+, numpy, pandas, pytest, uv. Run tests with `uv run pytest`.

---

## File Structure

- **Modify:** `src/ml_lab/data/synthetic.py` — the whole slice's logic. Currently a broken partial (references config constants without importing them).
- **Depends on (already exists, currently untracked):** `src/ml_lab/config.py` — provides `SMOKE_SEED`, `SMOKE_ROWS`, `SMOKE_TRAIN_ROWS`, `SMOKE_TEST_ROWS`, `FEATURE_NAMES`, `TARGET_NAME`, `DATA_PROCESSED_SMOKE`. Committed in Task 1 alongside `synthetic.py` since the module imports it.
- **Create:** `tests/test_synthetic_data.py` — the Data-tests assertion list.

pytest is configured (`pyproject.toml`) with `pythonpath = ["src"]` and `testpaths = ["tests"]`, so imports use `from ml_lab.config import ...` and `from ml_lab.data.synthetic import ...`.

---

### Task 1: `generate_smoke_data` — seeded features + deterministic labels

**Files:**
- Modify: `src/ml_lab/data/synthetic.py`
- Test: `tests/test_synthetic_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_synthetic_data.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_synthetic_data.py -v`
Expected: FAIL — `ImportError`/`NameError` because `synthetic.py` doesn't import its constants and the module currently errors on import.

- [ ] **Step 3: Rewrite `synthetic.py` with proper imports and generation**

Replace the entire contents of `src/ml_lab/data/synthetic.py` with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_synthetic_data.py -v`
Expected: PASS — all six tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/config.py src/ml_lab/data/synthetic.py tests/test_synthetic_data.py
git commit -m "feat: deterministic synthetic smoke-data generation"
```

---

### Task 2: `split_smoke_data` — first 180 train / last 60 test

**Files:**
- Modify: `src/ml_lab/data/synthetic.py`
- Test: `tests/test_synthetic_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthetic_data.py`. First extend the imports:

```python
from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    SMOKE_TEST_ROWS,
    SMOKE_TRAIN_ROWS,
    TARGET_NAME,
)
from ml_lab.data.synthetic import generate_smoke_data, split_smoke_data
```

Then add:

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_synthetic_data.py -k split -v`
Expected: FAIL — `ImportError: cannot import name 'split_smoke_data'`.

- [ ] **Step 3: Add `split_smoke_data` to `synthetic.py`**

Add the split constant and the function. Update the config import block to:

```python
from ml_lab.config import (
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    SMOKE_TRAIN_ROWS,
    TARGET_NAME,
)
```

Append after `generate_smoke_data`:

```python
def split_smoke_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First SMOKE_TRAIN_ROWS rows train, remaining rows test. No shuffle."""
    train_df = df.iloc[:SMOKE_TRAIN_ROWS].reset_index(drop=True)
    test_df = df.iloc[SMOKE_TRAIN_ROWS:].reset_index(drop=True)
    return train_df, test_df
```

(The split is defined by the train boundary, so the tail is exactly the remaining 60 rows — `SMOKE_TEST_ROWS` is asserted in the tests, not needed inside the module.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_synthetic_data.py -v`
Expected: PASS — all generation and split tests green.

- [ ] **Step 5: Commit**

```bash
git add src/ml_lab/data/synthetic.py tests/test_synthetic_data.py
git commit -m "feat: deterministic 180/60 train-test split"
```

---

### Task 3: `write_smoke_data` — persist CSVs + checksummed manifest

**Files:**
- Modify: `src/ml_lab/data/synthetic.py`
- Test: `tests/test_synthetic_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthetic_data.py`. Extend imports:

```python
import json

from ml_lab.data.synthetic import (
    generate_smoke_data,
    split_smoke_data,
    write_smoke_data,
)
```

Then add:

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_synthetic_data.py -k write -v`
Expected: FAIL — `ImportError: cannot import name 'write_smoke_data'`.

- [ ] **Step 3: Add `write_smoke_data` + helpers to `synthetic.py`**

Add these imports at the top of `synthetic.py` (above the numpy/pandas imports):

```python
import hashlib
import json
from pathlib import Path
```

Add `DATA_PROCESSED_SMOKE` to the config import block:

```python
from ml_lab.config import (
    DATA_PROCESSED_SMOKE,
    FEATURE_NAMES,
    SMOKE_ROWS,
    SMOKE_SEED,
    SMOKE_TRAIN_ROWS,
    TARGET_NAME,
)
```

Append after `split_smoke_data`:

```python
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
```

- [ ] **Step 4: Run the full test file to verify it passes**

Run: `uv run pytest tests/test_synthetic_data.py -v`
Expected: PASS — all generation, split, and write/manifest tests green.

- [ ] **Step 5: Sanity-check the real artifacts (optional, non-committed)**

Run: `uv run python -c "from ml_lab.data.synthetic import write_smoke_data; import json; print(json.dumps(write_smoke_data(), indent=2))"`
Expected: prints the manifest; `data/processed/smoke/{train.csv,test.csv,split_manifest.json}` now exist locally (gitignored).

- [ ] **Step 6: Commit**

```bash
git add src/ml_lab/data/synthetic.py tests/test_synthetic_data.py
git commit -m "feat: persist smoke splits with checksummed manifest"
```

---

## Definition of Done

- `uv run pytest tests/test_synthetic_data.py -v` is fully green.
- `src/ml_lab/data/synthetic.py` exports `generate_smoke_data`, `split_smoke_data`, `write_smoke_data`.
- `split_manifest.json` carries seed, row counts, ordered feature names, target name, and stable SHA-256 checksums.
- No CLI, model, evaluation, MLflow, or GPU code added (deferred to later slices).
