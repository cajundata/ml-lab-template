from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED_SMOKE = REPO_ROOT / "data" / "processed" / "smoke"
MODELS_SMOKE_LATEST = REPO_ROOT / "models" / "smoke" / "latest"
REPORTS_SMOKE_LATEST = REPO_ROOT / "reports" / "smoke" / "latest"
MLRUNS_DIR = REPO_ROOT / "mlruns"

SMOKE_SEED = 20260706
SMOKE_ROWS = 240
SMOKE_TRAIN_ROWS = 180
SMOKE_TEST_ROWS = 60
FEATURE_NAMES = ["x0", "x1", "x2", "x3"]
TARGET_NAME = "target"
EXPERIMENT_NAME = "phase0-smoke-lifecycle"