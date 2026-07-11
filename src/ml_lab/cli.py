from pathlib import Path

import typer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml_lab.config import (
    DATA_PROCESSED_SMOKE,
    EXPERIMENT_NAME,
    FEATURE_NAMES,
    MLRUNS_DIR,
    MODELS_SMOKE_LATEST,
    REPORTS_SMOKE_LATEST,
    SMOKE_SEED,
    TARGET_NAME,
)
from ml_lab.data.synthetic import write_smoke_data
from ml_lab.evaluation.logistic_eval import evaluate_smoke_model
from ml_lab.models.logistic_smoke import train_smoke_model
from ml_lab.tracking.mlflow_utils import (
    log_evaluation_metrics,
    start_training_run,
)


def train_smoke(
    data_dir=DATA_PROCESSED_SMOKE,
    model_dir=MODELS_SMOKE_LATEST,
    reports_dir=REPORTS_SMOKE_LATEST,
    tracking_uri=MLRUNS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Generate data, train the smoke model, log to MLflow, persist artifacts."""
    manifest = write_smoke_data(dest=data_dir)
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    model = train_smoke_model(train_df)

    features = train_df[FEATURE_NAMES]
    y_true = train_df[TARGET_NAME]
    y_pred = model.predict(features)
    train_metrics = {
        "train_accuracy": float(accuracy_score(y_true, y_pred)),
        "train_f1": float(f1_score(y_true, y_pred)),
    }

    params = {
        "model_type": model.metadata["model_type"],
        "random_seed": SMOKE_SEED,
        "train_rows": manifest["train_rows"],
        "test_rows": manifest["test_rows"],
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "target_name": TARGET_NAME,
    }

    run_id = start_training_run(
        params,
        train_metrics,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    model.metadata["mlflow_run_id"] = run_id
    model.save(dest=model_dir)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "train_run_id.txt").write_text(
        run_id + "\n", encoding="utf-8"
    )
    return run_id


def evaluate_smoke(
    data_dir=DATA_PROCESSED_SMOKE,
    model_dir=MODELS_SMOKE_LATEST,
    reports_dir=REPORTS_SMOKE_LATEST,
    tracking_uri=MLRUNS_DIR,
) -> dict:
    """Evaluate the saved model and append eval metrics to the training run."""
    run_id_path = Path(reports_dir) / "train_run_id.txt"
    if not run_id_path.exists():
        raise FileNotFoundError(
            f"train_run_id.txt not found at {run_id_path}; run train first."
        )
    run_id = run_id_path.read_text(encoding="utf-8").strip()

    metrics = evaluate_smoke_model(
        model_dir=model_dir, data_dir=data_dir, reports_dir=reports_dir
    )
    eval_metrics = {
        k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")
    }
    log_evaluation_metrics(run_id, eval_metrics, tracking_uri=tracking_uri)
    return metrics


app = typer.Typer()


@app.command("train-smoke")
def train_smoke_command() -> None:
    run_id = train_smoke()
    typer.echo(f"Training run: {run_id}")


@app.command("evaluate-smoke")
def evaluate_smoke_command() -> None:
    metrics = evaluate_smoke()
    typer.echo(f"Evaluation metrics: {metrics}")
