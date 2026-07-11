# ml-lab-template

## Setup
```bash
uv sync --dev
```

## Running tests
```bash
make test
```
## Training
```bash
make train
```

## Evaluation
```bash
make evaluate
```

## MLflow UI
```bash
make mlflow-ui
```

## GPU teardown warning
GPU droplets must be destroyed, never powered off. Run 
```bash
make gpu-audit
```
if you are unsure anything is still running.