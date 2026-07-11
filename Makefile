.PHONY: setup test train evaluate mlflow-ui clean gpu-run gpu-up gpu-down gpu-audit

setup:
	uv sync --dev

test:
	uv run pytest

train:
	uv run ml-lab train-smoke

evaluate:
	uv run ml-lab evaluate-smoke

mlflow-ui:
	uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

clean:
	rm -rf .pytest_cache
	rm -rf data/processed/*
	rm -rf models/smoke/*
	rm -rf reports/smoke/*

gpu-run:
	uv run python scripts/do_gpu.py run

gpu-up:
	uv run python scripts/do_gpu.py up

gpu-down:
	@test -n "$(DROPLET_ID)" || (echo "DROPLET_ID is required: make gpu-down DROPLET_ID=<id>" && exit 1)
	uv run python scripts/do_gpu.py down --droplet-id $(DROPLET_ID)

gpu-audit:
	uv run python scripts/do_gpu.py audit