"""Entry point: `uv run python scripts/do_gpu.py {run,up,down,audit}` (Makefile wraps this)."""

from ml_lab.gpu.cli import app

if __name__ == "__main__":
    app()
