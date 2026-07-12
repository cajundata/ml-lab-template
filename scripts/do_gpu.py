"""Entry point: `uv run python scripts/do_gpu.py {audit,down}` (Makefile wraps this)."""

from ml_lab.gpu.cli import app

if __name__ == "__main__":
    app()
