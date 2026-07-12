"""Typer surface for the GPU safety spine: `audit` and `down`.

`run` / `up` are added in a later slice. `scripts/do_gpu.py` imports this `app`.
"""

import typer

from ml_lab.gpu import audit as audit_mod
from ml_lab.gpu.lifecycle import gpu_run, gpu_up
from ml_lab.gpu.teardown import destroy_and_verify

app = typer.Typer(help="DigitalOcean GPU-lab lifecycle (Phase 0 safety spine).")


@app.command("audit")
def audit_command() -> None:
    """Report any billable lab GPU resources; exit nonzero if the account is dirty."""
    report = audit_mod.collect_audit()
    typer.echo(audit_mod.format_report(report))
    if not audit_mod.is_clean(report):
        raise typer.Exit(code=1)


@app.command("down")
def down_command(
    droplet_id: int = typer.Option(..., "--droplet-id", help="Droplet id to destroy."),
) -> None:
    """Destroy a droplet and verify it is gone and audit is clean."""
    destroy_and_verify(droplet_id)


@app.command("up")
def up_command(
    ttl_seconds: int = typer.Option(
        None,
        "--ttl-seconds",
        help="Short-fuse self-destruct test; bypasses the benchmark-budget check.",
    ),
) -> None:
    """Create a GPU droplet and hand it off once bootstrap is verified (leaves it alive)."""
    if ttl_seconds is None:
        gpu_up()
    else:
        gpu_up(ttl_seconds=ttl_seconds, enforce_budget=False)


@app.command("run")
def run_command() -> None:
    """Full GPU lifecycle: create, benchmark, pull, upload to Spaces, always destroy."""
    raise typer.Exit(code=gpu_run())
