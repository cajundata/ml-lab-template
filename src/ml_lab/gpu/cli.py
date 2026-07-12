"""Typer surface for the GPU safety spine: `audit` and `down`.

`run` / `up` are added in a later slice. `scripts/do_gpu.py` imports this `app`.
"""

import typer

from ml_lab.gpu import audit as audit_mod
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
