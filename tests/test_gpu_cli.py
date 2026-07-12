from typer.testing import CliRunner

from ml_lab.gpu import audit as audit_mod
from ml_lab.gpu import cli as gpu_cli
from ml_lab.gpu.audit import AuditReport, DropletInfo


def _info():
    return DropletInfo(
        id=555,
        name="ml-lab-gpu-phase0-x",
        status="active",
        region="atl1",
        size="gpu-rtx4000x1-20gb",
        image="img",
        public_ip="143.0.0.1",
        age="0h01m",
        ttl_expiry=None,
        overdue=False,
        destroy_command="make gpu-down DROPLET_ID=555",
    )


def test_help_names_audit_and_down():
    result = CliRunner().invoke(gpu_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output
    assert "down" in result.output


def test_down_calls_destroy_and_verify(monkeypatch):
    called = {}
    monkeypatch.setattr(gpu_cli, "destroy_and_verify", lambda did: called.setdefault("id", did))
    result = CliRunner().invoke(gpu_cli.app, ["down", "--droplet-id", "555"])
    assert result.exit_code == 0
    assert called["id"] == 555


def test_audit_exits_zero_when_clean(monkeypatch):
    monkeypatch.setattr(audit_mod, "collect_audit", lambda: AuditReport())
    result = CliRunner().invoke(gpu_cli.app, ["audit"])
    assert result.exit_code == 0
    assert "audit clean" in result.output


def test_audit_exits_nonzero_when_dirty(monkeypatch):
    monkeypatch.setattr(audit_mod, "collect_audit", lambda: AuditReport(droplets=[_info()]))
    result = CliRunner().invoke(gpu_cli.app, ["audit"])
    assert result.exit_code == 1
    assert "DIRTY" in result.output


def test_help_names_up():
    result = CliRunner().invoke(gpu_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "up" in result.output


def test_up_default_calls_gpu_up_with_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(gpu_cli, "gpu_up", lambda **k: seen.setdefault("kwargs", k))
    result = CliRunner().invoke(gpu_cli.app, ["up"])
    assert result.exit_code == 0
    assert seen["kwargs"] == {}  # no --ttl-seconds → gpu_up() defaults


def test_up_short_ttl_bypasses_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr(gpu_cli, "gpu_up", lambda **k: seen.setdefault("kwargs", k))
    result = CliRunner().invoke(gpu_cli.app, ["up", "--ttl-seconds", "900"])
    assert result.exit_code == 0
    assert seen["kwargs"] == {"ttl_seconds": 900, "enforce_budget": False}


def test_help_names_run():
    result = CliRunner().invoke(gpu_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output


def test_run_exits_zero_when_gpu_run_returns_zero(monkeypatch):
    monkeypatch.setattr(gpu_cli, "gpu_run", lambda: 0)
    result = CliRunner().invoke(gpu_cli.app, ["run"])
    assert result.exit_code == 0


def test_run_exits_with_gpu_run_nonzero_code(monkeypatch):
    monkeypatch.setattr(gpu_cli, "gpu_run", lambda: 7)
    result = CliRunner().invoke(gpu_cli.app, ["run"])
    assert result.exit_code == 7
