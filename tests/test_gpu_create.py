import pytest

from ml_lab.gpu import create, do_client
from ml_lab.gpu.create import ConstantsError


def _valid(monkeypatch):
    """Mock do_client so validate_constants passes for the corrected constants."""
    monkeypatch.setattr(do_client, "list_region_slugs", lambda: ["atl1"])
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["atl1"]}]
    )
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["gpu-h100x1-base"])


def test_validate_constants_passes(monkeypatch):
    _valid(monkeypatch)
    create.validate_constants()  # no raise


def test_validate_constants_size_missing(monkeypatch):
    _valid(monkeypatch)
    monkeypatch.setattr(do_client, "list_sizes", lambda: [{"slug": "other", "regions": ["atl1"]}])
    with pytest.raises(ConstantsError):
        create.validate_constants()


def test_validate_constants_ignores_region(monkeypatch):
    # Region is resolved live at create time (resolve_region), not gated here, so a
    # size whose live regions exclude DO_REGION still passes preflight.
    _valid(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["sfo3"]}]
    )
    create.validate_constants()  # no raise


# --- resolve_region: GPU capacity shifts regions within minutes, so the create
# --- region is resolved from DO's live per-size regions rather than statically pinned.


def test_resolve_region_prefers_do_region(monkeypatch):
    # DO_REGION is atl1; when the size is live-available there, prefer it.
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["nyc2", "atl1"]}],
    )
    assert create.resolve_region("gpu-h200x1-141gb") == "atl1"


def test_resolve_region_picks_available_when_do_region_absent(monkeypatch):
    # atl1 capacity gone; DO now reports the size only in nyc2 -> create in nyc2.
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["nyc2"]}]
    )
    assert create.resolve_region("gpu-h200x1-141gb") == "nyc2"


def test_resolve_region_falls_back_when_null(monkeypatch):
    # DO reports no regions (common for GPU SKUs) -> fall back to DO_REGION (atl1),
    # let the create call's DO 422 be the authority.
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": None}]
    )
    assert create.resolve_region("gpu-h200x1-141gb") == "atl1"


def test_resolve_region_size_not_found(monkeypatch):
    monkeypatch.setattr(do_client, "list_sizes", lambda: [{"slug": "other", "regions": ["atl1"]}])
    with pytest.raises(ConstantsError):
        create.resolve_region("gpu-h200x1-141gb")


def test_resolve_region_excludes_tried(monkeypatch):
    # atl1 already 422'd this create; the next resolve must skip it and pick nyc2.
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["atl1", "nyc2"]}],
    )
    assert create.resolve_region("gpu-h200x1-141gb", exclude={"atl1"}) == "nyc2"


def test_resolve_region_none_when_no_candidate(monkeypatch):
    # only atl1 (== DO_REGION) reported, and it's excluded -> no region left to try.
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["atl1"]}]
    )
    assert create.resolve_region("gpu-h200x1-141gb", exclude={"atl1"}) is None


def test_validate_constants_image_missing(monkeypatch):
    _valid(monkeypatch)
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["other"])
    with pytest.raises(ConstantsError):
        create.validate_constants()


def test_generate_run_id_format():
    from datetime import datetime, timezone

    now = 1783728000.0  # a fixed instant
    expected_date = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d")
    run_id = create.generate_run_id(now)
    assert run_id.startswith(expected_date + "-")
    date, suffix = run_id.split("-")
    assert len(date) == 8 and date.isdigit()
    assert len(suffix) == 6 and all(c in "0123456789abcdef" for c in suffix)


def test_build_tags():
    tags = create.build_tags("20260711-abc123", 1783735200)
    assert tags[:4] == ["ml-lab", "ml-pathway", "phase-0", "owner-weldon"]
    assert tags[-2:] == ["run-20260711-abc123", "ttl-expiry-1783735200"]


def _happy(monkeypatch):
    """Full happy-path environment; returns a dict capturing create_droplet kwargs."""
    monkeypatch.setattr(do_client, "list_lab_droplets", lambda: [])
    monkeypatch.setattr(do_client, "probe_destroy_token", lambda: None)
    _valid(monkeypatch)  # so real validate_constants passes
    calls = {}

    def fake_create(**kwargs):
        calls["kwargs"] = kwargs
        return {"id": 42, "name": kwargs["name"], "status": "new"}

    monkeypatch.setattr(do_client, "create_droplet", fake_create)
    return calls


def test_create_lab_droplet_happy(monkeypatch, capsys):
    calls = _happy(monkeypatch)
    result = create.create_lab_droplet(
        "#cloud-config\n", run_id="20260711-abc123", now=1783728000.0
    )
    assert result == {
        "id": 42,
        "name": "ml-lab-gpu-phase0-20260711-abc123",
        "run_id": "20260711-abc123",
        "region": "atl1",
    }
    kw = calls["kwargs"]
    assert kw["name"] == "ml-lab-gpu-phase0-20260711-abc123"
    assert kw["region"] == "atl1"
    assert kw["size"] == "gpu-h200x1-141gb"
    assert kw["image"] == "gpu-h100x1-base"
    assert kw["user_data"] == "#cloud-config\n"
    assert "run-20260711-abc123" in kw["tags"]
    assert f"ttl-expiry-{1783728000 + 7200}" in kw["tags"]
    out = capsys.readouterr().out
    assert "id: 42" in out
    assert "ml-lab-gpu-phase0-20260711-abc123" in out
    assert "local_pid:" in out


def test_create_lab_droplet_uses_resolved_region(monkeypatch, capsys):
    # atl1 (DO_REGION) capacity gone; DO reports the size only in nyc2 now. Create
    # must follow the live signal to nyc2, and the printed/returned region matches.
    calls = _happy(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["nyc2"]}]
    )
    result = create.create_lab_droplet(
        "#cloud-config\n", run_id="20260711-abc123", now=1783728000.0
    )
    assert calls["kwargs"]["region"] == "nyc2"
    assert result["region"] == "nyc2"
    assert "region: nyc2" in capsys.readouterr().out


def test_create_lab_droplet_retries_region_on_422(monkeypatch):
    # GPU capacity flips between resolve and create: atl1 422s "not available in this
    # region", so create must retry the size's next live region (nyc2) and land there.
    _happy(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["atl1", "nyc2"]}],
    )
    tried = []

    def flaky_create(**kw):
        tried.append(kw["region"])
        if kw["region"] == "atl1":
            raise do_client.DOClientError(
                'doctl ... failed: {"errors":[{"detail":"POST ...: 422 ... '
                'Size is not available in this region."}]}'
            )
        return {"id": 42, "name": kw["name"], "status": "new"}

    monkeypatch.setattr(do_client, "create_droplet", flaky_create)
    result = create.create_lab_droplet("#cloud-config\n", run_id="r", now=1783728000.0)
    assert tried == ["atl1", "nyc2"]  # 422 on atl1, retried nyc2
    assert result["region"] == "nyc2"


def test_create_lab_droplet_reraises_non_capacity_error(monkeypatch):
    # A non-capacity doctl error must NOT be swallowed by the region retry loop.
    _happy(monkeypatch)

    def boom(**kw):
        raise do_client.DOClientError("doctl ... failed: some other 500 error")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(do_client.DOClientError, match="500"):
        create.create_lab_droplet("#cloud-config\n", run_id="r", now=1783728000.0)


def test_create_lab_droplet_raises_when_no_capacity_window(monkeypatch):
    # Every region 422s and the bounded wait elapses -> a clear no-capacity error.
    _happy(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h200x1-141gb", "regions": ["atl1"]}]
    )

    def always_422(**kw):
        raise do_client.DOClientError(
            'doctl ... failed: {"errors":[{"detail":"422 ... Size is not available in this region."}]}'
        )

    monkeypatch.setattr(do_client, "create_droplet", always_422)
    monkeypatch.setattr(create.time, "sleep", lambda s: None)
    clock = iter([0.0] + [10_000.0] * 20)
    monkeypatch.setattr(create.time, "monotonic", lambda: next(clock))
    with pytest.raises(ConstantsError, match="capacity"):
        create.create_lab_droplet("#cloud-config\n", run_id="r", now=1783728000.0)


def test_create_lab_droplet_refuses_when_exists(monkeypatch):
    _happy(monkeypatch)
    monkeypatch.setattr(do_client, "list_lab_droplets", lambda: [{"id": 1, "tags": ["ml-lab"]}])

    def boom(**k):
        raise AssertionError("create_droplet must not be called when a droplet exists")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(create.LabDropletExistsError):
        create.create_lab_droplet("#cloud-config\n")


def test_create_lab_droplet_ttl_expiry_tag(monkeypatch):
    calls = _happy(monkeypatch)
    create.create_lab_droplet(
        "#cloud-config\n", ttl_seconds=3600, enforce_budget=False, run_id="r", now=1000.0
    )
    assert "ttl-expiry-4600" in calls["kwargs"]["tags"]


def test_create_lab_droplet_enforce_budget_true_raises(monkeypatch):
    _happy(monkeypatch)

    def boom(**k):
        raise AssertionError("create_droplet must not be called when the budget check fails")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(ValueError):
        create.create_lab_droplet("#cloud-config\n", ttl_seconds=900)


def test_create_lab_droplet_enforce_budget_false_allows_short_ttl(monkeypatch):
    _happy(monkeypatch)
    result = create.create_lab_droplet(
        "#cloud-config\n", ttl_seconds=900, enforce_budget=False, run_id="r", now=1000.0
    )
    assert result["id"] == 42


def test_create_lab_droplet_passes_ssh_keys(monkeypatch):
    calls = _happy(monkeypatch)
    create.create_lab_droplet(
        "#cloud-config\n", ssh_key_ids=["aa:bb"], run_id="r", now=1000.0
    )
    assert calls["kwargs"]["ssh_key_ids"] == ["aa:bb"]


def test_create_lab_droplet_validates_before_create(monkeypatch):
    _happy(monkeypatch)
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["other"])  # image now missing

    def boom(**k):
        raise AssertionError("create_droplet must not be called when constants invalid")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(create.ConstantsError):
        create.create_lab_droplet("#cloud-config\n")
