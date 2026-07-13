import pytest

from ml_lab.gpu import create, do_client
from ml_lab.gpu.create import ConstantsError


def _all_sizes(h100_regions=("nyc2",), h200_regions=("nyc2",), l40s_regions=("tor1",)):
    """The three acceptable SKUs as DO would list them (regions tunable per test)."""
    return [
        {"slug": "gpu-h100x1-80gb", "regions": list(h100_regions)},
        {"slug": "gpu-h200x1-141gb", "regions": list(h200_regions)},
        {"slug": "gpu-l40sx1-48gb", "regions": list(l40s_regions)},
    ]


def _valid(monkeypatch):
    """Mock do_client so validate_constants passes for the corrected constants."""
    monkeypatch.setattr(do_client, "list_region_slugs", lambda: ["nyc2", "atl1", "tor1"])
    monkeypatch.setattr(do_client, "list_sizes", _all_sizes)
    monkeypatch.setattr(do_client, "list_image_slugs", lambda: ["gpu-h100x1-base"])


def test_validate_constants_passes(monkeypatch):
    _valid(monkeypatch)
    create.validate_constants()  # no raise


def test_validate_constants_size_missing(monkeypatch):
    # An acceptable SKU absent from the account -> preflight fails.
    _valid(monkeypatch)
    monkeypatch.setattr(do_client, "list_sizes", lambda: [{"slug": "other", "regions": ["atl1"]}])
    with pytest.raises(ConstantsError):
        create.validate_constants()


def test_validate_constants_ignores_region(monkeypatch):
    # Region/capacity is resolved live at create time, not gated here: acceptable SKUs
    # present (even with regions that exclude DO_REGION) still pass preflight.
    _valid(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: _all_sizes(h100_regions=["sfo3"], h200_regions=["sfo3"], l40s_regions=["sfo3"]),
    )
    create.validate_constants()  # no raise


# --- resolve_region: GPU capacity shifts regions within minutes, so the create
# --- region is resolved from DO's live per-size regions rather than statically pinned.


def test_resolve_region_prefers_do_region(monkeypatch):
    # DO_REGION is nyc2; when the size is live-available there, prefer it.
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: [{"slug": "gpu-h100x1-80gb", "regions": ["atl1", "nyc2"]}],
    )
    assert create.resolve_region("gpu-h100x1-80gb") == "nyc2"


def test_resolve_region_picks_available_when_do_region_absent(monkeypatch):
    # nyc2 (DO_REGION) capacity gone; DO now reports the size only in atl1 -> use atl1.
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h100x1-80gb", "regions": ["atl1"]}]
    )
    assert create.resolve_region("gpu-h100x1-80gb") == "atl1"


def test_resolve_region_falls_back_when_null(monkeypatch):
    # DO reports no regions (common for GPU SKUs) -> fall back to DO_REGION (nyc2),
    # let the create call's DO 422 be the authority.
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h100x1-80gb", "regions": None}]
    )
    assert create.resolve_region("gpu-h100x1-80gb") == "nyc2"


def test_resolve_region_size_not_found(monkeypatch):
    monkeypatch.setattr(do_client, "list_sizes", lambda: [{"slug": "other", "regions": ["nyc2"]}])
    with pytest.raises(ConstantsError):
        create.resolve_region("gpu-h100x1-80gb")


def test_resolve_region_excludes_tried(monkeypatch):
    # nyc2 (DO_REGION) already 422'd this create; the next resolve must skip it -> atl1.
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: [{"slug": "gpu-h100x1-80gb", "regions": ["atl1", "nyc2"]}],
    )
    assert create.resolve_region("gpu-h100x1-80gb", exclude={"nyc2"}) == "atl1"


def test_resolve_region_none_when_no_candidate(monkeypatch):
    # only nyc2 (== DO_REGION) reported, and it's excluded -> no region left to try.
    monkeypatch.setattr(
        do_client, "list_sizes", lambda: [{"slug": "gpu-h100x1-80gb", "regions": ["nyc2"]}]
    )
    assert create.resolve_region("gpu-h100x1-80gb", exclude={"nyc2"}) is None


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
        "size": "gpu-h100x1-80gb",  # first acceptable SKU, has capacity
        "region": "nyc2",
    }
    kw = calls["kwargs"]
    assert kw["name"] == "ml-lab-gpu-phase0-20260711-abc123"
    assert kw["region"] == "nyc2"
    assert kw["size"] == "gpu-h100x1-80gb"
    assert kw["image"] == "gpu-h100x1-base"
    assert kw["user_data"] == "#cloud-config\n"
    assert "run-20260711-abc123" in kw["tags"]
    assert f"ttl-expiry-{1783728000 + 7200}" in kw["tags"]
    out = capsys.readouterr().out
    assert "id: 42" in out
    assert "size: gpu-h100x1-80gb" in out
    assert "local_pid:" in out


def test_create_lab_droplet_uses_resolved_region(monkeypatch, capsys):
    # nyc2 (DO_REGION) capacity gone for the first SKU; DO reports it only in atl1 now.
    # Create must follow the live signal to atl1; printed/returned region matches.
    calls = _happy(monkeypatch)
    monkeypatch.setattr(do_client, "list_sizes", lambda: _all_sizes(h100_regions=["atl1"]))
    result = create.create_lab_droplet(
        "#cloud-config\n", run_id="20260711-abc123", now=1783728000.0
    )
    assert calls["kwargs"]["size"] == "gpu-h100x1-80gb"
    assert calls["kwargs"]["region"] == "atl1"
    assert result["region"] == "atl1"
    assert "region: atl1" in capsys.readouterr().out


def test_create_lab_droplet_retries_region_on_422(monkeypatch):
    # Capacity flips between resolve and create: preferred region nyc2 422s, so create
    # retries the same SKU's next live region (atl1) before moving to another SKU.
    _happy(monkeypatch)
    monkeypatch.setattr(do_client, "list_sizes", lambda: _all_sizes(h100_regions=["atl1", "nyc2"]))
    tried = []

    def flaky_create(**kw):
        tried.append((kw["size"], kw["region"]))
        if kw["region"] == "nyc2":
            raise do_client.DOClientError(
                'doctl ... failed: {"errors":[{"detail":"POST ...: 422 ... '
                'Size is not available in this region."}]}'
            )
        return {"id": 42, "name": kw["name"], "status": "new"}

    monkeypatch.setattr(do_client, "create_droplet", flaky_create)
    result = create.create_lab_droplet("#cloud-config\n", run_id="r", now=1783728000.0)
    assert tried == [("gpu-h100x1-80gb", "nyc2"), ("gpu-h100x1-80gb", "atl1")]
    assert result["size"] == "gpu-h100x1-80gb"
    assert result["region"] == "atl1"


def test_create_lab_droplet_falls_back_to_next_sku(monkeypatch):
    # The first SKU (H100) has no capacity anywhere; create must fall through to the
    # next acceptable SKU (H200) and land there.
    _happy(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: _all_sizes(h100_regions=["nyc2"], h200_regions=["nyc2"]),
    )
    landed = []

    def flaky_create(**kw):
        if kw["size"] == "gpu-h100x1-80gb":
            raise do_client.DOClientError(
                'doctl ... failed: {"errors":[{"detail":"422 ... Size is not available in this region."}]}'
            )
        landed.append(kw["size"])
        return {"id": 42, "name": kw["name"], "status": "new"}

    monkeypatch.setattr(do_client, "create_droplet", flaky_create)
    result = create.create_lab_droplet("#cloud-config\n", run_id="r", now=1783728000.0)
    assert landed == ["gpu-h200x1-141gb"]
    assert result["size"] == "gpu-h200x1-141gb"


def test_create_lab_droplet_reraises_non_capacity_error(monkeypatch):
    # A non-capacity doctl error must NOT be swallowed by the retry loop.
    _happy(monkeypatch)

    def boom(**kw):
        raise do_client.DOClientError("doctl ... failed: some other 500 error")

    monkeypatch.setattr(do_client, "create_droplet", boom)
    with pytest.raises(do_client.DOClientError, match="500"):
        create.create_lab_droplet("#cloud-config\n", run_id="r", now=1783728000.0)


def test_create_lab_droplet_raises_when_no_capacity_window(monkeypatch):
    # Every SKU+region 422s and the bounded wait elapses -> a clear no-capacity error.
    _happy(monkeypatch)
    monkeypatch.setattr(
        do_client, "list_sizes",
        lambda: _all_sizes(h100_regions=["nyc2"], h200_regions=["nyc2"], l40s_regions=["nyc2"]),
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
