"""
API contract tests.

The API serves prebuilt artifacts, so these tests build a small real artifact
set once per session into a temp directory and point the store at it. That
exercises the same load path production uses, rather than mocking it — a
contract test that stubs out artifact loading would not catch the failure mode
that actually matters (a deployment whose artifacts do not match its code).

Nothing here touches the network: the fixture site is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gridguard.api import main as api_main
from gridguard.api.store import ArtifactStore

FIXTURE_SITE = "gmu_fairfax"


@pytest.fixture(scope="session")
def built_store(tmp_path_factory):
    """Build a minimal artifact set and return a store loaded from it."""
    root = tmp_path_factory.mktemp("artifacts")
    model_dir, data_dir = root / "models", root / "data"

    from gridguard.artifacts.manifest import MANIFEST_FILENAME, ArtifactManifest
    from gridguard.pipeline import build_site

    result = build_site(
        FIXTURE_SITE,
        model_dir=model_dir,
        data_dir=data_dir,
        model_names=["linear"],  # keep the build fast
        include_physics=False,
        use_cache=False,
        demo=True,
        start="2016-01-01",
        end="2016-12-31",
    )

    manifest = ArtifactManifest()
    manifest.sites.append(result.artifact)
    manifest.save(model_dir / MANIFEST_FILENAME)

    return ArtifactStore(model_dir=model_dir, data_dir=data_dir).load()


@pytest.fixture(scope="session")
def client(built_store):
    """A TestClient whose module-level store is the built fixture store."""
    original = api_main.store
    api_main.store = built_store
    # lifespan would reload from the default directories, so bypass it.
    with TestClient(api_main.app, raise_server_exceptions=True) as test_client:
        api_main.store = built_store
        yield test_client
    api_main.store = original


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_loaded_artifacts(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sites_loaded"] >= 1
    assert body["manifest_present"] is True
    assert body["data_rows"] > 0
    assert body["detection_method"] in ("conformal", "sigma")


def test_health_leaks_no_filesystem_paths(client):
    """Health output must be safe to expose publicly."""
    body = client.get("/health").json()
    serialised = str(body)
    assert "/home/" not in serialised
    assert "/tmp/" not in serialised


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def test_sites_endpoint_lists_built_sites(client):
    body = client.get("/sites").json()
    assert body["total"] >= 1
    assert any(s["site_id"] == FIXTURE_SITE for s in body["sites"])


def test_every_site_carries_a_disclaimer(client):
    """No client can render a site without knowing its provenance."""
    for site in client.get("/sites").json()["sites"]:
        assert site["data_mode"] in ("real", "synthetic")
        assert site["disclaimer"]


def test_site_detail_endpoint(client):
    body = client.get(f"/sites/{FIXTURE_SITE}").json()
    assert body["site_id"] == FIXTURE_SITE
    assert body["capacity_kw"] > 0


def test_unknown_site_returns_404(client):
    response = client.get("/sites/not_a_real_site")
    assert response.status_code == 404
    assert "Available" in response.json()["detail"]


def test_site_timeseries_includes_expected_band(client):
    body = client.get(f"/sites/{FIXTURE_SITE}/timeseries", params={"limit": 50}).json()
    assert body["site_id"] == FIXTURE_SITE
    assert body["disclaimer"]
    assert len(body["points"]) <= 50
    point = body["points"][0]
    assert {"actual_kw", "predicted_kw", "expected_lower_kw"} <= set(point)


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------


def test_fleet_summary_aggregates(client):
    body = client.get("/fleet/summary").json()
    assert body["total_sites"] >= 1
    assert body["total_capacity_kw"] > 0
    counted = (
        body["sites_healthy"]
        + body["sites_warning"]
        + body["sites_critical"]
        + body["sites_no_data"]
    )
    assert counted == body["total_sites"]
    assert body["bounds"]["min_latitude"] < body["bounds"]["max_latitude"]


def test_fleet_summary_site_entries_have_health_and_mode(client):
    for site in client.get("/fleet/summary").json()["sites"]:
        assert site["health"] in ("healthy", "warning", "critical", "no_data")
        assert site["data_mode"] in ("real", "synthetic")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_endpoint(client):
    body = client.get("/events", params={"limit": 5}).json()
    assert body["total_events"] >= 1
    assert len(body["events"]) <= 5
    event = body["events"][0]
    assert event["severity"] in ("low", "medium", "high")
    assert event["explanation"]


def test_events_severity_filter(client):
    body = client.get("/events", params={"severity": "high", "limit": 20}).json()
    assert all(e["severity"] == "high" for e in body["events"])


def test_events_rejects_invalid_severity(client):
    assert client.get("/events", params={"severity": "catastrophic"}).status_code == 422


def test_event_detail_includes_spatial_context_and_actions(client):
    events = client.get("/events", params={"limit": 1}).json()["events"]
    event_id = events[0]["global_event_id"] or str(events[0]["event_id"])
    body = client.get(f"/events/{event_id}").json()

    assert body["event"]["event_id"] == events[0]["event_id"]
    assert body["recommended_actions"]
    # A single-site build has no neighbours, so scope must be indeterminate
    # rather than a fabricated verdict.
    if body["spatial_context"]:
        assert body["spatial_context"]["scope"] in (
            "site_specific",
            "regional",
            "indeterminate",
        )


def test_unknown_event_returns_404(client):
    assert client.get("/events/nonexistent:999").status_code == 404


def test_malformed_event_id_returns_400(client):
    assert client.get("/events/not-a-number").status_code == 400


# ---------------------------------------------------------------------------
# Detection and forecast
# ---------------------------------------------------------------------------


def test_anomalies_endpoint(client):
    body = client.get("/anomalies", params={"limit": 10}).json()
    assert body["detection_method"] in ("conformal", "sigma")
    assert all(r["is_anomaly"] for r in body["records"])


def test_forecast_returns_prediction_and_band(client):
    response = client.post(
        "/forecast",
        json={
            "timestamp": "2016-07-15T13:00:00",
            "irradiance_wm2": 850.0,
            "temperature_c": 30.0,
            "wind_speed_ms": 2.0,
            "site_id": FIXTURE_SITE,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_kw"] > 0
    assert body["expected_lower_kw"] <= body["predicted_kw"]
    assert body["data_mode"] == "synthetic"


def test_forecast_at_night_predicts_near_zero(client):
    body = client.post(
        "/forecast",
        json={
            "timestamp": "2016-07-15T02:00:00",
            "irradiance_wm2": 0.0,
            "temperature_c": 18.0,
            "site_id": FIXTURE_SITE,
        },
    ).json()
    assert body["predicted_kw"] < 5.0


def test_forecast_rejects_malformed_timestamp(client):
    response = client.post(
        "/forecast",
        json={"timestamp": "not-a-date", "irradiance_wm2": 500.0, "temperature_c": 20.0},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Metrics, methodology, provenance
# ---------------------------------------------------------------------------


def test_metrics_come_from_the_manifest(client):
    body = client.get("/metrics", params={"site_id": FIXTURE_SITE}).json()
    assert body["best_model"]
    assert body["models"]
    assert body["train_test_split_date"]
    for entry in body["models"]:
        assert entry["rmse_kw"] >= 0


def test_methodology_states_limits_not_just_capabilities(client):
    body = client.get("/methodology").json()
    assert body["conformal_guarantee"]
    assert len(body["conformal_limitations"]) >= 2
    assert body["feature_modes"]["weather_only"]["rationale"]
    assert body["spatial_method"]["limitation"]
    assert any(f["is_generation_loss"] is False for f in body["fault_taxonomy"])


def test_data_endpoint_separates_real_from_synthetic(client):
    body = client.get("/data").json()
    assert "not measured operational data" in body["synthetic_data_note"].lower()
    for record in body["synthetic_datasets"]:
        assert record["data_mode"] == "synthetic"
    for record in body["real_datasets"]:
        assert record["data_mode"] == "real"
        assert record["source_url"]


def test_demo_scenario(client):
    body = client.get("/demo/scenario").json()
    assert body["site_id"] == FIXTURE_SITE
    assert body["data_mode"] == "synthetic"
    assert "not measured operational data" in body["disclaimer"].lower()
    assert body["event"] is not None
    assert body["recommended_actions"]


def test_data_page_exposes_measured_provenance(built_store):
    """Regression: measured provenance lives in data/curated/, not the cache.

    The store originally searched only the working cache, which left the /data
    page reporting zero measured datasets while happily listing simulated ones —
    exactly inverting the credibility the page exists to establish.
    """
    from gridguard.api.store import ArtifactStore

    store = ArtifactStore(
        model_dir=built_store.model_dir,
        data_dir=built_store.data_dir,
    )
    assert store.curated_dir.name == "curated"


def test_real_sites_resolve_provenance_from_curated_directory():
    """The shipped curated datasets must each carry a loadable provenance record."""
    from gridguard.config import settings
    from gridguard.data.provenance import load_provenance
    from gridguard.sites.registry import list_sites

    curated = Path(settings.data_curated_dir)
    if not curated.exists():
        pytest.skip("curated datasets not present in this checkout")

    for site in list_sites(data_mode="real"):
        records = load_provenance(curated / f"telemetry_{site.site_id}.provenance.json")
        assert records, f"no curated provenance for {site.site_id}"
        assert records[0].data_mode == "real"
        assert records[0].source_url, f"{site.site_id} provenance has no source URL"
