"""Tests for the FastAPI backend.

These are integration tests that spin up the app with TestClient.
Models don't need to be pre-loaded — the API handles graceful degradation.
"""

from fastapi.testclient import TestClient

from gridguard.api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data


def test_forecast_endpoint_valid_input():
    payload = {
        "timestamp": "2023-06-15T12:00:00",
        "irradiance_wm2": 800.0,
        "temperature_c": 28.0,
        "wind_speed_ms": 3.5,
        "ac_power_lag1": 7.2,
        "ac_power_lag4": 6.8,
    }
    resp = client.post("/forecast", json=payload)
    # Will return 503 if models not loaded (CI without artifacts) — that's acceptable
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "predicted_kw" in data
        assert data["predicted_kw"] >= 0


def test_forecast_rejects_negative_irradiance():
    payload = {
        "timestamp": "2023-06-15T12:00:00",
        "irradiance_wm2": -100.0,  # invalid
        "temperature_c": 25.0,
    }
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 422


def test_anomalies_endpoint():
    resp = client.get("/anomalies?limit=10")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "total_anomalies" in data
        assert "total_lost_kwh" in data
        assert isinstance(data["records"], list)


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "models" in data
        assert "best_model" in data
        assert len(data["models"]) > 0


def test_anomalies_limit_respected():
    resp = client.get("/anomalies?limit=5&only_anomalies=false")
    if resp.status_code == 200:
        data = resp.json()
        assert len(data["records"]) <= 5


def test_events_endpoint():
    resp = client.get("/events?limit=5")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "total_events" in data
        assert "events" in data
        for ev in data["events"]:
            assert "explanation" in ev
            assert isinstance(ev["explanation"], str)


def test_docs_available():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_sites_endpoint():
    resp = client.get("/sites")
    assert resp.status_code == 200
    data = resp.json()
    assert "sites" in data
    assert data["total"] > 0
    for s in data["sites"]:
        assert "site_id" in s
        assert "latitude" in s
        assert "longitude" in s
        assert "capacity_kw" in s


def test_site_detail_endpoint():
    resp = client.get("/sites/gmu_fairfax")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "gmu_fairfax"
    assert "capacity_kw" in data


def test_site_detail_not_found():
    resp = client.get("/sites/nonexistent_site_xyz")
    assert resp.status_code == 404


def test_events_severity_filter():
    for sev in ("low", "medium", "high"):
        resp = client.get(f"/events?severity={sev}")
        assert resp.status_code in (200, 503)
    resp = client.get("/events?severity=invalid")
    # 400 if events are loaded, 503 if artifacts not present (CI without artifacts)
    assert resp.status_code in (400, 503)


def test_events_by_id():
    # events list to find a valid id
    list_resp = client.get("/events?limit=1")
    if list_resp.status_code == 200 and list_resp.json()["events"]:
        eid = list_resp.json()["events"][0]["event_id"]
        resp = client.get(f"/events/{eid}")
        assert resp.status_code == 200
        assert resp.json()["event_id"] == eid


def test_events_by_id_not_found():
    resp = client.get("/events/999999")
    assert resp.status_code in (404, 503)


def test_demo_scenario_endpoint():
    resp = client.get("/demo/scenario")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "data_label" in data
        assert "demo_date" in data
        assert "recommended_actions" in data
        assert isinstance(data["recommended_actions"], list)


def test_forecast_get_endpoint():
    resp = client.get("/forecast?limit=5")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, list)
        for item in data:
            assert "predicted_kw" in item
            assert item["predicted_kw"] >= 0
