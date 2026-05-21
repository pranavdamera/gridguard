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
