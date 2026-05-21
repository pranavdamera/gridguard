"""Pydantic request/response schemas for the GridGuard API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Forecast endpoint
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    """Input features for a single 15-minute interval forecast."""

    timestamp: datetime
    irradiance_wm2: float = Field(..., ge=0, le=1500, description="Global horizontal irradiance W/m²")
    temperature_c: float = Field(..., ge=-20, le=60, description="Ambient temperature °C")
    wind_speed_ms: float = Field(default=0.0, ge=0, le=50)
    ac_power_lag1: float = Field(default=0.0, ge=0, description="Previous interval AC power kW")
    ac_power_lag4: float = Field(default=0.0, ge=0, description="Power 1 hour ago kW")


class ForecastResponse(BaseModel):
    timestamp: datetime
    predicted_kw: float
    model_name: str


# ---------------------------------------------------------------------------
# Anomaly endpoint
# ---------------------------------------------------------------------------


class AnomalyRecord(BaseModel):
    timestamp: datetime
    actual_kw: float
    predicted_kw: float
    residual_kw: float
    residual_sigma: float
    lost_energy_kwh: float
    is_anomaly: bool


class AnomalyResponse(BaseModel):
    total_anomalies: int
    total_lost_kwh: float
    records: list[AnomalyRecord]


# ---------------------------------------------------------------------------
# Events endpoint
# ---------------------------------------------------------------------------


class EventRecord(BaseModel):
    event_id: int
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    interval_count: int
    total_lost_kwh: float
    max_residual_sigma: float
    mean_actual_kw: float
    mean_predicted_kw: float
    severity: Literal["low", "medium", "high"]
    explanation: str
    site_id: str | None = None


class EventsResponse(BaseModel):
    total_events: int
    total_lost_kwh: float
    events: list[EventRecord]


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------


class ExplainContributor(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: Literal["positive", "negative"]


class ExplainResponse(BaseModel):
    timestamp: datetime
    predicted_kw: float
    contributors: list[ExplainContributor]
    model_name: str
    shap_available: bool


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class ModelMetrics(BaseModel):
    model: str
    mae_kw: float
    rmse_kw: float
    mape_pct: float
    r2: float


class MetricsResponse(BaseModel):
    models: list[ModelMetrics]
    best_model: str


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_rows: int
    version: str
    residual_calibration_loaded: bool
