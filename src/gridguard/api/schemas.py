"""
Pydantic response schemas.

Every response that carries generation numbers also carries the ``data_mode``
that produced them. That is a deliberate API-design decision, not decoration:
it makes it impossible for a client to render a chart without knowing whether
it is showing measured telemetry or a simulation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DataMode = Literal["real", "synthetic"]
Severity = Literal["low", "medium", "high"]
Health = Literal["healthy", "warning", "critical", "no_data"]
Scope = Literal["site_specific", "regional", "indeterminate"]


# ---------------------------------------------------------------------------
# Health and metadata
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    sites_loaded: int
    sites_with_models: int
    sites_with_calibration: int
    data_rows: int
    manifest_present: bool
    artifact_built_at: str | None = None
    artifact_commit: str | None = None
    detection_method: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


class SiteRecord(BaseModel):
    site_id: str
    name: str
    region: str
    data_mode: DataMode
    latitude: float
    longitude: float
    elevation_m: float
    capacity_kw: float
    capacity_basis: str
    tilt_deg: float | None = None
    azimuth_deg: float | None = None
    source_system_id: int | None = None
    disclaimer: str
    notes: str


class SitesResponse(BaseModel):
    sites: list[SiteRecord]
    total: int
    real_count: int
    synthetic_count: int


class TimeseriesPoint(BaseModel):
    timestamp: str
    actual_kw: float
    predicted_kw: float
    expected_lower_kw: float | None = None
    expected_upper_kw: float | None = None
    irradiance_wm2: float | None = None
    temperature_c: float | None = None
    is_anomaly: bool = False


class TimeseriesResponse(BaseModel):
    site_id: str
    data_mode: DataMode
    disclaimer: str
    interval_minutes: int
    points: list[TimeseriesPoint]
    total_points: int
    truncated: bool = False


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------


class SiteStatusRecord(BaseModel):
    site_id: str
    name: str
    data_mode: DataMode
    latitude: float
    longitude: float
    capacity_kw: float
    health: Health
    actual_kwh: float
    expected_kwh: float
    expected_ratio: float | None = None
    lost_energy_kwh: float
    loss_fraction: float | None = None
    anomaly_intervals: int
    assessable_intervals: int
    active_events: int
    max_severity: str
    latest_timestamp: str | None = None
    latest_actual_kw: float | None = None
    latest_expected_kw: float | None = None
    latest_expected_lower_kw: float | None = None
    anomaly_scope: Scope | None = None
    scope_confidence: str | None = None
    scope_explanation: str | None = None
    disclaimer: str


class FleetBounds(BaseModel):
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float
    center_latitude: float
    center_longitude: float


class FleetSummaryResponse(BaseModel):
    total_sites: int
    total_capacity_kw: float
    sites_healthy: int
    sites_warning: int
    sites_critical: int
    sites_no_data: int
    total_actual_kwh: float
    total_expected_kwh: float
    total_lost_kwh: float
    fleet_expected_ratio: float | None = None
    active_events: int
    real_sites: int
    synthetic_sites: int
    window_start: str | None = None
    window_end: str | None = None
    sites: list[SiteStatusRecord]
    bounds: FleetBounds | None = None


# ---------------------------------------------------------------------------
# Events and anomalies
# ---------------------------------------------------------------------------


class SpatialContextRecord(BaseModel):
    scope: Scope
    confidence: str
    explanation: str
    site_normalised_residual: float
    neighbor_count: int
    neighbors_considered: list[str] = Field(default_factory=list)
    neighbor_residual_median: float | None = None
    distance_weighted_residual: float | None = None
    neighbors_affected: int = 0
    regional_anomaly_score: float = 0.0
    excess_deviation: float | None = None
    radius_km: float = 0.0


class EventRecord(BaseModel):
    event_id: int
    global_event_id: str | None = None
    site_id: str | None = None
    site_name: str | None = None
    data_mode: DataMode | None = None
    start_time: str
    end_time: str
    duration_minutes: float
    interval_count: int
    total_lost_kwh: float
    max_residual_sigma: float | None = None
    mean_actual_kw: float
    mean_predicted_kw: float
    mean_expected_lower_kw: float | None = None
    severity: Severity
    explanation: str
    disclaimer: str | None = None


class EventsResponse(BaseModel):
    total_events: int
    total_lost_kwh: float
    events: list[EventRecord]


class EventDetailResponse(BaseModel):
    event: EventRecord
    spatial_context: SpatialContextRecord | None = None
    neighbor_comparison: list[dict] = Field(default_factory=list)
    timeseries: list[TimeseriesPoint] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    contributors: list[ExplainContributor] = Field(default_factory=list)


class AnomalyRecord(BaseModel):
    timestamp: str
    site_id: str | None = None
    actual_kw: float
    predicted_kw: float
    expected_lower_kw: float | None = None
    residual_kw: float
    residual_sigma: float | None = None
    lost_energy_kwh: float
    is_anomaly: bool


class AnomalyResponse(BaseModel):
    total_anomalies: int
    total_lost_kwh: float
    detection_method: str
    records: list[AnomalyRecord]


# ---------------------------------------------------------------------------
# Forecast, metrics, explanation
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    timestamp: str
    irradiance_wm2: float
    temperature_c: float
    wind_speed_ms: float = 2.0
    site_id: str | None = None


class ForecastResponse(BaseModel):
    timestamp: str
    predicted_kw: float
    expected_lower_kw: float | None = None
    expected_upper_kw: float | None = None
    model_name: str
    site_id: str | None = None
    data_mode: DataMode | None = None


class ModelMetrics(BaseModel):
    model: str
    mae_kw: float
    rmse_kw: float
    mape_pct: float | None = None
    r2: float
    n_samples: int | None = None


class MetricsResponse(BaseModel):
    site_id: str
    data_mode: DataMode
    best_model: str
    models: list[ModelMetrics]
    weather_only_models: list[ModelMetrics] = Field(default_factory=list)
    evaluation_note: str = ""
    detection_metrics: dict | None = None
    conformal_coverage: dict | None = None
    train_test_split_date: str | None = None
    data_start: str | None = None
    data_end: str | None = None


class ExplainContributor(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: Literal["positive", "negative"]


class ExplainResponse(BaseModel):
    timestamp: str
    site_id: str
    predicted_kw: float
    actual_kw: float | None = None
    expected_lower_kw: float | None = None
    contributors: list[ExplainContributor]
    model_name: str
    shap_available: bool
    note: str = ""


# ---------------------------------------------------------------------------
# Provenance and methodology
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    site_id: str
    data_mode: DataMode
    dataset: str
    system_identifier: str = ""
    site_name: str = ""
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None
    capacity_kw: float | None = None
    capacity_basis: str = ""
    tilt_deg: float | None = None
    azimuth_deg: float | None = None
    interval_minutes: int | None = None
    native_interval_minutes: int | None = None
    start: str = ""
    end: str = ""
    timezone_note: str = ""
    row_count: int | None = None
    weather_source: str = ""
    irradiance_kind: str = ""
    irradiance_channel: str = ""
    irradiance_scale_factor: float | None = None
    irradiance_scale_basis: str = ""
    source_url: str = ""
    license_note: str = ""
    retrieved_at: str = ""
    processing_notes: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class DataProvenanceResponse(BaseModel):
    real_datasets: list[ProvenanceRecord]
    synthetic_datasets: list[ProvenanceRecord]
    real_data_note: str
    synthetic_data_note: str


class MethodologyResponse(BaseModel):
    detection_method: str
    conformal_alpha: float
    conformal_guarantee: str
    conformal_limitations: list[str]
    feature_modes: dict
    fault_taxonomy: list[dict]
    spatial_method: dict
    physics_model: dict | None = None
    artifact: dict | None = None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


class DemoScenarioResponse(BaseModel):
    data_label: str
    data_mode: DataMode
    disclaimer: str
    demo_date: str
    site_id: str
    site_name: str
    event: EventRecord | None = None
    spatial_context: SpatialContextRecord | None = None
    anomaly_records: list[AnomalyRecord] = Field(default_factory=list)
    jump_to_event_id: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)
    real_data_site_id: str | None = None
    real_data_note: str = ""


EventDetailResponse.model_rebuild()
