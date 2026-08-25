"""
GridGuard API.

Serves prebuilt artifacts. The backend never trains, downloads, or calibrates
on request: everything comes from ``scripts/build_artifacts.py`` and is loaded
once at startup, so responses are fast, deterministic, and traceable to a
specific commit via the artifact manifest.

Routes
------
``/health``                        Liveness plus what artifacts loaded.
``/sites``                         Site registry with provenance disclaimers.
``/sites/{id}``                    One site.
``/sites/{id}/timeseries``         Actual, expected, and the calibrated band.
``/fleet/summary``                 Fleet KPIs, per-site health, map bounds.
``/events`` and ``/events/{id}``   Underperformance events and investigation.
``/anomalies``                     Interval-level detector output.
``/forecast``                      Expected generation for supplied weather.
``/metrics``                       Model evaluation from the manifest.
``/methodology``                   How detection works, and its limits.
``/data``                          Dataset provenance: real versus simulated.
``/demo/scenario``                 The guided walkthrough entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from gridguard import __version__
from gridguard.api import schemas
from gridguard.api.store import ArtifactStore, SiteBundle
from gridguard.config import settings
from gridguard.data.faults import FAULT_DESCRIPTIONS, GENERATION_LOSS_FAULTS
from gridguard.data.synthetic import DEMO_DATE, DEMO_SITE_ID
from gridguard.features.engineer import LAG_AWARE_FEATURES, WEATHER_ONLY_FEATURES
from gridguard.fleet.aggregate import summarise_fleet, summarise_site
from gridguard.sites.registry import REAL_DATA_NOTE, SYNTHETIC_DISCLAIMER
from gridguard.spatial.context import SpatialContext, compute_fleet_spatial_context
from gridguard.spatial.geo import bounding_box

logger = logging.getLogger(__name__)

store = ArtifactStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading GridGuard artifacts …")
    store.load()
    yield
    logger.info("GridGuard API shutting down.")


app = FastAPI(
    title="GridGuard API",
    version=__version__,
    description=(
        "Spatial intelligence and anomaly detection for distributed solar. "
        "Every response carries the data mode (`real` or `synthetic`) that "
        "produced it."
    ),
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    origins = settings.allowed_origin_list
    if "*" in origins:
        logger.warning(
            "CORS is configured to allow all origins. Set CORS_ALLOWED_ORIGINS to the "
            "specific frontend origins before deploying to production."
        )
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    # Vercel preview deployments get a fresh subdomain per branch, so they can
    # only be matched by pattern. Off unless explicitly enabled.
    allow_origin_regex=(
        r"^https://[a-z0-9-]+\.vercel\.app$" if settings.cors_allow_vercel_previews else None
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_bundle(site_id: str) -> SiteBundle:
    bundle = store.bundle(site_id)
    if bundle is None:
        available = ", ".join(store.site_ids) or "none — run `make build-artifacts`"
        raise HTTPException(
            status_code=404,
            detail=f"Unknown or unbuilt site '{site_id}'. Available: {available}",
        )
    return bundle


def _resolve_site_id(site_id: str | None, data_mode: str | None = None) -> str:
    resolved = site_id or store.default_site_id(data_mode)
    if resolved is None:
        raise HTTPException(
            status_code=503,
            detail="No artifacts are loaded. Run `make build-artifacts` and restart the API.",
        )
    return resolved


def _filter_window(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
    column: str = "timestamp",
) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    out = df
    if start:
        out = out[out[column] >= pd.Timestamp(start)]
    if end:
        out = out[out[column] <= pd.Timestamp(end)]
    return out


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _timeseries_points(df: pd.DataFrame, limit: int) -> tuple[list[schemas.TimeseriesPoint], bool]:
    truncated = len(df) > limit
    rows = df.tail(limit) if truncated else df
    points = [
        schemas.TimeseriesPoint(
            timestamp=str(row["timestamp"]),
            actual_kw=float(row["ac_power_kw"]),
            predicted_kw=float(row["predicted_kw"]),
            expected_lower_kw=_to_float(row.get("expected_lower_kw")),
            expected_upper_kw=_to_float(row.get("expected_upper_kw")),
            irradiance_wm2=_to_float(row.get("irradiance_wm2")),
            temperature_c=_to_float(row.get("temperature_c")),
            is_anomaly=bool(row.get("is_anomaly", False)),
        )
        for _, row in rows.iterrows()
    ]
    return points, truncated


def _event_record(row: pd.Series, bundle: SiteBundle | None = None) -> schemas.EventRecord:
    site = bundle.site if bundle else None
    return schemas.EventRecord(
        event_id=int(row["event_id"]),
        global_event_id=row.get("global_event_id"),
        site_id=row.get("site_id"),
        site_name=site.name if site else None,
        data_mode=site.data_mode if site else None,
        start_time=str(row["start_time"]),
        end_time=str(row["end_time"]),
        duration_minutes=float(row["duration_minutes"]),
        interval_count=int(row["interval_count"]),
        total_lost_kwh=float(row["total_lost_kwh"]),
        max_residual_sigma=_to_float(row.get("max_residual_sigma")),
        mean_actual_kw=float(row["mean_actual_kw"]),
        mean_predicted_kw=float(row["mean_predicted_kw"]),
        mean_expected_lower_kw=_to_float(row.get("mean_expected_lower_kw")),
        severity=row["severity"],
        explanation=str(row.get("explanation", "")),
        disclaimer=site.disclaimer if site else None,
    )


def _spatial_record(context: SpatialContext) -> schemas.SpatialContextRecord:
    payload = context.to_dict()
    return schemas.SpatialContextRecord(
        scope=payload["scope"],
        confidence=payload["confidence"],
        explanation=payload["explanation"],
        site_normalised_residual=payload["site_normalised_residual"],
        neighbor_count=payload["neighbor_count"],
        neighbors_considered=payload["neighbors_considered"],
        neighbor_residual_median=payload.get("neighbor_residual_median"),
        distance_weighted_residual=payload.get("distance_weighted_residual"),
        neighbors_affected=payload["neighbors_affected"],
        regional_anomaly_score=payload["regional_anomaly_score"],
        excess_deviation=payload.get("excess_deviation"),
        radius_km=payload["radius_km"],
    )


def _window_frames(start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
    """Each site's detector output over a shared window, for spatial context."""
    frames: dict[str, pd.DataFrame] = {}
    for site_id, bundle in store.bundles.items():
        if bundle.detected is None or bundle.detected.empty:
            continue
        frames[site_id] = _filter_window(bundle.detected, start, end)
    return frames


def _default_fleet_window(days: int = 1) -> tuple[str | None, str | None]:
    """The window the fleet view opens on when the caller does not specify one.

    Summarising the whole evaluation period is not useful: over several months
    every site accumulates some event, so every site reads "critical" and no
    site stands out against its neighbours. Health and spatial attribution are
    both statements about a *moment*, not a season.

    So the default anchors on the day of the fleet's most severe event — the
    thing an operator would want the fleet view to open on — falling back to the
    most recent day of data when there are no events. The resolved window is
    always returned in the response so the UI can state what it is showing.
    """
    events = store.all_events()
    anchor: pd.Timestamp | None = None

    if not events.empty:
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        ranked = events.assign(_rank=events["severity"].map(severity_rank).fillna(3)).sort_values(
            ["_rank", "total_lost_kwh"], ascending=[True, False]
        )
        anchor = pd.Timestamp(ranked.iloc[0]["start_time"])
    else:
        latest = [
            pd.Timestamp(b.detected["timestamp"].max())
            for b in store.bundles.values()
            if b.detected is not None and not b.detected.empty
        ]
        if latest:
            anchor = max(latest)

    if anchor is None:
        return None, None

    start = anchor.normalize()
    end = start + pd.Timedelta(days=days) - pd.Timedelta(minutes=1)
    return str(start), str(end)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
def health():
    """Liveness and artifact status. Reports no paths, secrets, or host detail."""
    detail = store.health()
    return schemas.HealthResponse(
        status="ok" if detail["sites_loaded"] else "degraded",
        version=__version__,
        detection_method=settings.anomaly_method,
        **detail,
    )


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


@app.get("/sites", response_model=schemas.SitesResponse, tags=["sites"])
def sites(data_mode: str | None = Query(default=None, pattern="^(real|synthetic)$")):
    """The site registry, each entry carrying its provenance disclaimer."""
    records = [
        schemas.SiteRecord(
            site_id=s.site_id,
            name=s.name,
            region=s.region,
            data_mode=s.data_mode,
            latitude=s.latitude,
            longitude=s.longitude,
            elevation_m=s.elevation_m,
            capacity_kw=s.capacity_kw,
            capacity_basis=s.capacity_basis,
            tilt_deg=s.tilt_deg,
            azimuth_deg=s.azimuth_deg,
            source_system_id=s.source_system_id,
            disclaimer=s.disclaimer,
            notes=s.notes,
        )
        for s in store.sites(data_mode)
    ]
    return schemas.SitesResponse(
        sites=records,
        total=len(records),
        real_count=sum(1 for r in records if r.data_mode == "real"),
        synthetic_count=sum(1 for r in records if r.data_mode == "synthetic"),
    )


@app.get("/sites/{site_id}", response_model=schemas.SiteRecord, tags=["sites"])
def site_detail(site_id: str):
    s = _require_bundle(site_id).site
    return schemas.SiteRecord(
        site_id=s.site_id,
        name=s.name,
        region=s.region,
        data_mode=s.data_mode,
        latitude=s.latitude,
        longitude=s.longitude,
        elevation_m=s.elevation_m,
        capacity_kw=s.capacity_kw,
        capacity_basis=s.capacity_basis,
        tilt_deg=s.tilt_deg,
        azimuth_deg=s.azimuth_deg,
        source_system_id=s.source_system_id,
        disclaimer=s.disclaimer,
        notes=s.notes,
    )


@app.get(
    "/sites/{site_id}/timeseries",
    response_model=schemas.TimeseriesResponse,
    tags=["sites"],
)
def site_timeseries(
    site_id: str,
    start: str | None = Query(default=None, description="ISO 8601 start"),
    end: str | None = Query(default=None, description="ISO 8601 end"),
    limit: int = Query(default=2000, ge=1, le=20000),
):
    """Actual versus expected generation with the calibrated expected range."""
    bundle = _require_bundle(site_id)
    if not bundle.has_data:
        raise HTTPException(status_code=404, detail=f"No detected telemetry for '{site_id}'.")

    frame = _filter_window(bundle.detected, start, end)
    points, truncated = _timeseries_points(frame, limit)
    return schemas.TimeseriesResponse(
        site_id=site_id,
        data_mode=bundle.site.data_mode,
        disclaimer=bundle.site.disclaimer,
        interval_minutes=bundle.provenance.interval_minutes if bundle.provenance else 15,
        points=points,
        total_points=len(frame),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------


@app.get("/fleet/summary", response_model=schemas.FleetSummaryResponse, tags=["fleet"])
def fleet_summary(
    data_mode: str | None = Query(default=None, pattern="^(real|synthetic)$"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    days: int = Query(default=1, ge=1, le=90, description="Window length when start/end omitted"),
):
    """Fleet KPIs, per-site health, spatial attribution, and map bounds.

    With no explicit window, opens on the day of the fleet's most severe event —
    see :func:`_default_fleet_window`. The resolved window is echoed back in
    ``window_start``/``window_end``.
    """
    selected = store.sites(data_mode)
    if not selected:
        raise HTTPException(
            status_code=503,
            detail="No artifacts are loaded. Run `make build-artifacts` and restart the API.",
        )

    if start is None and end is None:
        start, end = _default_fleet_window(days=days)

    frames = {
        s.site_id: _filter_window(store.bundle(s.site_id).detected, start, end)
        for s in selected
        if store.bundle(s.site_id) and store.bundle(s.site_id).has_data
    }

    contexts = compute_fleet_spatial_context(frames, settings.neighbor_radius_km, sites=selected)

    statuses = []
    for site in selected:
        bundle = store.bundle(site.site_id)
        events = bundle.events if bundle else None
        if events is not None and not events.empty:
            events = _filter_window(events, start, end, column="start_time")
        statuses.append(summarise_site(site, frames.get(site.site_id), events))

    summary = summarise_fleet(statuses, spatial_contexts=contexts)
    payload = summary.to_dict()
    payload["bounds"] = bounding_box(selected)
    payload["window_start"] = start
    payload["window_end"] = end or payload.get("window_end")
    return schemas.FleetSummaryResponse(**payload)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@app.get("/events", response_model=schemas.EventsResponse, tags=["events"])
def events(
    site_id: str | None = Query(default=None),
    data_mode: str | None = Query(default=None, pattern="^(real|synthetic)$"),
    severity: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Underperformance events across the fleet, most severe first."""
    frame = store.all_events()
    if frame.empty:
        return schemas.EventsResponse(total_events=0, total_lost_kwh=0.0, events=[])

    if site_id:
        frame = frame[frame["site_id"] == site_id]
    if data_mode:
        allowed = {s.site_id for s in store.sites(data_mode)}
        frame = frame[frame["site_id"].isin(allowed)]
    if severity:
        frame = frame[frame["severity"] == severity]
    frame = _filter_window(frame, start, end, column="start_time")

    total_lost = float(frame["total_lost_kwh"].sum()) if not frame.empty else 0.0
    records = [
        _event_record(row, store.bundle(row.get("site_id")))
        for _, row in frame.head(limit).iterrows()
    ]
    return schemas.EventsResponse(
        total_events=len(frame), total_lost_kwh=total_lost, events=records
    )


@app.get("/events/{event_id}", response_model=schemas.EventDetailResponse, tags=["events"])
def event_detail(event_id: str):
    """One event, with spatial attribution and neighbour comparison.

    ``event_id`` accepts either the global form ``site_id:number`` or a bare
    number, which resolves against the fleet-wide ordering.
    """
    frame = store.all_events()
    if frame.empty:
        raise HTTPException(status_code=404, detail="No events available.")

    if ":" in event_id:
        match = frame[frame["global_event_id"] == event_id]
    else:
        try:
            match = frame[frame["event_id"] == int(event_id)]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Malformed event id '{event_id}'. Use 'site_id:number' or a number.",
            ) from None

    if match.empty:
        raise HTTPException(status_code=404, detail=f"No event '{event_id}'.")

    row = match.iloc[0]
    site_id = row.get("site_id")
    bundle = store.bundle(site_id) if site_id else None
    record = _event_record(row, bundle)

    start, end = str(row["start_time"]), str(row["end_time"])
    # Pad the window so the chart shows the approach to and recovery from the event.
    pad = pd.Timedelta(hours=3)
    view_start = str(pd.Timestamp(start) - pad)
    view_end = str(pd.Timestamp(end) + pad)

    points: list[schemas.TimeseriesPoint] = []
    if bundle and bundle.has_data:
        window = _filter_window(bundle.detected, view_start, view_end)
        points, _ = _timeseries_points(window, 1000)

    contexts = compute_fleet_spatial_context(
        _window_frames(start, end), settings.neighbor_radius_km
    )
    context = contexts.get(site_id) if site_id else None

    neighbor_comparison = []
    if context:
        for neighbor_id in context.neighbors_considered:
            neighbor = store.bundle(neighbor_id)
            if neighbor is None or not neighbor.has_data:
                continue
            window = _filter_window(neighbor.detected, start, end)
            if window.empty:
                continue
            expected = float(window["predicted_kw"].sum())
            actual = float(window["ac_power_kw"].sum())
            neighbor_comparison.append(
                {
                    "site_id": neighbor_id,
                    "name": neighbor.site.name,
                    "capacity_kw": neighbor.site.capacity_kw,
                    "actual_kw_mean": round(float(window["ac_power_kw"].mean()), 3),
                    "expected_kw_mean": round(float(window["predicted_kw"].mean()), 3),
                    "expected_ratio": round(actual / expected, 4) if expected > 0 else None,
                    "anomaly_intervals": int(window["is_anomaly"].fillna(False).sum()),
                }
            )

    return schemas.EventDetailResponse(
        event=record,
        spatial_context=_spatial_record(context) if context else None,
        neighbor_comparison=neighbor_comparison,
        timeseries=points,
        recommended_actions=_recommended_actions(record, context),
        contributors=_explain_contributors(bundle, start),
    )


def _recommended_actions(
    event: schemas.EventRecord,
    context: SpatialContext | None,
) -> list[str]:
    """Operational next steps implied by the evidence.

    Deliberately phrased as investigation steps, not diagnoses. GridGuard
    detects and localises deviation; it does not identify the failed component,
    and recommending "replace the inverter" would claim knowledge it lacks.
    """
    actions: list[str] = []
    scope = context.scope.value if context else None

    if scope == "site_specific":
        actions.append(
            "Neighbouring sites under the same sky performed normally, so treat this as "
            "equipment-side until ruled out. Check inverter fault logs and string-level "
            "currents for the event window first."
        )
        actions.append(
            "If string currents are uniformly low, look at soiling or a curtailment "
            "setpoint before assuming hardware failure."
        )
    elif scope == "regional":
        actions.append(
            "Nearby sites dropped at the same time, which points at the weather input "
            "rather than the equipment. No site visit is warranted on this evidence alone."
        )
        actions.append(
            "If this recurs under similar conditions, the expected-generation model may "
            "need recalibration for that weather regime."
        )
    else:
        actions.append(
            "Spatial evidence is inconclusive. Confirm against on-site instrumentation "
            "before dispatching anyone."
        )

    if event.severity == "high":
        actions.append(
            f"Estimated {event.total_lost_kwh:.1f} kWh not generated over "
            f"{event.duration_minutes:.0f} minutes — prioritise accordingly."
        )
    if event.data_mode == "synthetic":
        actions.append(
            "This event is part of a simulated demonstration scenario. No real "
            "installation is affected."
        )
    return actions


def _explain_contributors(
    bundle: SiteBundle | None,
    timestamp: str,
) -> list[schemas.ExplainContributor]:
    """SHAP attribution for the expected-generation model at one interval."""
    if bundle is None or bundle.anomaly_model is None or not bundle.has_data:
        return []
    try:
        from gridguard.explainability.shap_explain import explain_interval

        return [
            schemas.ExplainContributor(**c)
            for c in explain_interval(bundle.detected, bundle.anomaly_model, timestamp)
        ]
    except Exception as exc:
        logger.debug("SHAP explanation unavailable: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Anomalies and forecast
# ---------------------------------------------------------------------------


@app.get("/anomalies", response_model=schemas.AnomalyResponse, tags=["detection"])
def anomalies(
    site_id: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    only_anomalies: bool = Query(default=True),
    limit: int = Query(default=200, ge=1, le=5000),
):
    """Interval-level detector output."""
    resolved = _resolve_site_id(site_id)
    bundle = _require_bundle(resolved)
    if not bundle.has_data:
        raise HTTPException(status_code=404, detail=f"No detected telemetry for '{resolved}'.")

    frame = _filter_window(bundle.detected, start, end)
    if only_anomalies:
        frame = frame[frame["is_anomaly"].fillna(False)]

    records = [
        schemas.AnomalyRecord(
            timestamp=str(row["timestamp"]),
            site_id=resolved,
            actual_kw=float(row["ac_power_kw"]),
            predicted_kw=float(row["predicted_kw"]),
            expected_lower_kw=_to_float(row.get("expected_lower_kw")),
            residual_kw=float(row["residual_kw"]),
            residual_sigma=_to_float(row.get("residual_sigma")),
            lost_energy_kwh=float(row.get("lost_energy_kwh", 0.0)),
            is_anomaly=bool(row.get("is_anomaly", False)),
        )
        for _, row in frame.head(limit).iterrows()
    ]
    return schemas.AnomalyResponse(
        total_anomalies=int(frame["is_anomaly"].fillna(False).sum()),
        total_lost_kwh=float(frame.get("lost_energy_kwh", pd.Series(dtype=float)).sum()),
        detection_method=settings.anomaly_method,
        records=records,
    )


@app.post("/forecast", response_model=schemas.ForecastResponse, tags=["detection"])
def forecast(request: schemas.ForecastRequest):
    """Expected generation for supplied weather, with the calibrated range."""
    resolved = _resolve_site_id(request.site_id)
    bundle = _require_bundle(resolved)
    if bundle.anomaly_model is None:
        raise HTTPException(
            status_code=503, detail=f"No expected-generation model loaded for '{resolved}'."
        )

    from gridguard.features.engineer import build_features

    try:
        stamp = pd.Timestamp(request.timestamp)
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"Could not parse timestamp '{request.timestamp}'."
        ) from None

    frame = pd.DataFrame(
        {
            "timestamp": [stamp],
            "irradiance_wm2": [request.irradiance_wm2],
            "temperature_c": [request.temperature_c],
            "wind_speed_ms": [request.wind_speed_ms],
            "ac_power_kw": [np.nan],
        }
    )
    features = build_features(frame, include_lags=False)
    columns = [c for c in WEATHER_ONLY_FEATURES if c in features.columns]
    predicted = float(np.clip(bundle.anomaly_model.predict(features[columns])[0], 0, None))

    lower = upper = None
    if bundle.calibration:
        lower = max(0.0, predicted + bundle.calibration.lower_bound_for(request.irradiance_wm2))
        upper = max(0.0, predicted + bundle.calibration.upper_bound_for(request.irradiance_wm2))

    return schemas.ForecastResponse(
        timestamp=str(stamp),
        predicted_kw=round(predicted, 4),
        expected_lower_kw=round(lower, 4) if lower is not None else None,
        expected_upper_kw=round(upper, 4) if upper is not None else None,
        model_name="xgboost_weather_only",
        site_id=resolved,
        data_mode=bundle.site.data_mode,
    )


# ---------------------------------------------------------------------------
# Metrics, methodology, provenance
# ---------------------------------------------------------------------------


@app.get("/metrics", response_model=schemas.MetricsResponse, tags=["meta"])
def metrics(site_id: str | None = Query(default=None)):
    """Model evaluation, read from the artifact manifest.

    These are the numbers produced by the build that created the loaded
    artifacts — never hand-entered, and never recomputed on request.
    """
    resolved = _resolve_site_id(site_id)
    bundle = _require_bundle(resolved)

    if store.manifest is None:
        raise HTTPException(
            status_code=503, detail="No artifact manifest loaded; cannot report metrics."
        )
    entry = store.manifest.site(resolved)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No manifest entry for '{resolved}'.")

    weather_only = []
    for name in ("physics", "physics_hybrid", "weather_only_xgboost"):
        detail = entry.physics_metrics.get(name)
        if isinstance(detail, dict) and "mae_kw" in detail:
            weather_only.append(schemas.ModelMetrics(model=name, **detail))

    return schemas.MetricsResponse(
        site_id=resolved,
        data_mode=bundle.site.data_mode,
        best_model=entry.best_forecast_model,
        models=[schemas.ModelMetrics(**m) for m in entry.forecast_metrics],
        weather_only_models=weather_only,
        evaluation_note=str(entry.physics_metrics.get("evaluation_note", "")),
        detection_metrics=entry.detection_metrics or None,
        conformal_coverage=entry.conformal_coverage or None,
        train_test_split_date=entry.train_test_split_date,
        data_start=entry.data_start,
        data_end=entry.data_end,
    )


@app.get("/methodology", response_model=schemas.MethodologyResponse, tags=["meta"])
def methodology():
    """How detection works, stated with its limits."""
    physics = None
    manifest_summary = None
    if store.manifest:
        manifest_summary = {
            "artifact_version": store.manifest.artifact_version,
            "built_at": store.manifest.built_at,
            "git_commit": store.manifest.git_commit,
            "python_version": store.manifest.python_version,
            "package_versions": store.manifest.package_versions,
            "sites_built": [s.site_id for s in store.manifest.sites],
        }
        for entry in store.manifest.sites:
            assumptions = entry.physics_metrics.get("assumptions")
            if assumptions:
                physics = assumptions
                break

    sample = next(
        (b.calibration for b in store.bundles.values() if b.calibration is not None), None
    )

    return schemas.MethodologyResponse(
        detection_method=settings.anomaly_method,
        conformal_alpha=settings.conformal_alpha,
        conformal_guarantee=(
            sample.to_dict()["guarantee"]
            if sample
            else (
                f"Under exchangeability, at least "
                f"{(1 - settings.conformal_alpha) * 100:.0f}% of healthy intervals fall at or "
                "above the calibrated lower bound."
            )
        ),
        conformal_limitations=[
            "Coverage is marginal within each irradiance bucket, not conditional on every "
            "individual interval.",
            "Exchangeability is an assumption about the world. Panel degradation, sensor "
            "drift and seasonal shift all break it, so calibration is recomputed per build "
            "and its window is recorded in the manifest.",
            "A breach means the interval fell outside the calibrated range of healthy "
            "behaviour. It is not by itself a diagnosis of a fault.",
        ],
        feature_modes={
            "weather_only": {
                "purpose": "Expected generation for anomaly detection.",
                "features": WEATHER_ONLY_FEATURES,
                "rationale": (
                    "Excludes lagged power. A system degraded for days produces low output, "
                    "so its lagged power is low, so a lag-aware model would predict low output "
                    "and call the degradation normal."
                ),
            },
            "lag_aware": {
                "purpose": "Short-horizon operational forecasting.",
                "features": LAG_AWARE_FEATURES,
                "rationale": (
                    "Recent observed output is a strong predictor of the next interval. Used "
                    "for forecasting only, never for anomaly baselines."
                ),
            },
        },
        fault_taxonomy=[
            {
                "fault_type": fault.value,
                "description": description,
                "is_generation_loss": fault in GENERATION_LOSS_FAULTS,
            }
            for fault, description in FAULT_DESCRIPTIONS.items()
        ],
        spatial_method={
            "distance": "Haversine great-circle distance on a spherical Earth.",
            "neighbor_radius_km": settings.neighbor_radius_km,
            "normalisation": "Residuals divided by nameplate capacity, making sites comparable.",
            "weighting": "Inverse-distance weighted neighbour residual, plus an unweighted median.",
            "constraint": (
                "Neighbours are restricted to the same data mode. A simulated site's residuals "
                "carry no information about a measured site's weather."
            ),
            "scope_values": ["site_specific", "regional", "indeterminate"],
            "limitation": (
                "This is not a fault classifier. It answers only whether a deviation is shared "
                "with nearby sites, and reports the evidence behind that answer."
            ),
        },
        physics_model=physics,
        artifact=manifest_summary,
    )


@app.get("/data", response_model=schemas.DataProvenanceResponse, tags=["meta"])
def data_provenance():
    """Dataset provenance: what is measured, what is simulated, and from where."""
    records = [schemas.ProvenanceRecord(**r.to_dict()) for r in store.provenance_records()]
    return schemas.DataProvenanceResponse(
        real_datasets=[r for r in records if r.data_mode == "real"],
        synthetic_datasets=[r for r in records if r.data_mode == "synthetic"],
        real_data_note=REAL_DATA_NOTE,
        synthetic_data_note=SYNTHETIC_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


@app.get("/demo/scenario", response_model=schemas.DemoScenarioResponse, tags=["demo"])
def demo_scenario():
    """Entry point for the guided walkthrough.

    Returns the scripted demonstration event together with the spatial evidence
    that it is site-specific, and points at a measured site so the walkthrough
    can end on real data.
    """
    bundle = store.bundle(DEMO_SITE_ID)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Demo site '{DEMO_SITE_ID}' is not built. Run `make build-artifacts`.",
        )

    event_record = None
    context = None
    anomaly_records: list[schemas.AnomalyRecord] = []

    if bundle.events is not None and not bundle.events.empty:
        day = bundle.events[bundle.events["start_time"].astype(str).str.startswith(DEMO_DATE)]
        chosen = day if not day.empty else bundle.events
        row = chosen.sort_values("total_lost_kwh", ascending=False).iloc[0]
        if "global_event_id" not in row:
            row = row.copy()
            row["global_event_id"] = f"{DEMO_SITE_ID}:{int(row['event_id'])}"
        event_record = _event_record(row, bundle)

        contexts = compute_fleet_spatial_context(
            _window_frames(str(row["start_time"]), str(row["end_time"])),
            settings.neighbor_radius_km,
        )
        context = contexts.get(DEMO_SITE_ID)

    if bundle.has_data:
        window = bundle.detected[bundle.detected["timestamp"].astype(str).str.startswith(DEMO_DATE)]
        anomaly_records = [
            schemas.AnomalyRecord(
                timestamp=str(r["timestamp"]),
                site_id=DEMO_SITE_ID,
                actual_kw=float(r["ac_power_kw"]),
                predicted_kw=float(r["predicted_kw"]),
                expected_lower_kw=_to_float(r.get("expected_lower_kw")),
                residual_kw=float(r["residual_kw"]),
                residual_sigma=_to_float(r.get("residual_sigma")),
                lost_energy_kwh=float(r.get("lost_energy_kwh", 0.0)),
                is_anomaly=bool(r.get("is_anomaly", False)),
            )
            for _, r in window.iterrows()
        ]

    real_site = store.default_site_id("real")
    return schemas.DemoScenarioResponse(
        data_label="Simulated demonstration scenario",
        data_mode="synthetic",
        disclaimer=bundle.site.disclaimer,
        demo_date=DEMO_DATE,
        site_id=DEMO_SITE_ID,
        site_name=bundle.site.name,
        event=event_record,
        spatial_context=_spatial_record(context) if context else None,
        anomaly_records=anomaly_records,
        jump_to_event_id=event_record.global_event_id if event_record else None,
        recommended_actions=(_recommended_actions(event_record, context) if event_record else []),
        real_data_site_id=real_site,
        real_data_note=(
            "The same forecasting and detection pipeline is evaluated against measured "
            "photovoltaic telemetry from NREL PVDAQ. See the Data page for provenance."
        ),
    )
