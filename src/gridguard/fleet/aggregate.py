"""
Fleet-level aggregation.

Turns per-site detector output into the two views an operator actually works
from: a ranked list of sites needing attention, and a handful of fleet totals
that answer "is anything wrong right now, and how much is it costing?"

Health states
-------------
``healthy``   Generation inside the calibrated expected range.
``warning``   Sustained shortfall, or a low/medium-severity active event.
``critical``  A high-severity active event, or a large fraction of nameplate
              lost over the window.
``no_data``   No usable telemetry for the window. Deliberately distinct from
              healthy: a site that has stopped reporting is not a site that is
              fine, and collapsing the two is how outages go unnoticed.

Performance ratio
-----------------
Reported here as **actual generation divided by model-expected generation** over
the window. Note this is *not* the IEC 61724 performance ratio, which
normalises by irradiance and nameplate. This one answers "is the array doing
what we predicted?" rather than "how efficient is the array?", and it is
labelled ``expected_ratio`` in the API to avoid implying the standard metric.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from gridguard.anomaly.detect import INTERVAL_HOURS
from gridguard.sites.registry import Site

logger = logging.getLogger(__name__)


class HealthState(StrEnum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    NO_DATA = "no_data"


#: Fraction of expected generation lost over the window that escalates a site.
WARNING_LOSS_FRACTION = 0.08
CRITICAL_LOSS_FRACTION = 0.25

#: Fraction of assessable intervals flagged that escalates a site.
WARNING_ANOMALY_FRACTION = 0.05
CRITICAL_ANOMALY_FRACTION = 0.20


@dataclass
class SiteStatus:
    """One site's condition over a window."""

    site_id: str
    name: str
    data_mode: str
    latitude: float
    longitude: float
    capacity_kw: float

    health: HealthState = HealthState.NO_DATA
    actual_kwh: float = 0.0
    expected_kwh: float = 0.0
    expected_ratio: float | None = None
    lost_energy_kwh: float = 0.0
    loss_fraction: float | None = None

    anomaly_intervals: int = 0
    assessable_intervals: int = 0
    active_events: int = 0
    max_severity: str = "none"

    latest_timestamp: str | None = None
    latest_actual_kw: float | None = None
    latest_expected_kw: float | None = None
    latest_expected_lower_kw: float | None = None

    anomaly_scope: str | None = None
    scope_confidence: str | None = None
    scope_explanation: str | None = None

    disclaimer: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["health"] = self.health.value
        for key, value in payload.items():
            if isinstance(value, float) and np.isfinite(value):
                payload[key] = round(value, 4)
        return payload


@dataclass
class FleetSummary:
    """Fleet-wide rollup."""

    total_sites: int = 0
    total_capacity_kw: float = 0.0
    sites_healthy: int = 0
    sites_warning: int = 0
    sites_critical: int = 0
    sites_no_data: int = 0

    total_actual_kwh: float = 0.0
    total_expected_kwh: float = 0.0
    total_lost_kwh: float = 0.0
    fleet_expected_ratio: float | None = None

    active_events: int = 0
    real_sites: int = 0
    synthetic_sites: int = 0

    window_start: str | None = None
    window_end: str | None = None
    sites: list[SiteStatus] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["sites"] = [s.to_dict() for s in self.sites]
        for key, value in payload.items():
            if isinstance(value, float) and np.isfinite(value):
                payload[key] = round(value, 4)
        return payload


def classify_health(
    loss_fraction: float | None,
    anomaly_fraction: float,
    max_severity: str,
    has_data: bool,
) -> HealthState:
    """Map window statistics onto a health state.

    Severity from grouped events takes precedence over the aggregate fractions:
    a short, deep outage matters more than its dilution across a long window
    would suggest.
    """
    if not has_data:
        return HealthState.NO_DATA

    if max_severity == "high":
        return HealthState.CRITICAL
    if loss_fraction is not None and loss_fraction >= CRITICAL_LOSS_FRACTION:
        return HealthState.CRITICAL
    if anomaly_fraction >= CRITICAL_ANOMALY_FRACTION:
        return HealthState.CRITICAL

    if max_severity in ("medium", "low"):
        return HealthState.WARNING
    if loss_fraction is not None and loss_fraction >= WARNING_LOSS_FRACTION:
        return HealthState.WARNING
    if anomaly_fraction >= WARNING_ANOMALY_FRACTION:
        return HealthState.WARNING

    return HealthState.HEALTHY


def summarise_site(
    site: Site,
    detected: pd.DataFrame | None,
    events: pd.DataFrame | None = None,
) -> SiteStatus:
    """Roll one site's detector output up into a status card.

    Args:
        site:     Registry entry.
        detected: Detector output for the window, or None/empty when the site
                  has no usable telemetry.
        events:   Grouped events for the window, used for severity.
    """
    status = SiteStatus(
        site_id=site.site_id,
        name=site.name,
        data_mode=site.data_mode,
        latitude=site.latitude,
        longitude=site.longitude,
        capacity_kw=site.capacity_kw,
        disclaimer=site.disclaimer,
    )

    if detected is None or detected.empty:
        status.health = HealthState.NO_DATA
        return status

    frame = detected
    status.actual_kwh = float(frame["ac_power_kw"].sum() * INTERVAL_HOURS)
    status.expected_kwh = float(frame["predicted_kw"].sum() * INTERVAL_HOURS)
    status.lost_energy_kwh = float(frame.get("lost_energy_kwh", pd.Series(dtype=float)).sum())

    if status.expected_kwh > 0:
        status.expected_ratio = status.actual_kwh / status.expected_kwh
        status.loss_fraction = status.lost_energy_kwh / status.expected_kwh

    assessable = frame["irradiance_wm2"] > 50
    status.assessable_intervals = int(assessable.sum())
    status.anomaly_intervals = int(frame.loc[assessable, "is_anomaly"].fillna(False).sum())

    if events is not None and not events.empty:
        site_events = events
        if "site_id" in events.columns:
            site_events = events[events["site_id"] == site.site_id]
        status.active_events = int(len(site_events))
        if not site_events.empty and "severity" in site_events.columns:
            for level in ("high", "medium", "low"):
                if (site_events["severity"] == level).any():
                    status.max_severity = level
                    break

    latest = frame.sort_values("timestamp").iloc[-1]
    status.latest_timestamp = str(latest["timestamp"])
    status.latest_actual_kw = float(latest["ac_power_kw"])
    status.latest_expected_kw = float(latest["predicted_kw"])
    if "expected_lower_kw" in frame.columns:
        status.latest_expected_lower_kw = float(latest["expected_lower_kw"])

    anomaly_fraction = status.anomaly_intervals / max(status.assessable_intervals, 1)
    status.health = classify_health(
        loss_fraction=status.loss_fraction,
        anomaly_fraction=anomaly_fraction,
        max_severity=status.max_severity,
        has_data=status.assessable_intervals > 0,
    )
    return status


def summarise_fleet(
    statuses: list[SiteStatus],
    spatial_contexts: dict | None = None,
) -> FleetSummary:
    """Aggregate site statuses into fleet KPIs.

    Args:
        statuses:         Per-site statuses.
        spatial_contexts: Optional ``{site_id: SpatialContext}`` to attach
                          regional-versus-local attribution to each site.
    """
    summary = FleetSummary(total_sites=len(statuses))

    for status in statuses:
        if spatial_contexts and status.site_id in spatial_contexts:
            context = spatial_contexts[status.site_id]
            status.anomaly_scope = context.scope.value
            status.scope_confidence = context.confidence
            status.scope_explanation = context.explanation

        summary.total_capacity_kw += status.capacity_kw
        summary.total_actual_kwh += status.actual_kwh
        summary.total_expected_kwh += status.expected_kwh
        summary.total_lost_kwh += status.lost_energy_kwh
        summary.active_events += status.active_events

        if status.data_mode == "real":
            summary.real_sites += 1
        else:
            summary.synthetic_sites += 1

        if status.health is HealthState.HEALTHY:
            summary.sites_healthy += 1
        elif status.health is HealthState.WARNING:
            summary.sites_warning += 1
        elif status.health is HealthState.CRITICAL:
            summary.sites_critical += 1
        else:
            summary.sites_no_data += 1

    if summary.total_expected_kwh > 0:
        summary.fleet_expected_ratio = summary.total_actual_kwh / summary.total_expected_kwh

    # Worst first, so the list doubles as a work queue.
    order = {
        HealthState.CRITICAL: 0,
        HealthState.WARNING: 1,
        HealthState.NO_DATA: 2,
        HealthState.HEALTHY: 3,
    }
    summary.sites = sorted(
        statuses,
        key=lambda s: (order[s.health], -(s.lost_energy_kwh or 0.0)),
    )

    timestamps = [s.latest_timestamp for s in statuses if s.latest_timestamp]
    if timestamps:
        summary.window_end = max(timestamps)

    return summary
