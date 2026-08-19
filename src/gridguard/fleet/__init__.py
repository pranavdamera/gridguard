"""Fleet-level aggregation: per-site health rollups and fleet KPIs."""

from gridguard.fleet.aggregate import (
    FleetSummary,
    HealthState,
    SiteStatus,
    classify_health,
    summarise_fleet,
    summarise_site,
)

__all__ = [
    "FleetSummary",
    "HealthState",
    "SiteStatus",
    "classify_health",
    "summarise_fleet",
    "summarise_site",
]
