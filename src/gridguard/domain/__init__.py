"""
The GridGuard domain model.

Types only: what a site, an asset, a measurement and a fault hypothesis *are*.
No I/O, no models, no network. Loading assets from the registry lives in
:mod:`gridguard.sites.registry`; injecting faults lives in
:mod:`gridguard.data.faults`. Keeping the vocabulary separate from its consumers
is what lets the edge agent, the collector and the API all speak it without
depending on each other.
"""

from gridguard.domain.asset import (
    GENERATING_KINDS,
    SENSING_KINDS,
    Asset,
    AssetKind,
    AssetTree,
    build_asset_id,
)
from gridguard.domain.fault import (
    FAULT_CATEGORIES,
    Evidence,
    FaultCategory,
    FaultHypothesis,
    category_for,
)
from gridguard.domain.ids import InvalidIdError, asset_id, stable_seed, validate_id
from gridguard.domain.telemetry import (
    EVENT_TIME_COLUMN,
    INGEST_TIME_COLUMN,
    QUALITY_COLUMN,
    SEQUENCE_COLUMN,
    QualityFlag,
    TelemetryRecord,
    event_time_to_local,
    local_to_event_time,
)

__all__ = [
    "EVENT_TIME_COLUMN",
    "FAULT_CATEGORIES",
    "GENERATING_KINDS",
    "INGEST_TIME_COLUMN",
    "QUALITY_COLUMN",
    "SENSING_KINDS",
    "SEQUENCE_COLUMN",
    "Asset",
    "AssetKind",
    "AssetTree",
    "Evidence",
    "FaultCategory",
    "FaultHypothesis",
    "InvalidIdError",
    "QualityFlag",
    "TelemetryRecord",
    "asset_id",
    "build_asset_id",
    "category_for",
    "event_time_to_local",
    "local_to_event_time",
    "stable_seed",
    "validate_id",
]
