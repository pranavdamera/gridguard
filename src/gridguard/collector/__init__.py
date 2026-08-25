"""
The collector and the telemetry store.

Takes wire batches, writes them idempotently, and records what is known about
delivery. Transport-agnostic: it is handed bytes, so one code path serves an
MQTT subscription, a replayed capture, and a test.

Idempotency is the property this layer exists for. An agent recovering from an
outage resends records the collector may already hold, and every write is an
upsert keyed on ``(asset_id, channel, event_time)`` — the identity of a
measurement, independent of how often it was sent or in what order it arrived.
Applying a batch twice leaves the table exactly as one application would.

``ingest_time`` is stamped here, not taken from the sender: an edge node's clock
is not evidence about when someone else took delivery. The gap between it and
``event_time`` is arrival lag, and it is the first quantity in this project that
describes the network rather than the plant.

Backends
--------
One SQL dialect runs on both SQLite and Postgres, and the same test suite runs
against both — SQLite always, Postgres when reachable. Postgres is the deployed
choice (decision D2); SQLite is the test backend and a single-process option.
"""

from gridguard.collector.collector import BatchOutcome, Collector, CollectorStats
from gridguard.collector.metrics import (
    AgentHealth,
    CompletenessSummary,
    LagSummary,
    agent_health,
    arrival_lag,
    completeness,
)
from gridguard.collector.store import (
    SCHEMA,
    StoredRecord,
    TelemetryStore,
    WriteResult,
    from_epoch_micros,
    postgres_store,
    sqlite_store,
    to_epoch_micros,
)

__all__ = [
    "SCHEMA",
    "AgentHealth",
    "BatchOutcome",
    "Collector",
    "CollectorStats",
    "CompletenessSummary",
    "LagSummary",
    "StoredRecord",
    "TelemetryStore",
    "WriteResult",
    "agent_health",
    "arrival_lag",
    "completeness",
    "from_epoch_micros",
    "postgres_store",
    "sqlite_store",
    "to_epoch_micros",
]
