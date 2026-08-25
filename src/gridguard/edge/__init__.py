"""
The edge agent.

One process per site group: reads telemetry, flags what it can see locally,
buffers it durably, publishes when it can, and replays what it could not.

    ingest → quality checks → buffer (durable) → publish → acknowledge

Every record reaches disk before any attempt to send it, and none is
acknowledged until the transport confirms it. That ordering is what makes "no
sample lost, none duplicated" achievable across a broker outage *and* a process
restart. Sequence numbers are assigned once and persisted, so a replay reuses
them and the collector can deduplicate with an upsert on
``(asset_id, sequence)``.

The transport is an interface. Phase 5 ships an in-process implementation that
can be made to fail deterministically, which is what the loss and duplication
properties are established against; the MQTT client arrives in phase 6 with the
broker and the collector.
"""

from gridguard.edge.agent import DEFAULT_BATCH_SIZE, AgentStats, EdgeAgent
from gridguard.edge.buffer import BufferedRecord, OutboxBuffer
from gridguard.edge.quality import CHANNEL_RANGES, ChannelHistory, QualityChecker
from gridguard.edge.screening import (
    DEFAULT_SCREEN_FRACTION,
    ClearSkyScreen,
    ScreenResult,
)
from gridguard.edge.transport import (
    InMemoryTransport,
    Transport,
    TransportError,
    status_topic,
    telemetry_topic,
)

__all__ = [
    "CHANNEL_RANGES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SCREEN_FRACTION",
    "AgentStats",
    "BufferedRecord",
    "ChannelHistory",
    "ClearSkyScreen",
    "EdgeAgent",
    "InMemoryTransport",
    "OutboxBuffer",
    "QualityChecker",
    "ScreenResult",
    "Transport",
    "TransportError",
    "status_topic",
    "telemetry_topic",
]
