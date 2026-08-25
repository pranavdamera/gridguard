"""
The collector.

Takes wire batches, writes them to the store idempotently, and records what it
learned about the agent that sent them. Transport-agnostic: it is handed bytes,
so the same code path serves an MQTT subscription, a replayed capture file, and
a test.

Arrival lag becomes real here
-----------------------------
Up to this point ``ingest_time`` has equalled ``event_time``, because nothing
travelled. The collector is the first component that genuinely receives
something, so it stamps arrival itself rather than trusting the sender: an edge
node's clock is not evidence about when someone else took delivery. The
difference between the two is arrival lag, and it is the quantity the whole
migration exists to reason about.

What the collector does not do
------------------------------
It does not detect, score, or attribute. It writes down what arrived and when.
Keeping ingestion free of judgement is what lets a later experiment replay the
same capture through different detectors and compare them on identical input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from gridguard.collector.store import TelemetryStore
from gridguard.contract.codec import decode_batch, deserialize_batch
from gridguard.domain.telemetry import TelemetryRecord

logger = logging.getLogger(__name__)


@dataclass
class CollectorStats:
    """What the collector has seen."""

    batches: int = 0
    records: int = 0
    inserted: int = 0
    duplicates: int = 0
    replays: int = 0
    decode_failures: int = 0

    def as_dict(self) -> dict:
        return {
            "batches": self.batches,
            "records": self.records,
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "replays": self.replays,
            "decode_failures": self.decode_failures,
        }


@dataclass(frozen=True)
class BatchOutcome:
    """What happened to one batch."""

    agent_id: str
    site_id: str
    records: int
    inserted: int
    duplicates: int
    is_replay: bool
    max_arrival_lag_seconds: float | None

    @property
    def was_entirely_duplicate(self) -> bool:
        """True when nothing in this batch was new.

        The expected outcome of replaying a batch the store already holds, and
        the thing the phase gate checks.
        """
        return self.records > 0 and self.inserted == 0


@dataclass
class Collector:
    """Writes delivered telemetry into the store, idempotently."""

    store: TelemetryStore
    stats: CollectorStats = field(default_factory=CollectorStats)

    def handle_payload(
        self, payload: bytes, *, received_at: datetime | None = None
    ) -> BatchOutcome | None:
        """Decode and store one wire batch.

        A payload that will not decode is counted and dropped rather than
        raised: one malformed message from one agent must not stop the
        collector from serving every other agent. It is logged loudly, because
        silent tolerance of corruption is how a fleet-wide encoding bug goes
        unnoticed for a week.
        """
        try:
            batch = deserialize_batch(payload)
            records = decode_batch(batch)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            self.stats.decode_failures += 1
            logger.error("Dropping undecodable payload (%d bytes): %s", len(payload), exc)
            return None

        return self.handle_batch(
            records,
            agent_id=batch.agent_id,
            site_id=batch.site_id,
            is_replay=batch.is_replay,
            buffered_remaining=batch.buffered_remaining,
            received_at=received_at,
        )

    def handle_batch(
        self,
        records: list[TelemetryRecord],
        *,
        agent_id: str,
        site_id: str,
        is_replay: bool = False,
        buffered_remaining: int = 0,
        received_at: datetime | None = None,
    ) -> BatchOutcome:
        """Store already-decoded records."""
        arrived = received_at or datetime.now(UTC)

        result = self.store.write(
            records, ingest_time=arrived, agent_id=agent_id, is_replay=is_replay
        )

        self.stats.batches += 1
        self.stats.records += len(records)
        self.stats.inserted += result.inserted
        self.stats.duplicates += result.updated
        if is_replay:
            self.stats.replays += 1

        last_event_time = max((r.event_time for r in records), default=None)
        lag = max((arrived - r.event_time).total_seconds() for r in records) if records else None

        self.store.record_agent_batch(
            agent_id=agent_id,
            site_id=site_id,
            seen_at=arrived,
            last_event_time=last_event_time,
            records=len(records),
            duplicates=result.updated,
            buffered_remaining=buffered_remaining,
        )

        if is_replay:
            logger.info(
                "%s: replay of %d record(s) — %d new, %d already held, %d still buffered.",
                agent_id,
                len(records),
                result.inserted,
                result.updated,
                buffered_remaining,
            )

        return BatchOutcome(
            agent_id=agent_id,
            site_id=site_id,
            records=len(records),
            inserted=result.inserted,
            duplicates=result.updated,
            is_replay=is_replay,
            max_arrival_lag_seconds=lag,
        )
