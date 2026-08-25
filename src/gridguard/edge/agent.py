"""
The edge agent.

One process per site group. Reads telemetry, flags what it can see locally,
buffers durably, publishes when it can, and replays what it could not.

    ingest → quality checks → buffer (durable) → publish → acknowledge

The order is the design. Every record is on disk before any attempt to send it,
and no record is acknowledged until the transport has confirmed it. That
ordering is what makes "no sample lost, none duplicated" achievable, and every
plausible shortcut breaks one half of it:

* Publishing before buffering loses records when the process dies mid-send.
* Acknowledging before confirmation loses records when the send silently fails.
* Renumbering on replay duplicates records the collector already holds.

Sequence numbers are assigned once, at ingest, and persisted, per
``(asset_id, channel)``. A record keeps its number across a restart, so
``(asset_id, channel, sequence)`` remains a stable idempotency key and the
collector can deduplicate a replay with an upsert. The channel belongs in that
key because one asset emits several channels per interval — an asset-scoped key
collides between them.

Not in this phase
-----------------
There is no MQTT client here. The transport is an interface, and phase 6 brings
the real one alongside the broker and the collector. The agent's failure
behaviour is what phase 5 has to get right, and that is established against a
transport that can be made to fail deterministically — which a real broker
cannot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gridguard.contract.codec import encode_batch, serialize
from gridguard.domain.telemetry import TelemetryRecord
from gridguard.edge.buffer import OutboxBuffer
from gridguard.edge.quality import QualityChecker
from gridguard.edge.transport import Transport, TransportError, telemetry_topic

logger = logging.getLogger(__name__)

#: Records per published batch. Bounded so one enormous replay after a long
#: outage does not become a single message the broker refuses.
DEFAULT_BATCH_SIZE = 500


@dataclass
class AgentStats:
    """What the agent has done, for observability and for tests."""

    ingested: int = 0
    published: int = 0
    acknowledged: int = 0
    publish_failures: int = 0
    replays: int = 0

    def as_dict(self) -> dict:
        return {
            "ingested": self.ingested,
            "published": self.published,
            "acknowledged": self.acknowledged,
            "publish_failures": self.publish_failures,
            "replays": self.replays,
        }


@dataclass
class EdgeAgent:
    """Buffers telemetry locally and delivers it when it can.

    Args:
        agent_id: this agent's identity, carried on every batch.
        site_id: the site it reports for.
        transport: how it publishes. Any :class:`Transport` implementation.
        buffer_path: where the durable outbox lives.
        batch_size: maximum records per published batch.
    """

    agent_id: str
    site_id: str
    transport: Transport
    buffer_path: Path | str
    batch_size: int = DEFAULT_BATCH_SIZE

    buffer: OutboxBuffer = field(init=False)
    checker: QualityChecker = field(default_factory=QualityChecker)
    stats: AgentStats = field(default_factory=AgentStats)

    #: True once a publish has failed and records are waiting. The next
    #: successful send is marked as a replay so the collector can tell a
    #: recovery from a node whose clock has drifted.
    _pending_replay: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.buffer = OutboxBuffer(self.buffer_path)

    def close(self) -> None:
        self.buffer.close()

    def __enter__(self) -> EdgeAgent:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- ingest -------------------------------------------------------------

    def ingest(self, records: list[TelemetryRecord]) -> int:
        """Flag, number and durably buffer records. Sends nothing.

        Returns how many were newly buffered. Re-ingesting a record the buffer
        already holds is a no-op, so an agent restarted mid-batch can safely be
        handed the same input again without knowing how far it had got.
        """
        buffered = 0
        for record in records:
            checked = self.checker.check(record)

            # A record that already carries a sequence keeps it — that is what
            # makes re-ingest idempotent. Only genuinely new records draw one.
            sequence = (
                checked.sequence
                if checked.sequence is not None
                else self.buffer.next_sequence(checked.asset_id, checked.channel)
            )
            numbered = _with_sequence(checked, sequence)

            if self.buffer.append(
                asset_id=numbered.asset_id,
                channel=numbered.channel,
                sequence=sequence,
                event_time=numbered.event_time.isoformat(),
                payload=serialize(_encode_single(numbered)),
            ):
                buffered += 1

        self.stats.ingested += buffered
        return buffered

    # -- publish ------------------------------------------------------------

    def flush(self, *, now: datetime | None = None) -> int:
        """Try to deliver everything buffered. Returns records acknowledged.

        Records are acknowledged only after the transport returns without
        raising. A failure leaves them pending, so the next flush retries them
        with their original sequence numbers — which is what lets the collector
        deduplicate rather than double-count.
        """
        sent_at = now or datetime.now(UTC)
        acknowledged = 0

        while True:
            pending = self.buffer.pending(limit=self.batch_size)
            if not pending:
                break

            records = [_decode_single(item.payload) for item in pending]
            remaining = max(0, self.buffer.pending_count() - len(pending))

            batch = encode_batch(
                records,
                agent_id=self.agent_id,
                site_id=self.site_id,
                sent_at=sent_at,
                is_replay=self._pending_replay,
                buffered_remaining=remaining,
            )

            try:
                self.transport.publish(
                    telemetry_topic(self.site_id, self.agent_id), serialize(batch)
                )
            except TransportError as exc:
                self.stats.publish_failures += 1
                self._pending_replay = True
                logger.info(
                    "%s: publish failed (%s). %d record(s) stay buffered.",
                    self.agent_id,
                    exc,
                    self.buffer.pending_count(),
                )
                return acknowledged

            if self._pending_replay:
                self.stats.replays += 1
                self._pending_replay = False

            self.stats.published += len(pending)
            newly_acked = self.buffer.acknowledge([item.id for item in pending])
            acknowledged += newly_acked
            self.stats.acknowledged += newly_acked

        return acknowledged

    def publish(self, records: list[TelemetryRecord], *, now: datetime | None = None) -> int:
        """Ingest then flush. The ordinary path for a live agent."""
        self.ingest(records)
        return self.flush(now=now)

    # -- state --------------------------------------------------------------

    @property
    def buffered(self) -> int:
        """Records held locally and not yet delivered."""
        return self.buffer.pending_count()

    @property
    def is_behind(self) -> bool:
        """Whether the agent is buffering rather than delivering live."""
        return self.buffered > 0

    def status(self, *, observed_at: datetime | None = None):
        """This agent's liveness, as a wire message.

        Published on its own topic, separate from telemetry, because "is the
        plant generating" and "can we still hear the node" fail independently
        and a design that infers the second from the first cannot tell a dark
        site from a deaf one.
        """
        from gridguard.contract.codec import to_timestamp
        from gridguard.contract.v1 import telemetry_pb2 as pb

        state = pb.NODE_STATE_DEGRADED if self.is_behind else pb.NODE_STATE_ONLINE
        message = pb.NodeStatus(
            agent_id=self.agent_id,
            site_id=self.site_id,
            state=state,
            buffered_records=self.buffered,
            detail=("buffering; not delivering live" if self.is_behind else "delivering live"),
        )
        message.observed_at.CopyFrom(to_timestamp(observed_at or datetime.now(UTC)))
        return message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_sequence(record: TelemetryRecord, sequence: int) -> TelemetryRecord:
    if record.sequence == sequence:
        return record
    return TelemetryRecord(
        asset_id=record.asset_id,
        site_id=record.site_id,
        channel=record.channel,
        value=record.value,
        unit=record.unit,
        event_time=record.event_time,
        ingest_time=record.ingest_time,
        sequence=sequence,
        quality=record.quality,
        metadata=record.metadata,
    )


def _encode_single(record: TelemetryRecord):
    """One record as a wire message, for storage in the buffer."""
    from gridguard.contract.codec import encode_record

    return encode_record(record)


def _decode_single(payload: bytes) -> TelemetryRecord:
    from gridguard.contract.codec import decode_record, deserialize_record

    return decode_record(deserialize_record(payload))
