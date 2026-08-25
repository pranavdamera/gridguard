"""
Translation between the domain model and the wire contract.

Two representations, deliberately kept apart:

``gridguard.domain.telemetry.TelemetryRecord``
    What the rest of the project reasons about. Python types, validation in
    ``__post_init__``, no wire concerns.

``gridguard.contract.v1.telemetry_pb2.TelemetryRecord``
    What travels. Field numbers, presence semantics, forward compatibility.

Keeping them separate costs a translation layer and buys two things. The domain
model stays free to change shape without breaking every deployed agent, and the
wire format stays free to gain fields without the domain model growing a
``google.protobuf`` import. This module is the only place that knows both.

Presence is the thing to get right
----------------------------------
A PV array legitimately reports zero at night. An absent reading is a different
statement entirely. proto3 without explicit presence cannot tell those apart —
an unset ``double`` reads back as ``0.0`` — so ``value`` is declared ``optional``
in the schema and this module never substitutes a default for it. The domain
model's rule (a record with no value must carry ``MISSING``) is enforced on the
way back in, so a malformed message fails loudly rather than becoming a plausible
zero.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from google.protobuf.timestamp_pb2 import Timestamp

from gridguard.contract.v1 import telemetry_pb2 as pb
from gridguard.domain.telemetry import QualityFlag, TelemetryRecord

logger = logging.getLogger(__name__)

#: Payload schema version carried in every batch. Bumped for additive changes
#: within ``gridguard.v1``; a breaking change means a new proto package.
SCHEMA_VERSION = "1.0"


class ContractError(ValueError):
    """Raised when a message cannot be represented, or cannot be trusted."""


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def to_timestamp(value: datetime) -> Timestamp:
    """UTC datetime → protobuf Timestamp."""
    if value.tzinfo is None:
        raise ContractError(
            "Refusing to encode a naive datetime. Simulated and measured time are "
            "both UTC by contract; a naive value here means an offset was lost."
        )
    stamp = Timestamp()
    stamp.FromDatetime(value.astimezone(UTC))
    return stamp


def from_timestamp(stamp: Timestamp) -> datetime:
    """protobuf Timestamp → timezone-aware UTC datetime."""
    return stamp.ToDatetime(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def encode_record(record: TelemetryRecord) -> pb.TelemetryRecord:
    """Domain record → wire record."""
    message = pb.TelemetryRecord(
        asset_id=record.asset_id,
        site_id=record.site_id,
        channel=record.channel,
        unit=record.unit,
        quality=int(record.quality),
    )

    # Only set when present. Assigning None would be a type error, and assigning
    # 0.0 would silently turn "no reading" into "read zero".
    if record.value is not None:
        message.value = float(record.value)

    message.event_time.CopyFrom(to_timestamp(record.event_time))
    if record.ingest_time is not None:
        message.ingest_time.CopyFrom(to_timestamp(record.ingest_time))
    if record.sequence is not None:
        message.sequence = int(record.sequence)

    return message


def decode_record(message: pb.TelemetryRecord) -> TelemetryRecord:
    """Wire record → domain record.

    Every optional field is read through its presence check rather than its
    value, so a zero reading and an absent reading stay distinct all the way
    into the domain model.
    """
    if not message.HasField("event_time"):
        raise ContractError(
            f"{message.asset_id}/{message.channel}: event_time is required. A record "
            "with no instant cannot be ordered, joined to weather, or scored."
        )

    return TelemetryRecord(
        asset_id=message.asset_id,
        site_id=message.site_id,
        channel=message.channel,
        unit=message.unit,
        value=message.value if message.HasField("value") else None,
        event_time=from_timestamp(message.event_time),
        ingest_time=(
            from_timestamp(message.ingest_time) if message.HasField("ingest_time") else None
        ),
        # Sequence 0 is a legitimate first record, so presence is not inferable
        # from the value. proto3 scalars have no presence without `optional`,
        # and adding it here would cost a wire-format change for a field that is
        # always set by the agent — so 0 means 0.
        sequence=int(message.sequence),
        quality=QualityFlag(message.quality),
    )


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def encode_batch(
    records: list[TelemetryRecord],
    *,
    agent_id: str,
    site_id: str,
    sent_at: datetime,
    is_replay: bool = False,
    buffered_remaining: int = 0,
) -> pb.TelemetryBatch:
    """Domain records → one wire batch."""
    batch = pb.TelemetryBatch(
        schema_version=SCHEMA_VERSION,
        agent_id=agent_id,
        site_id=site_id,
        is_replay=is_replay,
        buffered_remaining=buffered_remaining,
    )
    batch.sent_at.CopyFrom(to_timestamp(sent_at))
    batch.records.extend(encode_record(r) for r in records)
    return batch


def decode_batch(message: pb.TelemetryBatch) -> list[TelemetryRecord]:
    """Wire batch → domain records.

    A batch whose ``schema_version`` this build does not recognise is logged and
    decoded anyway. That is deliberate: refusing unknown *minor* versions would
    make a partially-upgraded fleet stop reporting, which is a worse failure
    than reading a message that may carry fields we ignore. Protobuf preserves
    those fields, so nothing is destroyed by passing through an older reader.
    """
    if message.schema_version and message.schema_version != SCHEMA_VERSION:
        logger.info(
            "Batch from %s declares schema %s; this build speaks %s. Decoding anyway — "
            "unknown fields are preserved, not dropped.",
            message.agent_id,
            message.schema_version,
            SCHEMA_VERSION,
        )
    return [decode_record(r) for r in message.records]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def serialize(message) -> bytes:
    return message.SerializeToString()


def deserialize_batch(payload: bytes) -> pb.TelemetryBatch:
    batch = pb.TelemetryBatch()
    batch.ParseFromString(payload)
    return batch


def deserialize_record(payload: bytes) -> pb.TelemetryRecord:
    record = pb.TelemetryRecord()
    record.ParseFromString(payload)
    return record


def unknown_field_count(message) -> int:
    """How many fields a parsed message carried that this build does not know.

    ``UnknownFieldSet`` rather than ``message.UnknownFields()``: the latter is a
    pure-Python-implementation accessor and raises ``NotImplementedError`` under
    the default upb (C++) backend, which is what ships in every wheel we install.

    For observability rather than control flow. A rising count means part of the
    fleet is running a newer schema — worth surfacing before someone debugs a
    "missing" field that was never dropped.
    """
    from google.protobuf.unknown_fields import UnknownFieldSet

    return len(UnknownFieldSet(message))


def has_unknown_fields(message) -> bool:
    """Whether a parsed message carried fields this build does not know."""
    return unknown_field_count(message) > 0
