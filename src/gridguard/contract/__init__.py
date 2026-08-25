"""
The wire contract.

What travels between an edge agent and the collector, defined in
``proto/gridguard/contract/v1/telemetry.proto`` and compiled into
``gridguard.contract.v1``.

The schema is the contract, and it is independent of the transport that carries
it. MQTT is the chosen broker (see the architecture notes), but nothing in the
proto mentions MQTT — so changing transport later touches the agent and the
collector, not the schema and not anything that reads it.

Phase 4 defines the contract and nothing sends it anywhere. That ordering is on
purpose: fixing the message shape before there is a network means the first
thing to cross a wire is already versioned, already round-trip tested, and
already able to survive a partially-upgraded fleet.

Regenerating
------------
The stubs in ``v1/`` are generated and committed, so running the project needs
only the ``protobuf`` runtime. Regenerate after editing the ``.proto``::

    make proto

``tests/test_contract.py`` fails if the committed stubs no longer match the
schema, which is the same drift guard the Dockerfile has: two copies of one
truth, checked rather than trusted.
"""

from gridguard.contract.codec import (
    SCHEMA_VERSION,
    ContractError,
    decode_batch,
    decode_record,
    deserialize_batch,
    deserialize_record,
    encode_batch,
    encode_record,
    from_timestamp,
    has_unknown_fields,
    serialize,
    to_timestamp,
    unknown_field_count,
)

__all__ = [
    "SCHEMA_VERSION",
    "ContractError",
    "decode_batch",
    "decode_record",
    "deserialize_batch",
    "deserialize_record",
    "encode_batch",
    "encode_record",
    "from_timestamp",
    "has_unknown_fields",
    "serialize",
    "to_timestamp",
    "unknown_field_count",
]
