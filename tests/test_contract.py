"""
Wire contract tests.

Three properties carry real weight here, and each corresponds to a way a
distributed fleet breaks in practice:

1. **Presence.** An absent reading and a reading of zero must stay distinct all
   the way from the domain model, onto the wire, and back. A PV array reports
   zero legitimately every night, so conflating the two would make every
   nightfall look like a data loss and every data loss look like nightfall.

2. **Forward compatibility.** A reader built against an older schema must pass
   unknown fields through untouched. Without this, a partially-upgraded fleet
   silently destroys data every time an old node relays a new message.

3. **Stability.** Field numbers are the contract. Renumbering one silently
   reinterprets every message already in flight or on disk.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from gridguard.contract import (
    SCHEMA_VERSION,
    ContractError,
    decode_batch,
    decode_record,
    deserialize_batch,
    deserialize_record,
    encode_batch,
    encode_record,
    has_unknown_fields,
    serialize,
)
from gridguard.contract.v1 import telemetry_pb2 as pb
from gridguard.domain.telemetry import QualityFlag, TelemetryRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO = REPO_ROOT / "proto" / "gridguard" / "contract" / "v1" / "telemetry.proto"
GENERATED = REPO_ROOT / "src" / "gridguard" / "contract" / "v1"

EVENT_TIME = datetime(2016, 6, 1, 17, 0, tzinfo=UTC)


def _record(**overrides) -> TelemetryRecord:
    kwargs = {
        "asset_id": "nist_roof_array_1",
        "site_id": "nist_roof",
        "channel": "ac_power_kw",
        "unit": "kW",
        "value": 42.5,
        "event_time": EVENT_TIME,
        "sequence": 7,
        "quality": QualityFlag.OK,
    }
    kwargs.update(overrides)
    return TelemetryRecord(**kwargs)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_record_round_trips_through_bytes():
    original = _record(ingest_time=EVENT_TIME + timedelta(seconds=3))
    restored = decode_record(deserialize_record(serialize(encode_record(original))))

    assert restored.asset_id == original.asset_id
    assert restored.site_id == original.site_id
    assert restored.channel == original.channel
    assert restored.unit == original.unit
    assert restored.value == original.value
    assert restored.event_time == original.event_time
    assert restored.ingest_time == original.ingest_time
    assert restored.sequence == original.sequence
    assert restored.quality == original.quality


def test_batch_round_trips_through_bytes():
    records = [_record(sequence=i, value=float(i)) for i in range(5)]
    batch = encode_batch(
        records,
        agent_id="agent_nist_roof",
        site_id="nist_roof",
        sent_at=EVENT_TIME,
        is_replay=True,
        buffered_remaining=12,
    )
    parsed = deserialize_batch(serialize(batch))

    assert parsed.schema_version == SCHEMA_VERSION
    assert parsed.agent_id == "agent_nist_roof"
    assert parsed.is_replay is True
    assert parsed.buffered_remaining == 12

    restored = decode_batch(parsed)
    assert [r.sequence for r in restored] == [0, 1, 2, 3, 4]
    assert [r.value for r in restored] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_timestamps_stay_utc_and_exact():
    original = _record(event_time=datetime(2016, 12, 31, 23, 59, 59, 123456, tzinfo=UTC))
    restored = decode_record(deserialize_record(serialize(encode_record(original))))
    assert restored.event_time == original.event_time
    assert restored.event_time.tzinfo is not None


def test_encoding_a_naive_datetime_is_refused():
    """A naive datetime on the wire means an offset was silently lost.

    The domain model already rejects one, so this exercises the codec's own
    guard directly — the codec is also called from the simulator and the future
    collector, which do not all construct through TelemetryRecord.
    """
    from gridguard.contract.codec import to_timestamp

    with pytest.raises(ContractError, match="naive datetime"):
        to_timestamp(datetime(2016, 6, 1, 12, 0))


# ---------------------------------------------------------------------------
# Presence — zero is not absence
# ---------------------------------------------------------------------------


def test_a_zero_reading_survives_as_zero():
    """Night is not a data loss."""
    restored = decode_record(deserialize_record(serialize(encode_record(_record(value=0.0)))))
    assert restored.value == 0.0
    assert restored.value is not None
    assert not (restored.quality & QualityFlag.MISSING)


def test_an_absent_reading_survives_as_absent():
    """A data loss is not night."""
    original = _record(value=None, quality=QualityFlag.MISSING)
    restored = decode_record(deserialize_record(serialize(encode_record(original))))
    assert restored.value is None
    assert restored.quality & QualityFlag.MISSING


def test_zero_and_absent_produce_different_bytes():
    """The distinction is carried on the wire, not reconstructed by luck."""
    zero = serialize(encode_record(_record(value=0.0)))
    absent = serialize(encode_record(_record(value=None, quality=QualityFlag.MISSING)))
    assert zero != absent


def test_value_field_declares_explicit_presence():
    """Guards the schema decision, not just its current effect.

    Without `optional` on `value`, proto3 has no presence for a double and an
    absent reading decodes as 0.0. The test asserts the schema property so the
    keyword cannot be dropped in a later edit without something failing.
    """
    field = pb.TelemetryRecord.DESCRIPTOR.fields_by_name["value"]
    assert field.has_presence, "TelemetryRecord.value must be `optional` in the .proto"


def test_a_missing_value_without_the_missing_flag_is_refused():
    """The domain rule is enforced on the way back in, not only on the way out."""
    message = pb.TelemetryRecord(
        asset_id="a_array_1", site_id="a", channel="ac_power_kw", unit="kW", quality=0
    )
    message.event_time.FromDatetime(EVENT_TIME)
    with pytest.raises(ValueError, match="MISSING"):
        decode_record(message)


def test_a_record_without_event_time_is_refused():
    message = pb.TelemetryRecord(
        asset_id="a_array_1", site_id="a", channel="ac_power_kw", unit="kW"
    )
    message.value = 1.0
    with pytest.raises(ContractError, match="event_time is required"):
        decode_record(message)


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------


def test_composed_quality_flags_survive_the_wire():
    original = _record(quality=QualityFlag.STALE | QualityFlag.OUT_OF_ORDER)
    restored = decode_record(deserialize_record(serialize(encode_record(original))))
    assert restored.quality & QualityFlag.STALE
    assert restored.quality & QualityFlag.OUT_OF_ORDER
    assert not restored.quality & QualityFlag.FROZEN


def test_proto_quality_enum_matches_the_domain_flags():
    """Two copies of one truth, so they are checked rather than trusted."""
    for flag in QualityFlag:
        if flag is QualityFlag.OK:
            continue
        name = f"QUALITY_FLAG_{flag.name}"
        assert name in pb.QualityFlag.keys(), f"{name} missing from the proto enum"
        assert pb.QualityFlag.Value(name) == int(flag), f"{name} value differs from the domain"


# ---------------------------------------------------------------------------
# Forward compatibility
# ---------------------------------------------------------------------------


def _append_unknown_field(payload: bytes, field_number: int, value: int) -> bytes:
    """Append a varint field the current schema does not define.

    Hand-encoded rather than compiled from a second .proto, so the test needs no
    protoc at run time and cannot drift from a fixture schema.
    """
    tag = (field_number << 3) | 0  # wire type 0 = varint
    out = bytearray(payload)
    for part in (tag, value):
        while True:
            byte = part & 0x7F
            part >>= 7
            out.append(byte | (0x80 if part else 0))
            if not part:
                break
    return bytes(out)


def test_unknown_fields_are_preserved_through_an_older_reader():
    """A partially-upgraded fleet must not destroy data it does not understand.

    Field 900 stands in for something a newer schema added. This build has never
    heard of it, parses the message anyway, and must hand the bytes back intact
    when it re-serialises.
    """
    payload = serialize(encode_record(_record()))
    extended = _append_unknown_field(payload, field_number=900, value=12345)

    parsed = deserialize_record(extended)
    assert has_unknown_fields(parsed), "protobuf did not retain the unknown field"

    # The known fields decoded correctly ...
    restored = decode_record(parsed)
    assert restored.value == 42.5
    assert restored.sequence == 7

    # ... and re-serialising gives the unknown field back, byte for byte.
    assert serialize(parsed) == extended


def test_a_clean_message_reports_no_unknown_fields():
    """The counterpart, so the check above is not always-true."""
    parsed = deserialize_record(serialize(encode_record(_record())))
    assert not has_unknown_fields(parsed)


def test_an_unrecognised_schema_version_still_decodes():
    """Refusing unknown minor versions would stop a fleet mid-upgrade."""
    batch = encode_batch([_record()], agent_id="a", site_id="nist_roof", sent_at=EVENT_TIME)
    batch.schema_version = "1.99"
    restored = decode_batch(deserialize_batch(serialize(batch)))
    assert len(restored) == 1


# ---------------------------------------------------------------------------
# Stability of the contract
# ---------------------------------------------------------------------------

#: Field numbers are the contract. Renumbering one silently reinterprets every
#: message already in flight or on disk, so the numbering is pinned here and a
#: change has to be deliberate enough to edit this table.
EXPECTED_FIELD_NUMBERS = {
    "TelemetryRecord": {
        "asset_id": 1,
        "site_id": 2,
        "channel": 3,
        "unit": 4,
        "value": 5,
        "event_time": 6,
        "ingest_time": 7,
        "sequence": 8,
        "quality": 9,
    },
    "TelemetryBatch": {
        "schema_version": 1,
        "agent_id": 2,
        "site_id": 3,
        "records": 4,
        "sent_at": 5,
        "is_replay": 6,
        "buffered_remaining": 7,
    },
    "NodeStatus": {
        "agent_id": 1,
        "site_id": 2,
        "state": 3,
        "observed_at": 4,
        "detail": 5,
        "buffered_records": 6,
        "last_delivered_sequence": 7,
    },
}


@pytest.mark.parametrize("message_name", sorted(EXPECTED_FIELD_NUMBERS))
def test_field_numbers_are_stable(message_name: str):
    descriptor = getattr(pb, message_name).DESCRIPTOR
    actual = {f.name: f.number for f in descriptor.fields}
    assert actual == EXPECTED_FIELD_NUMBERS[message_name]


def test_hot_fields_use_one_byte_tags():
    """Fields present on every record belong below 16, where the tag is one byte."""
    descriptor = pb.TelemetryRecord.DESCRIPTOR
    for name in ("asset_id", "site_id", "channel", "value", "event_time", "sequence"):
        assert descriptor.fields_by_name[name].number < 16


def test_the_wire_package_carries_the_major_version():
    """A breaking change means a new package, so the package name is the version."""
    assert pb.TelemetryRecord.DESCRIPTOR.full_name.startswith("gridguard.v1.")


def test_removed_field_numbers_are_reserved():
    """Reserved numbers cannot be reassigned to something with a new meaning."""
    from google.protobuf import descriptor_pb2

    # Reserved ranges are not exposed on the upb descriptor object, so they are
    # read back from the descriptor proto instead.
    descriptor = descriptor_pb2.DescriptorProto()
    pb.TelemetryRecord.DESCRIPTOR.CopyToProto(descriptor)

    ranges = [(r.start, r.end) for r in descriptor.reserved_range]
    assert ranges, "TelemetryRecord should reserve the numbers it has retired"
    assert any(start <= 10 < end for start, end in ranges)
    assert set(descriptor.reserved_name) == {"site_name", "data_mode"}


# ---------------------------------------------------------------------------
# Generated stubs must match the schema
# ---------------------------------------------------------------------------


def test_committed_stubs_match_the_proto():
    """Two copies of one truth, checked rather than trusted.

    The stubs are committed so that running the project needs only the protobuf
    runtime. That is the same arrangement the Dockerfile's dependency list once
    had, and it drifted — so regeneration is compared here rather than assumed.
    """
    pytest.importorskip("grpc_tools", reason="grpcio-tools is a dev dependency")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={REPO_ROOT / 'proto'}",
                f"--python_out={tmp}",
                str(PROTO),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"protoc failed:\n{result.stderr}"

        regenerated = Path(tmp) / "gridguard" / "contract" / "v1" / "telemetry_pb2.py"
        committed = GENERATED / "telemetry_pb2.py"
        assert (
            regenerated.read_text() == committed.read_text()
        ), "Committed protobuf stubs no longer match the .proto. Run `make proto`."


# ---------------------------------------------------------------------------
# Bridge: simulator frames ↔ wire records
# ---------------------------------------------------------------------------


def test_simulator_output_becomes_wire_records():
    """The join between the simulator and what will travel in phase 5."""
    from gridguard.contract.bridge import CHANNEL_UNITS, records_from_frame
    from gridguard.simulation import FleetSimulator, SimulationConfig

    result = FleetSimulator(
        SimulationConfig(site_ids=("nist_roof",), start="2016-06-01", end="2016-06-01 05:45")
    ).run()
    frame = result.canonical()

    records = records_from_frame(frame)
    assert len(records) == len(frame) * len(CHANNEL_UNITS)
    assert {r.channel for r in records} == set(CHANNEL_UNITS)
    assert all(r.event_time.tzinfo is not None for r in records)
    assert all(r.unit for r in records), "every record must carry its unit"


def test_wire_records_round_trip_back_into_a_frame():
    from gridguard.contract.bridge import frame_from_records, records_from_frame
    from gridguard.simulation import FleetSimulator, SimulationConfig

    result = FleetSimulator(
        SimulationConfig(site_ids=("nist_roof",), start="2016-06-01", end="2016-06-01 05:45")
    ).run()
    original = result.canonical()

    rebuilt = frame_from_records(records_from_frame(original))
    assert len(rebuilt) == len(original)
    for channel in ("ac_power_kw", "irradiance_wm2", "temperature_c", "wind_speed_ms"):
        pd.testing.assert_series_equal(
            original[channel].reset_index(drop=True).astype(float),
            rebuilt[channel].reset_index(drop=True).astype(float),
            check_names=False,
        )


def test_a_nan_reading_narrows_to_absent_not_zero():
    """The bridge must not invent a zero where the simulator had no number."""
    import numpy as np

    from gridguard.contract.bridge import records_from_frame

    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2016-06-01", periods=2, freq="15min", tz=UTC),
            "site_id": "nist_roof",
            "asset_id": "nist_roof_array_1",
            "ac_power_kw": [10.0, np.nan],
            "irradiance_wm2": [500.0, 480.0],
        }
    )
    records = records_from_frame(frame)
    power = [r for r in records if r.channel == "ac_power_kw"]

    assert power[0].value == 10.0
    assert power[1].value is None
    assert power[1].quality & QualityFlag.MISSING


def test_widening_keeps_a_partially_delivered_interval():
    """Power arrived, irradiance did not — the row survives with a gap.

    Dropping it would discard a real reading; defaulting the gap to zero would
    invent one. Both are worse than an honest NaN.
    """
    from gridguard.contract.bridge import frame_from_records

    records = [
        TelemetryRecord(
            asset_id="nist_roof_array_1",
            site_id="nist_roof",
            channel="ac_power_kw",
            unit="kW",
            value=12.0,
            event_time=EVENT_TIME,
            sequence=1,
        )
    ]
    frame = frame_from_records(records)
    assert len(frame) == 1
    assert frame["ac_power_kw"].iloc[0] == 12.0
    assert pd.isna(frame["irradiance_wm2"].iloc[0])


def test_records_group_into_one_batch_per_site():
    from gridguard.contract.bridge import batches_by_site

    records = [
        _record(site_id="nist_roof", asset_id="nist_roof_array_1"),
        _record(site_id="gmu_fairfax", asset_id="gmu_fairfax_array_1"),
        _record(site_id="nist_roof", asset_id="nist_roof_array_1", sequence=8),
    ]
    batches = batches_by_site(records, sent_at=EVENT_TIME)

    assert set(batches) == {"nist_roof", "gmu_fairfax"}
    assert len(batches["nist_roof"].records) == 2
    assert batches["nist_roof"].agent_id == "agent_nist_roof"


def test_a_full_batch_survives_serialisation_end_to_end():
    """Simulator → narrow records → batch → bytes → records → frame."""
    from gridguard.contract.bridge import batches_by_site, frame_from_records, records_from_frame
    from gridguard.simulation import FleetSimulator, SimulationConfig

    result = FleetSimulator(
        SimulationConfig(site_ids=("nist_roof",), start="2016-06-01", end="2016-06-01 05:45")
    ).run()
    original = result.canonical()

    batch = batches_by_site(records_from_frame(original), sent_at=EVENT_TIME)["nist_roof"]
    restored = frame_from_records(decode_batch(deserialize_batch(serialize(batch))))

    assert len(restored) == len(original)
    pd.testing.assert_series_equal(
        original["ac_power_kw"].reset_index(drop=True).astype(float),
        restored["ac_power_kw"].reset_index(drop=True).astype(float),
        check_names=False,
    )
