"""
Edge agent tests.

The gate for this phase is one sentence: **kill the broker mid-run and no sample
is lost or duplicated.** Everything else here supports establishing that.

Loss and duplication are opposite failure modes with opposite causes, and a
design can easily avoid one by committing the other:

* Buffer nothing and publish eagerly → nothing is duplicated, everything is lost
  when the link drops.
* Republish everything on every flush → nothing is lost, everything is
  duplicated.

So both are asserted together, against a transport that fails exactly where
told. A real broker cannot be made to drop the third record of a batch on
demand; this one can, which is why phase 5 is built against the interface and
phase 6 brings the client.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from gridguard.contract.codec import decode_batch, deserialize_batch
from gridguard.domain.telemetry import QualityFlag, TelemetryRecord
from gridguard.edge import (
    ClearSkyScreen,
    EdgeAgent,
    InMemoryTransport,
    OutboxBuffer,
    QualityChecker,
    TransportError,
)

START = datetime(2016, 6, 1, 12, 0, tzinfo=UTC)
SITE = "nist_roof"
ASSET = "nist_roof_array_1"


def _records(count: int, *, channel: str = "ac_power_kw", start_value: float = 10.0):
    return [
        TelemetryRecord(
            asset_id=ASSET,
            site_id=SITE,
            channel=channel,
            unit="kW",
            value=start_value + i,
            event_time=START + timedelta(minutes=15 * i),
        )
        for i in range(count)
    ]


def _agent(tmp_path, transport=None, **kwargs) -> EdgeAgent:
    return EdgeAgent(
        agent_id="agent_nist_roof",
        site_id=SITE,
        transport=transport or InMemoryTransport(),
        buffer_path=tmp_path / "outbox.sqlite",
        **kwargs,
    )


def _delivered_records(transport: InMemoryTransport) -> list[TelemetryRecord]:
    """Every record the transport actually carried, in order."""
    out: list[TelemetryRecord] = []
    for payload in transport.payloads():
        out.extend(decode_batch(deserialize_batch(payload)))
    return out


# ---------------------------------------------------------------------------
# The gate: no loss, no duplication
# ---------------------------------------------------------------------------


def test_nothing_is_lost_when_the_transport_is_down(tmp_path):
    transport = InMemoryTransport(online=False)
    with _agent(tmp_path, transport) as agent:
        agent.publish(_records(20))

        assert transport.payloads() == [], "nothing should have been delivered"
        assert agent.buffered == 20, "everything should still be buffered"

        transport.connect()
        acknowledged = agent.flush()

        assert acknowledged == 20
        assert agent.buffered == 0
        assert len(_delivered_records(transport)) == 20


def test_nothing_is_duplicated_across_a_recovery(tmp_path):
    """The other half of the gate, and the one an eager retry would break."""
    transport = InMemoryTransport()
    with _agent(tmp_path, transport, batch_size=5) as agent:
        agent.publish(_records(10))  # delivered
        transport.disconnect()
        agent.publish(_records(10)[:0] + _records(20)[10:])  # buffered only
        transport.connect()
        agent.flush()

        delivered = _delivered_records(transport)
        keys = [(r.asset_id, r.sequence) for r in delivered]

        assert len(keys) == len(set(keys)), f"duplicate records delivered: {keys}"
        assert len(keys) == 20


def test_a_mid_batch_failure_loses_nothing_and_duplicates_nothing(tmp_path):
    """Cut the link after two successful sends, then recover.

    Off-by-one acknowledgement bugs live exactly here: a batch that was sent but
    not confirmed, or confirmed but not marked.
    """
    transport = InMemoryTransport(fail_after=2)
    with _agent(tmp_path, transport, batch_size=10) as agent:
        agent.publish(_records(50))

        assert agent.buffered == 30, "the unsent remainder must stay buffered"
        assert agent.stats.publish_failures == 1

        transport.reset_failures()
        agent.flush()

        delivered = _delivered_records(transport)
        keys = [(r.asset_id, r.sequence) for r in delivered]
        assert sorted(k[1] for k in keys) == list(range(50))
        assert len(keys) == len(set(keys))


def test_records_survive_a_process_restart(tmp_path):
    """An agent killed mid-outage comes back holding everything."""
    path = tmp_path / "outbox.sqlite"
    transport = InMemoryTransport(online=False)

    first = EdgeAgent(
        agent_id="agent_nist_roof", site_id=SITE, transport=transport, buffer_path=path
    )
    first.publish(_records(15))
    assert first.buffered == 15
    first.close()  # simulate the process dying

    revived = EdgeAgent(
        agent_id="agent_nist_roof", site_id=SITE, transport=transport, buffer_path=path
    )
    assert revived.buffered == 15, "the buffer did not survive the restart"

    transport.connect()
    assert revived.flush() == 15
    assert len(_delivered_records(transport)) == 15
    revived.close()


def test_sequence_numbers_do_not_restart_after_a_crash(tmp_path):
    """Renumbering would make (asset_id, sequence) useless as an idempotency key."""
    path = tmp_path / "outbox.sqlite"

    first = EdgeAgent(agent_id="a", site_id=SITE, transport=InMemoryTransport(), buffer_path=path)
    first.publish(_records(10))
    first.close()

    revived = EdgeAgent(agent_id="a", site_id=SITE, transport=InMemoryTransport(), buffer_path=path)
    assert revived.buffer.peek_sequence(ASSET, "ac_power_kw") == 10
    revived.ingest(_records(3))
    sequences = [r.sequence for r in revived.buffer.pending()]
    assert sequences == [10, 11, 12]
    revived.close()


def test_re_ingesting_the_same_records_is_a_no_op(tmp_path):
    """An agent restarted mid-batch can be handed the same input again."""
    transport = InMemoryTransport(online=False)
    with _agent(tmp_path, transport) as agent:
        batch = _records(10)
        assert agent.ingest(batch) == 10

        # Same records, now carrying the sequences the agent assigned.
        numbered = [
            TelemetryRecord(
                asset_id=r.asset_id,
                site_id=r.site_id,
                channel=r.channel,
                unit=r.unit,
                value=r.value,
                event_time=r.event_time,
                sequence=i,
            )
            for i, r in enumerate(batch)
        ]
        assert agent.ingest(numbered) == 0
        assert agent.buffered == 10


def test_a_replayed_record_is_byte_identical_to_its_first_attempt(tmp_path):
    """The buffer stores the wire form, so buffering cannot alter a message."""
    transport = InMemoryTransport(online=False)
    with _agent(tmp_path, transport) as agent:
        agent.ingest(_records(3))
        stored = [item.payload for item in agent.buffer.pending()]

        transport.connect()
        agent.flush()

        delivered = _delivered_records(transport)
        from gridguard.contract.codec import encode_record, serialize

        assert [serialize(encode_record(r)) for r in delivered] == stored


# ---------------------------------------------------------------------------
# Replay signalling
# ---------------------------------------------------------------------------


def test_a_recovery_batch_is_marked_as_a_replay(tmp_path):
    """A burst of old event_times is a recovery, not a clock problem.

    The collector should not have to infer that from timestamps.
    """
    transport = InMemoryTransport(online=False)
    with _agent(tmp_path, transport) as agent:
        agent.publish(_records(5))
        transport.connect()
        agent.flush()

        batches = [deserialize_batch(p) for p in transport.payloads()]
        assert batches[0].is_replay is True
        assert agent.stats.replays == 1


def test_a_live_batch_is_not_marked_as_a_replay(tmp_path):
    transport = InMemoryTransport()
    with _agent(tmp_path, transport) as agent:
        agent.publish(_records(5))
        batch = deserialize_batch(transport.payloads()[0])
        assert batch.is_replay is False


def test_batches_report_how_far_behind_the_agent_is(tmp_path):
    """A rising backlog is the signal a node is failing, not merely quiet."""
    transport = InMemoryTransport()
    with _agent(tmp_path, transport, batch_size=10) as agent:
        agent.publish(_records(25))
        remaining = [deserialize_batch(p).buffered_remaining for p in transport.payloads()]
        assert remaining == [15, 5, 0]


def test_status_reports_degraded_while_buffering(tmp_path):
    from gridguard.contract.v1 import telemetry_pb2 as pb

    transport = InMemoryTransport(online=False)
    with _agent(tmp_path, transport) as agent:
        assert agent.status().state == pb.NODE_STATE_ONLINE

        agent.publish(_records(4))
        status = agent.status()
        assert status.state == pb.NODE_STATE_DEGRADED
        assert status.buffered_records == 4

        transport.connect()
        agent.flush()
        assert agent.status().state == pb.NODE_STATE_ONLINE


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


def test_buffer_refuses_duplicate_keys(tmp_path):
    with OutboxBuffer(tmp_path / "b.sqlite") as buffer:
        kw = {"asset_id": ASSET, "channel": "ac_power_kw", "event_time": "t"}
        assert buffer.append(sequence=0, payload=b"x", **kw)
        assert not buffer.append(sequence=0, payload=b"y", **kw)
        assert buffer.total_count() == 1

        # Same sequence on a different channel is a different record, not a dupe.
        assert buffer.append(
            asset_id=ASSET, channel="irradiance_wm2", sequence=0, event_time="t", payload=b"z"
        )
        assert buffer.total_count() == 2


def test_buffer_only_prunes_delivered_records(tmp_path):
    """Pending records are never the safe thing to drop."""
    with OutboxBuffer(tmp_path / "b.sqlite") as buffer:
        for i in range(10):
            buffer.append(
                asset_id=ASSET, channel="ac_power_kw", sequence=i, event_time="t", payload=b"x"
            )
        buffer.acknowledge([1, 2, 3])

        buffer.prune_acknowledged(keep_last=0)
        assert buffer.pending_count() == 7
        assert buffer.total_count() == 7


def test_sequences_are_independent_per_asset(tmp_path):
    with OutboxBuffer(tmp_path / "b.sqlite") as buffer:
        assert buffer.next_sequence("a_array_1", "ac_power_kw") == 0
        assert buffer.next_sequence("b_array_1", "ac_power_kw") == 0
        assert buffer.next_sequence("a_array_1", "ac_power_kw") == 1
        assert buffer.peek_sequence("b_array_1", "ac_power_kw") == 1

        # Channels of one asset are independent streams with their own counters.
        assert buffer.next_sequence("a_array_1", "irradiance_wm2") == 0


def test_acknowledge_is_idempotent(tmp_path):
    with OutboxBuffer(tmp_path / "b.sqlite") as buffer:
        buffer.append(
            asset_id=ASSET, channel="ac_power_kw", sequence=0, event_time="t", payload=b"x"
        )
        ids = [item.id for item in buffer.pending()]
        assert buffer.acknowledge(ids) == 1
        assert buffer.acknowledge(ids) == 0


# ---------------------------------------------------------------------------
# Local quality checks
# ---------------------------------------------------------------------------


def test_out_of_range_readings_are_flagged():
    checker = QualityChecker()
    record = TelemetryRecord(
        asset_id=ASSET,
        site_id=SITE,
        channel="irradiance_wm2",
        unit="W/m2",
        value=9999.0,
        event_time=START,
    )
    assert checker.check(record).quality & QualityFlag.OUT_OF_RANGE


def test_a_frozen_channel_is_flagged_in_daylight():
    checker = QualityChecker(frozen_threshold=3)

    # Establish daylight for this site first.
    checker.check(
        TelemetryRecord(
            asset_id="nist_roof_pyranometer_1",
            site_id=SITE,
            channel="irradiance_wm2",
            unit="W/m2",
            value=600.0,
            event_time=START,
        )
    )

    flags = []
    for i in range(6):
        record = TelemetryRecord(
            asset_id=ASSET,
            site_id=SITE,
            channel="ac_power_kw",
            unit="kW",
            value=25.0,
            event_time=START + timedelta(minutes=15 * i),
        )
        flags.append(checker.check(record).quality)

    assert not (flags[0] & QualityFlag.FROZEN)
    assert flags[-1] & QualityFlag.FROZEN


def test_a_dark_zero_power_channel_is_not_called_frozen():
    """Every night would otherwise produce a fleet-wide burst of FROZEN.

    An operator who sees the flag every night learns to ignore it, which costs
    more than the flag is worth.
    """
    checker = QualityChecker(frozen_threshold=3)
    checker.check(
        TelemetryRecord(
            asset_id="nist_roof_pyranometer_1",
            site_id=SITE,
            channel="irradiance_wm2",
            unit="W/m2",
            value=0.0,
            event_time=START,
        )
    )

    flags = []
    for i in range(10):
        flags.append(
            checker.check(
                TelemetryRecord(
                    asset_id=ASSET,
                    site_id=SITE,
                    channel="ac_power_kw",
                    unit="kW",
                    value=0.0,
                    event_time=START + timedelta(minutes=15 * i),
                )
            ).quality
        )

    assert not any(f & QualityFlag.FROZEN for f in flags)


def test_checks_never_clear_an_existing_flag():
    """A flag set upstream reflects something upstream observed."""
    checker = QualityChecker()
    record = TelemetryRecord(
        asset_id=ASSET,
        site_id=SITE,
        channel="ac_power_kw",
        unit="kW",
        value=25.0,
        event_time=START,
        quality=QualityFlag.INTERPOLATED,
    )
    assert checker.check(record).quality & QualityFlag.INTERPOLATED


def test_a_missing_value_is_flagged_missing():
    checker = QualityChecker()
    record = TelemetryRecord(
        asset_id=ASSET,
        site_id=SITE,
        channel="ac_power_kw",
        unit="kW",
        value=None,
        event_time=START,
        quality=QualityFlag.MISSING,
    )
    assert checker.check(record).quality & QualityFlag.MISSING


# ---------------------------------------------------------------------------
# Local clear-sky screening
# ---------------------------------------------------------------------------


@pytest.fixture
def screen():
    from gridguard.sites.registry import get_site

    return ClearSkyScreen(get_site(SITE))


def test_the_screen_needs_no_network_and_no_calibration(screen):
    """The whole point of D4: it still works on an isolated node."""
    times = pd.date_range("2016-06-01 12:00", periods=8, freq="15min", tz=UTC)
    ceiling = screen.ceiling_kw(times)
    assert (ceiling > 0).any()


def test_the_screen_flags_a_dead_array_in_bright_sun(screen):
    times = pd.date_range("2016-06-01 16:00", periods=8, freq="15min", tz=UTC)
    dead = np.zeros(len(times))
    results = screen.screen(times, dead, asset_id=ASSET)

    assert results, "midday should produce screenable intervals"
    assert all(r.suspicious for r in results)
    assert "not a detection" in results[0].reason


def test_the_screen_stays_quiet_on_healthy_output(screen):
    times = pd.date_range("2016-06-01 16:00", periods=8, freq="15min", tz=UTC)
    healthy = screen.ceiling_kw(times) * 0.85
    results = screen.screen(times, healthy, asset_id=ASSET)

    assert results
    assert not any(r.suspicious for r in results)


def test_the_screen_says_nothing_at_night(screen):
    """No ceiling means no conclusion — silence, not a false alarm."""
    times = pd.date_range("2016-06-01 05:00", periods=8, freq="15min", tz=UTC)
    results = screen.screen(times, np.zeros(len(times)), asset_id=ASSET)
    assert results == []


def test_the_screen_misses_moderate_underperformance(screen):
    """Its weakness is measured, not hidden.

    A 50% derate is a serious fault and the local screen sails past it, because
    a clear-sky ceiling is an upper bound that ordinary weather sits far below.
    This is the honest reach of a check with no weather input and no calibrated
    bound — and the reason detection proper stays in the core.
    """
    times = pd.date_range("2016-06-01 16:00", periods=8, freq="15min", tz=UTC)
    halved = screen.ceiling_kw(times) * 0.5
    results = screen.screen(times, halved, asset_id=ASSET)

    assert results
    assert not any(r.suspicious for r in results), (
        "if this starts firing, the threshold has been raised into the range "
        "where ordinary overcast weather also fires"
    )


# ---------------------------------------------------------------------------
# Transport contract
# ---------------------------------------------------------------------------


def test_the_fake_transport_satisfies_the_protocol():
    from gridguard.edge import Transport as TransportProtocol

    assert isinstance(InMemoryTransport(), TransportProtocol)


def test_a_transport_that_raises_leaves_the_buffer_intact(tmp_path):
    class AlwaysFails:
        connected = False

        def connect(self): ...
        def disconnect(self): ...
        def publish(self, topic, payload):
            raise TransportError("no")

    with _agent(tmp_path, AlwaysFails()) as agent:
        agent.publish(_records(7))
        assert agent.buffered == 7
        assert agent.stats.acknowledged == 0


# ---------------------------------------------------------------------------
# End to end with the simulator
# ---------------------------------------------------------------------------


def test_a_simulated_day_survives_an_outage_intact(tmp_path):
    """The gate, against real simulator output rather than a synthetic list."""
    from gridguard.contract.bridge import records_from_frame
    from gridguard.simulation import FleetSimulator, SimulationConfig

    result = FleetSimulator(
        SimulationConfig(site_ids=(SITE,), start="2016-06-01", end="2016-06-01 23:45")
    ).run()
    records = records_from_frame(result.canonical())

    transport = InMemoryTransport()
    with _agent(tmp_path, transport, batch_size=64) as agent:
        # Deliver the first third, then lose the link entirely.
        third = len(records) // 3
        agent.publish(records[:third])
        transport.disconnect()
        agent.publish(records[third : 2 * third])
        agent.publish(records[2 * third :])
        assert agent.buffered == len(records) - third

        transport.connect()
        agent.flush()
        assert agent.buffered == 0

    delivered = _delivered_records(transport)
    keys = [(r.asset_id, r.channel, r.sequence) for r in delivered]

    assert len(delivered) == len(records), "records were lost"
    assert len(keys) == len(set(keys)), "records were duplicated"
