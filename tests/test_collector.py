"""
Collector and store tests.

The gate for this phase: **replaying a buffer produces identical table state.**
An agent recovering from an outage resends records the collector may already
hold, and that has to be a no-op rather than a double count.

Backends
--------
Every store test runs against both SQLite and Postgres. SQLite always; Postgres
only when ``GRIDGUARD_TEST_POSTGRES_DSN`` names a reachable database, which it
does in the compose integration profile and does not on a laptop. The same test
bodies run on both, so the shared SQL dialect is checked rather than assumed —
and when Postgres is absent the skip is visible rather than silent.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from gridguard.collector import (
    Collector,
    TelemetryStore,
    agent_health,
    arrival_lag,
    completeness,
    from_epoch_micros,
    sqlite_store,
    to_epoch_micros,
)
from gridguard.contract.codec import encode_batch, serialize
from gridguard.domain.telemetry import QualityFlag, TelemetryRecord

START = datetime(2016, 6, 1, 12, 0, tzinfo=UTC)
SITE = "nist_roof"
ASSET = "nist_roof_array_1"
AGENT = "agent_nist_roof"

POSTGRES_DSN = os.environ.get("GRIDGUARD_TEST_POSTGRES_DSN")


# ---------------------------------------------------------------------------
# Backend fixtures — every store test runs against both
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(
                not POSTGRES_DSN,
                reason="set GRIDGUARD_TEST_POSTGRES_DSN to run against Postgres",
            ),
        ),
    ]
)
def store(request) -> TelemetryStore:
    if request.param == "sqlite":
        backing = sqlite_store(":memory:")
        yield backing
        backing.close()
        return

    import psycopg

    connection = psycopg.connect(POSTGRES_DSN)
    backing = TelemetryStore(connection, paramstyle="%s")
    cursor = connection.cursor()
    cursor.execute("TRUNCATE telemetry, agent_state")
    connection.commit()
    yield backing
    backing.close()


def _records(count: int, *, channel: str = "ac_power_kw", start_value: float = 10.0):
    return [
        TelemetryRecord(
            asset_id=ASSET,
            site_id=SITE,
            channel=channel,
            unit="kW",
            value=start_value + i,
            event_time=START + timedelta(minutes=15 * i),
            sequence=i,
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# The gate: replay is a no-op
# ---------------------------------------------------------------------------


def test_replaying_a_batch_leaves_the_table_identical(store):
    """The phase gate, stated directly."""
    records = _records(50)
    collector = Collector(store)

    first = collector.handle_batch(records, agent_id=AGENT, site_id=SITE)
    fingerprint = store.fingerprint()
    count = store.count()

    replay = collector.handle_batch(records, agent_id=AGENT, site_id=SITE, is_replay=True)

    assert first.inserted == 50
    assert replay.inserted == 0, "a replay must insert nothing"
    assert replay.duplicates == 50
    assert replay.was_entirely_duplicate
    assert store.count() == count
    assert store.fingerprint() == fingerprint, "table state changed under a replay"


def test_replaying_ten_times_changes_nothing(store):
    records = _records(20)
    collector = Collector(store)
    collector.handle_batch(records, agent_id=AGENT, site_id=SITE)
    fingerprint = store.fingerprint()

    for _ in range(10):
        collector.handle_batch(records, agent_id=AGENT, site_id=SITE, is_replay=True)

    assert store.count() == 20
    assert store.fingerprint() == fingerprint


def test_a_partial_replay_inserts_only_the_new_records(store):
    """The realistic recovery: some of what an agent resends is already held."""
    collector = Collector(store)
    collector.handle_batch(_records(30), agent_id=AGENT, site_id=SITE)

    overlapping = _records(50)  # 0..29 already held, 30..49 new
    outcome = collector.handle_batch(overlapping, agent_id=AGENT, site_id=SITE, is_replay=True)

    assert outcome.inserted == 20
    assert outcome.duplicates == 30
    assert store.count() == 50


def test_out_of_order_arrival_produces_the_same_state(store):
    """Order of arrival must not affect what is stored.

    A recovering agent delivers a burst whose order the network does not
    guarantee, so state has to be a function of the set, not the sequence.
    """
    records = _records(30)

    forward = sqlite_store(":memory:")
    Collector(forward).handle_batch(records, agent_id=AGENT, site_id=SITE)

    Collector(store).handle_batch(list(reversed(records)), agent_id=AGENT, site_id=SITE)

    assert store.fingerprint() == forward.fingerprint()
    forward.close()


def test_the_key_is_the_measurement_not_the_sequence(store):
    """An agent redeployed mid-run renumbers; the measurement is unchanged.

    Keying on sequence would make the same reading land twice under two
    different numbers.
    """
    collector = Collector(store)
    collector.handle_batch(_records(10), agent_id=AGENT, site_id=SITE)

    renumbered = [
        TelemetryRecord(
            asset_id=r.asset_id,
            site_id=r.site_id,
            channel=r.channel,
            unit=r.unit,
            value=r.value,
            event_time=r.event_time,
            sequence=r.sequence + 1000,
        )
        for r in _records(10)
    ]
    outcome = collector.handle_batch(renumbered, agent_id="agent_rebuilt", site_id=SITE)

    assert outcome.inserted == 0
    assert store.count() == 10


def test_channels_of_one_asset_do_not_collide(store):
    """The bug phase 5 found, guarded on the storage side too."""
    collector = Collector(store)
    collector.handle_batch(_records(10, channel="ac_power_kw"), agent_id=AGENT, site_id=SITE)
    collector.handle_batch(_records(10, channel="irradiance_wm2"), agent_id=AGENT, site_id=SITE)
    assert store.count() == 20


# ---------------------------------------------------------------------------
# Presence survives storage
# ---------------------------------------------------------------------------


def test_an_absent_reading_stays_absent_in_the_store(store):
    """Zero and absent must stay distinct through the database too."""
    records = [
        TelemetryRecord(
            asset_id=ASSET,
            site_id=SITE,
            channel="ac_power_kw",
            unit="kW",
            value=0.0,
            event_time=START,
            sequence=0,
        ),
        TelemetryRecord(
            asset_id=ASSET,
            site_id=SITE,
            channel="ac_power_kw",
            unit="kW",
            value=None,
            event_time=START + timedelta(minutes=15),
            sequence=1,
            quality=QualityFlag.MISSING,
        ),
    ]
    Collector(store).handle_batch(records, agent_id=AGENT, site_id=SITE)

    stored = store.read()
    assert stored[0].value == 0.0
    assert stored[1].value is None
    assert stored[1].quality & QualityFlag.MISSING


def test_quality_flags_survive_storage(store):
    record = TelemetryRecord(
        asset_id=ASSET,
        site_id=SITE,
        channel="ac_power_kw",
        unit="kW",
        value=5.0,
        event_time=START,
        sequence=0,
        quality=QualityFlag.FROZEN | QualityFlag.STALE,
    )
    Collector(store).handle_batch([record], agent_id=AGENT, site_id=SITE)

    stored = store.read()[0]
    assert stored.quality & QualityFlag.FROZEN
    assert stored.quality & QualityFlag.STALE


def test_a_replay_can_correct_quality_flags(store):
    """A re-send may carry better information about the same measurement."""
    clean = _records(1)[0]
    Collector(store).handle_batch([clean], agent_id=AGENT, site_id=SITE)
    assert store.read()[0].quality == QualityFlag.OK

    corrected = TelemetryRecord(
        asset_id=clean.asset_id,
        site_id=clean.site_id,
        channel=clean.channel,
        unit=clean.unit,
        value=clean.value,
        event_time=clean.event_time,
        sequence=clean.sequence,
        quality=QualityFlag.FROZEN,
    )
    Collector(store).handle_batch([corrected], agent_id=AGENT, site_id=SITE, is_replay=True)
    assert store.read()[0].quality & QualityFlag.FROZEN


# ---------------------------------------------------------------------------
# Arrival time
# ---------------------------------------------------------------------------


def test_ingest_time_is_stamped_by_the_collector(store):
    """An edge clock is not evidence about when someone else took delivery."""
    arrived = START + timedelta(minutes=90)
    Collector(store).handle_batch(_records(3), agent_id=AGENT, site_id=SITE, received_at=arrived)

    for record in store.read():
        assert record.ingest_time == arrived
        assert record.arrival_lag_seconds > 0


def test_a_replay_does_not_rewrite_the_original_arrival_time(store):
    """History must not be edited to make the first delivery look slower."""
    records = _records(5)
    first_arrival = START + timedelta(minutes=15)
    Collector(store).handle_batch(records, agent_id=AGENT, site_id=SITE, received_at=first_arrival)

    much_later = START + timedelta(days=2)
    Collector(store).handle_batch(
        records, agent_id=AGENT, site_id=SITE, is_replay=True, received_at=much_later
    )

    assert {r.ingest_time for r in store.read()} == {first_arrival}


def test_arrival_lag_is_measured_not_inferred(store):
    arrived = START + timedelta(hours=6)
    Collector(store).handle_batch(_records(10), agent_id=AGENT, site_id=SITE, received_at=arrived)

    summary = arrival_lag(store)
    assert summary is not None
    assert summary.n == 10
    assert summary.median_seconds > 3600
    assert not summary.is_live


def test_live_delivery_reads_as_live(store):
    """A record arriving just after its interval closes is live.

    It cannot arrive sooner: a measurement covering 12:00-12:15 does not exist
    until 12:15. A threshold of one interval would call even instantaneous
    delivery late, which this test caught.
    """
    arrived = START + timedelta(minutes=15, seconds=30)
    Collector(store).handle_batch(_records(1), agent_id=AGENT, site_id=SITE, received_at=arrived)
    summary = arrival_lag(store)
    assert summary.median_seconds == pytest.approx(930.0)
    assert summary.is_live


def test_delivery_an_hour_late_does_not_read_as_live(store):
    """The counterpart, so the threshold is not simply permissive."""
    Collector(store).handle_batch(
        _records(1), agent_id=AGENT, site_id=SITE, received_at=START + timedelta(hours=1)
    )
    assert not arrival_lag(store).is_live


def test_epoch_micros_round_trip_exactly():
    for moment in (START, datetime(2016, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)):
        assert from_epoch_micros(to_epoch_micros(moment)) == moment


def test_storing_a_naive_datetime_is_refused():
    with pytest.raises(ValueError, match="naive datetime"):
        to_epoch_micros(datetime(2016, 6, 1, 12, 0))


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_completeness_measures_against_what_should_exist(store):
    """Counting only what arrived would make a silent site look perfect."""
    Collector(store).handle_batch(_records(10), agent_id=AGENT, site_id=SITE)

    # 10 delivered out of a 20-interval window.
    summary = completeness(
        store,
        asset_id=ASSET,
        channel="ac_power_kw",
        start=START,
        end=START + timedelta(minutes=15 * 19),
    )
    assert summary.expected == 20
    assert summary.received == 10
    assert summary.missing == 10
    assert summary.completeness == pytest.approx(0.5)


def test_a_totally_silent_asset_reads_as_zero_complete(store):
    summary = completeness(
        store,
        asset_id="never_reported_array_1",
        channel="ac_power_kw",
        start=START,
        end=START + timedelta(minutes=15 * 9),
    )
    assert summary.received == 0
    assert summary.completeness == 0.0
    assert summary.largest_gap_intervals == 10


def test_the_largest_gap_is_found(store):
    """A burst of loss matters more than the same count scattered."""
    records = _records(20)
    with_hole = records[:5] + records[15:]
    Collector(store).handle_batch(with_hole, agent_id=AGENT, site_id=SITE)

    summary = completeness(
        store,
        asset_id=ASSET,
        channel="ac_power_kw",
        start=START,
        end=START + timedelta(minutes=15 * 19),
    )
    assert summary.largest_gap_intervals == 10


# ---------------------------------------------------------------------------
# Agent health
# ---------------------------------------------------------------------------


def test_agent_state_accumulates_across_batches(store):
    collector = Collector(store)
    collector.handle_batch(_records(10), agent_id=AGENT, site_id=SITE, received_at=START)
    collector.handle_batch(
        _records(10),
        agent_id=AGENT,
        site_id=SITE,
        is_replay=True,
        received_at=START + timedelta(minutes=5),
    )

    state = store.agent_state(AGENT)
    assert state["batches_received"] == 2
    assert state["records_received"] == 20
    assert state["records_duplicate"] == 10


def test_a_silent_agent_is_visible_as_silent(store):
    Collector(store).handle_batch(_records(2), agent_id=AGENT, site_id=SITE, received_at=START)

    health = agent_health(store, AGENT, now=START + timedelta(hours=3))
    assert health.is_silent
    assert health.silent_for_seconds == pytest.approx(10800.0)


def test_a_recently_heard_agent_is_not_silent(store):
    Collector(store).handle_batch(_records(2), agent_id=AGENT, site_id=SITE, received_at=START)
    health = agent_health(store, AGENT, now=START + timedelta(minutes=10))
    assert not health.is_silent


def test_a_buffering_agent_reads_as_behind(store):
    Collector(store).handle_batch(
        _records(2), agent_id=AGENT, site_id=SITE, buffered_remaining=400, received_at=START
    )
    health = agent_health(store, AGENT, now=START)
    assert health.is_behind
    assert health.buffered_remaining == 400


def test_an_unknown_agent_has_no_health(store):
    assert agent_health(store, "never_seen", now=START) is None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_an_undecodable_payload_is_dropped_not_raised(store):
    """One malformed message must not stop the collector serving everyone else."""
    collector = Collector(store)
    assert collector.handle_payload(b"not a protobuf batch at all") is None
    assert collector.stats.decode_failures == 1

    # And the collector keeps working.
    good = serialize(encode_batch(_records(3), agent_id=AGENT, site_id=SITE, sent_at=START))
    outcome = collector.handle_payload(good, received_at=START)
    assert outcome.inserted == 3


def test_an_empty_batch_is_harmless(store):
    outcome = Collector(store).handle_batch([], agent_id=AGENT, site_id=SITE)
    assert outcome.records == 0
    assert outcome.max_arrival_lag_seconds is None
    assert not outcome.was_entirely_duplicate


# ---------------------------------------------------------------------------
# End to end: agent → wire → collector → store
# ---------------------------------------------------------------------------


def test_an_outage_and_recovery_stores_every_record_exactly_once(store, tmp_path):
    """The full path, with the link cut mid-run.

    This is the phase-5 gate and the phase-6 gate composed: the agent must not
    lose or duplicate what it sends, and the store must not double-count what it
    receives twice.
    """
    from gridguard.contract.bridge import records_from_frame
    from gridguard.edge import EdgeAgent, InMemoryTransport
    from gridguard.simulation import FleetSimulator, SimulationConfig

    result = FleetSimulator(
        SimulationConfig(site_ids=(SITE,), start="2016-06-01", end="2016-06-01 23:45")
    ).run()
    records = records_from_frame(result.canonical())

    transport = InMemoryTransport()
    with EdgeAgent(
        agent_id=AGENT,
        site_id=SITE,
        transport=transport,
        buffer_path=tmp_path / "outbox.sqlite",
        batch_size=64,
    ) as agent:
        half = len(records) // 2
        agent.publish(records[:half])
        transport.disconnect()
        agent.publish(records[half:])
        transport.connect()
        agent.flush()

    collector = Collector(store)
    for payload in transport.payloads():
        collector.handle_payload(payload, received_at=START)

    assert store.count() == len(records), "the store lost or invented records"
    assert collector.stats.decode_failures == 0

    # Now feed every payload a second time, as a broker redelivery would.
    fingerprint = store.fingerprint()
    for payload in transport.payloads():
        collector.handle_payload(payload, received_at=START + timedelta(hours=1))

    assert store.count() == len(records)
    assert store.fingerprint() == fingerprint, "redelivery changed the table"


# ---------------------------------------------------------------------------
# MQTT — interface only; see the module docstring for verification status
# ---------------------------------------------------------------------------


def test_the_mqtt_transport_satisfies_the_agent_transport_interface():
    """Conformance, not behaviour.

    The agent's failure properties were established against the in-process
    fake. This asserts the MQTT client is substitutable for it. It does not
    assert that it works against a broker — nothing here has run against one.
    """
    from gridguard.collector.mqtt import MqttTransport
    from gridguard.edge import Transport

    transport = MqttTransport(site_id=SITE, agent_id=AGENT)
    assert isinstance(transport, Transport)
    assert not transport.connected


def test_the_mqtt_transport_refuses_to_publish_while_disconnected():
    from gridguard.collector.mqtt import MqttTransport
    from gridguard.edge import TransportError

    transport = MqttTransport(site_id=SITE, agent_id=AGENT)
    with pytest.raises(TransportError, match="not connected"):
        transport.publish("t", b"payload")


@pytest.mark.skipif(
    not os.environ.get("GRIDGUARD_TEST_MQTT_HOST"),
    reason="set GRIDGUARD_TEST_MQTT_HOST to run against a real broker",
)
def test_a_round_trip_through_a_real_broker(store):
    """The integration test that would establish the MQTT client actually works.

    Skipped unless a broker is reachable. It has not been run: no Docker daemon
    was available where this was written, so Mosquitto could not be started.
    """
    import time

    from gridguard.collector.mqtt import MqttSubscriber, MqttTransport

    host = os.environ["GRIDGUARD_TEST_MQTT_HOST"]
    collector = Collector(store)

    subscriber = MqttSubscriber(on_payload=collector.handle_payload, host=host)
    subscriber.start()

    transport = MqttTransport(site_id=SITE, agent_id=AGENT, host=host)
    transport.connect()
    transport.publish(
        f"gridguard/v1/telemetry/{SITE}/{AGENT}",
        serialize(encode_batch(_records(5), agent_id=AGENT, site_id=SITE, sent_at=START)),
    )

    deadline = time.monotonic() + 10
    while store.count() < 5 and time.monotonic() < deadline:
        time.sleep(0.1)

    transport.disconnect()
    subscriber.stop()

    assert store.count() == 5
