"""
Simulator tests.

Four properties here are load-bearing for every experiment built on this
simulator, and each is asserted directly rather than assumed:

1. **Determinism.** Same config and seed, same bytes — across processes.
2. **Speed independence.** Pacing changes when ticks arrive, never what they
   contain.
3. **Layer independence.** Changing the sensor or transport configuration
   cannot perturb ground truth. Without this, a fault-attribution experiment
   cannot distinguish "the detector responded to the injected sensor fault"
   from "the weather changed because we injected a sensor fault".
4. **Truth and observation stay apart.** They are close but never identical,
   and both are retained.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from gridguard.simulation import (
    EquipmentConfig,
    EquipmentState,
    FleetSimulator,
    LogicalClock,
    SensorConfig,
    SimulationConfig,
    TransportConfig,
    simulate,
    stream_for,
)

SITES = ("nist_roof", "gmu_fairfax")
WINDOW = {"start": "2016-06-01", "end": "2016-06-04"}


def _config(**overrides) -> SimulationConfig:
    kwargs = {"site_ids": SITES, **WINDOW, "seed": 7}
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Logical clock
# ---------------------------------------------------------------------------


def test_clock_requires_timezone_aware_start():
    with pytest.raises(ValueError, match="timezone-aware"):
        LogicalClock(start=datetime(2016, 6, 1), periods=4)


def test_clock_generates_rather_than_reads_time():
    """Two clocks built a moment apart describe the same instants."""
    a = LogicalClock.from_range("2016-06-01", "2016-06-02")
    b = LogicalClock.from_range("2016-06-01", "2016-06-02")
    assert a.start == b.start
    assert len(a) == len(b)
    assert [t.event_time for t in a] == [t.event_time for t in b]


def test_clock_at_matches_iteration():
    clock = LogicalClock.from_range("2016-06-01", "2016-06-01 06:00")
    assert [clock.at(i) for i in range(len(clock))] == list(clock)


def test_clock_indexes_are_utc():
    index = LogicalClock.from_range("2016-06-01", "2016-06-02").index()
    assert str(index.tz) == "UTC"


def test_clock_rejects_reversed_range():
    with pytest.raises(ValueError, match="precedes"):
        LogicalClock.from_range("2016-06-02", "2016-06-01")


def test_clock_out_of_range_tick_is_an_error():
    clock = LogicalClock.from_range("2016-06-01", "2016-06-01 01:00")
    with pytest.raises(IndexError):
        clock.at(len(clock))


def test_speedup_does_not_change_the_instants():
    """Pacing affects delivery, never content."""
    unpaced = LogicalClock.from_range("2016-06-01", "2016-06-01 02:00")
    paced = unpaced.replace(speedup=1_000_000.0)
    assert [t.event_time for t in unpaced] == [t.event_time for t in paced]
    assert [t.index for t in unpaced] == [t.index for t in paced]


def test_simulation_output_is_identical_at_any_speed():
    """The headline guarantee for demos: watching it does not change it."""
    fast = FleetSimulator(_config(site_ids=("nist_roof",))).run()
    paced = FleetSimulator(_config(site_ids=("nist_roof",), speedup=1_000_000.0)).run()
    pd.testing.assert_frame_equal(fast.canonical(), paced.canonical())
    assert fast.fingerprint == paced.fingerprint, "speedup must not enter the fingerprint"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_config_and_seed_produce_identical_output():
    a = FleetSimulator(_config()).run()
    b = FleetSimulator(_config()).run()
    pd.testing.assert_frame_equal(a.canonical(), b.canonical())
    pd.testing.assert_frame_equal(a.truth(), b.truth())


def test_different_seeds_produce_different_weather():
    a = FleetSimulator(_config(seed=1)).run()
    b = FleetSimulator(_config(seed=2)).run()
    assert not np.allclose(
        a.sites["nist_roof"].environment.poa_irradiance_wm2,
        b.sites["nist_roof"].environment.poa_irradiance_wm2,
    )


def test_sites_get_independent_weather():
    result = FleetSimulator(_config()).run()
    roof = result.sites["nist_roof"].environment.cloud_transmittance
    gmu = result.sites["gmu_fairfax"].environment.cloud_transmittance
    assert not np.allclose(roof, gmu), "every site received the same cloud sequence"


def test_determinism_survives_the_process_hash_salt():
    """A run must not depend on PYTHONHASHSEED."""
    import subprocess
    import sys

    code = (
        "import hashlib;"
        "from gridguard.simulation import simulate;"
        "r = simulate(['nist_roof'], start='2016-06-01', end='2016-06-02', seed=7);"
        "print(hashlib.blake2b("
        "  r.canonical().to_csv(index=False).encode(), digest_size=8).hexdigest())"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "9999")
    }
    assert len(digests) == 1, f"simulation varied with PYTHONHASHSEED: {digests}"


def test_fingerprint_changes_with_configuration():
    assert _config().fingerprint() != _config(seed=99).fingerprint()
    assert _config().fingerprint() != _config(end="2016-06-05").fingerprint()


def test_named_streams_are_independent():
    """Adding a stream must not perturb an existing one."""
    a = stream_for(7, "site", "environment", "cloud").normal(size=50)
    b = stream_for(7, "site", "environment", "cloud").normal(size=50)
    c = stream_for(7, "site", "sensor", "noise").normal(size=50)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# ---------------------------------------------------------------------------
# Layer independence — the property the whole architecture exists for
# ---------------------------------------------------------------------------


def test_sensor_configuration_cannot_change_ground_truth():
    """The instruments do not affect the sun.

    If this fails, every fault-attribution experiment built on this simulator is
    invalid: a sensor fault would come bundled with different weather, and no
    detector response could be attributed to either.
    """
    baseline = FleetSimulator(_config()).run()
    noisy = FleetSimulator(
        _config(
            sensors=SensorConfig(
                irradiance_calibration_error=0.5,
                irradiance_noise=0.4,
                power_noise=0.3,
                temperature_noise_c=5.0,
                power_quantum_kw=1.0,
            )
        )
    ).run()

    for site_id in SITES:
        np.testing.assert_array_equal(
            baseline.sites[site_id].environment.poa_irradiance_wm2,
            noisy.sites[site_id].environment.poa_irradiance_wm2,
        )
        np.testing.assert_array_equal(
            baseline.sites[site_id].equipment.site_actual_kw,
            noisy.sites[site_id].equipment.site_actual_kw,
        )


def test_sensor_configuration_does_change_observations():
    """The counterpart: the guarantee above is not vacuous."""
    baseline = FleetSimulator(_config()).run()
    noisy = FleetSimulator(_config(sensors=SensorConfig(irradiance_noise=0.4))).run()
    assert not np.allclose(
        baseline.sites["nist_roof"].observation.observed_irradiance_wm2,
        noisy.sites["nist_roof"].observation.observed_irradiance_wm2,
    )


def test_equipment_configuration_cannot_change_the_weather():
    baseline = FleetSimulator(_config()).run()
    derated = FleetSimulator(
        _config(equipment=EquipmentConfig(system_derate=0.5, performance_drift=False))
    ).run()
    np.testing.assert_array_equal(
        baseline.sites["nist_roof"].environment.poa_irradiance_wm2,
        derated.sites["nist_roof"].environment.poa_irradiance_wm2,
    )
    assert (
        derated.sites["nist_roof"].equipment.site_actual_kw.sum()
        < baseline.sites["nist_roof"].equipment.site_actual_kw.sum()
    )


# ---------------------------------------------------------------------------
# Truth versus observation
# ---------------------------------------------------------------------------


def test_truth_and_observation_are_both_retained_and_differ():
    result = FleetSimulator(_config()).run()
    sim = result.sites["nist_roof"]

    truth = sim.equipment.site_actual_kw
    observed = sim.observation.observed_ac_power_kw

    assert not np.array_equal(truth, observed), "observation is implausibly perfect"
    # Close, though: sensor error is small relative to the signal.
    daylight = truth > 1.0
    relative = abs(observed[daylight] - truth[daylight]) / truth[daylight]
    assert relative.mean() < 0.1


def test_observation_error_is_measurable():
    """Only computable because the layers are kept apart."""
    errors = FleetSimulator(_config()).run().observation_error()
    assert set(errors["site_id"]) == set(SITES)
    assert (errors["power_mae_kw"] > 0).all()
    assert (errors["delivery_ratio"] == 1.0).all()


def test_ground_truth_records_equipment_state_not_just_power():
    """State is recorded, not inferred from the power it produced."""
    truth = FleetSimulator(_config()).run().truth()
    assert "equipment_state" in truth.columns
    assert set(truth["equipment_state"]) == {EquipmentState.HEALTHY.value}


def test_truth_carries_potential_alongside_actual():
    """Lost generation needs a counterfactual, and it must be recorded."""
    truth = FleetSimulator(_config()).run().truth()
    assert {"potential_ac_power_kw", "actual_ac_power_kw", "lost_ac_power_kw"} <= set(truth.columns)
    assert (truth["lost_ac_power_kw"] >= 0).all()


def test_per_asset_generation_sums_to_the_site():
    result = FleetSimulator(_config(site_ids=("dc_community",))).run()
    sim = result.sites["dc_community"]
    assert len(sim.equipment.assets) == 3, "dc_community has three arrays"
    np.testing.assert_allclose(
        sim.equipment.site_actual_kw,
        np.sum([a.actual_ac_power_kw for a in sim.equipment.assets], axis=0),
    )


def test_observations_are_attributed_to_instruments():
    """A reading belongs to a component, so a sensor fault has an owner."""
    sim = FleetSimulator(_config()).run().sites["nist_roof"]
    assert sim.observation.irradiance_asset_id == "nist_roof_pyranometer_1"
    assert sim.observation.power_asset_id.startswith("nist_roof")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_pass_through_delivers_everything_with_zero_lag():
    delivery = FleetSimulator(_config()).run().sites["nist_roof"].delivery
    assert delivery.delivered.all()
    assert (delivery.arrival_lag_seconds == 0).all()
    assert delivery.delivery_ratio == 1.0


def test_sequence_numbers_are_monotonic():
    delivery = FleetSimulator(_config()).run().sites["nist_roof"].delivery
    assert (np.diff(delivery.sequence) == 1).all()


def test_lossy_transport_refuses_rather_than_silently_passing_through():
    """Returning lossless output for a lossy request would be read as a result."""
    with pytest.raises(NotImplementedError, match="phase 5"):
        FleetSimulator(_config(transport=TransportConfig(loss_rate=0.2))).run()


def test_transport_config_knows_when_it_is_a_pass_through():
    assert TransportConfig().is_pass_through
    assert not TransportConfig(mean_delay_seconds=1.0).is_pass_through
    assert not TransportConfig(reorder=True).is_pass_through


# ---------------------------------------------------------------------------
# Bridge to the existing pipeline
# ---------------------------------------------------------------------------


def test_canonical_output_satisfies_the_shipped_schema():
    from gridguard.data.schema import CANONICAL_COLUMNS

    frame = FleetSimulator(_config()).run().canonical()
    for column in CANONICAL_COLUMNS:
        assert column in frame.columns
    assert (frame["data_mode"] == "synthetic").all()
    assert frame["timestamp"].is_monotonic_increasing


def test_canonical_output_feeds_the_existing_feature_pipeline():
    """The whole point of the bridge: nothing downstream needs to change."""
    from gridguard.features.engineer import get_X_y

    frame = FleetSimulator(_config(site_ids=("nist_roof",))).run().canonical()
    X, y = get_X_y(frame, mode="weather_only")
    assert len(X) == len(y) == len(frame)
    assert not X.isna().all().any()


def test_canonical_output_carries_both_time_representations():
    frame = FleetSimulator(_config(site_ids=("nist_roof",))).run().canonical()
    assert str(frame["event_time"].dt.tz) == "UTC"
    assert frame["timestamp"].dt.tz is None

    from gridguard.data.synthetic import UTC_OFFSET_HOURS
    from gridguard.domain.telemetry import event_time_to_local

    pd.testing.assert_series_equal(
        frame["timestamp"],
        event_time_to_local(frame["event_time"], UTC_OFFSET_HOURS),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# No wall-clock anywhere in the simulation
# ---------------------------------------------------------------------------


def test_no_layer_reads_the_wall_clock():
    """Simulated time is generated, never read.

    ``clock.py`` is exempt for one narrowly-scoped use: pacing a demo against
    the wall clock, which affects when ticks are handed out and never what they
    contain. Everywhere else, a wall-clock read is a reproducibility bug.
    """
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parent.parent / "src" / "gridguard" / "simulation"
    banned = {"now", "utcnow", "today", "time", "monotonic", "perf_counter"}
    offenders = []

    for path in sorted(package.glob("*.py")):
        if path.name == "clock.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in banned:
                    offenders.append(f"{path.name}:{node.lineno} {node.func.attr}()")

    assert not offenders, f"wall-clock reads outside clock.py: {offenders}"


def test_simulation_is_a_pure_function_of_config_and_seed():
    """No hidden inputs: two runs separated by other work still agree."""
    first = simulate(["nist_roof"], start="2016-06-01", end="2016-06-02", seed=3)
    _ = simulate(["gmu_fairfax"], start="2016-01-01", end="2016-01-05", seed=99)
    second = simulate(["nist_roof"], start="2016-06-01", end="2016-06-02", seed=3)
    pd.testing.assert_frame_equal(first.canonical(), second.canonical())


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_yields_every_tick_in_order():
    result = FleetSimulator(_config(site_ids=("nist_roof",))).run()
    ticks = [tick for tick, _ in result.replay()]
    assert len(ticks) == len(result.clock)
    assert [t.index for t in ticks] == list(range(len(result.clock)))


def test_replay_rows_reconstruct_the_canonical_frame():
    """Replay hands out the same data, not a recomputation of it."""
    result = FleetSimulator(_config()).run()
    chunks = [rows for _, rows in result.replay() if len(rows)]
    rebuilt = pd.concat(chunks, ignore_index=True).sort_values(
        ["timestamp", "site_id"], ignore_index=True
    )
    original = result.canonical().sort_values(["timestamp", "site_id"], ignore_index=True)
    pd.testing.assert_frame_equal(rebuilt, original)


def test_replay_pacing_does_not_change_the_data():
    """The values exist before pacing starts, so pacing cannot alter them."""
    unpaced = FleetSimulator(_config(site_ids=("nist_roof",))).run()
    paced = FleetSimulator(_config(site_ids=("nist_roof",), speedup=500_000.0)).run()

    a = pd.concat([r for _, r in unpaced.replay() if len(r)], ignore_index=True)
    b = pd.concat([r for _, r in paced.replay() if len(r)], ignore_index=True)
    pd.testing.assert_frame_equal(a, b)


def test_replay_actually_paces_when_asked():
    """A speedup that implies a measurable wall-clock cost must incur it.

    Guards the gap this feature was first shipped with: ``speedup`` was
    threaded through the config but the vectorised runner never iterated the
    clock, so pacing silently did nothing.
    """
    import time

    # 8 ticks x 15 simulated minutes = 7200 simulated seconds. At 20000x that
    # is ~0.36s of wall clock — long enough to measure, short enough for CI.
    config = _config(
        site_ids=("nist_roof",),
        start="2016-06-01",
        end="2016-06-01 01:45",
        speedup=20_000.0,
    )
    result = FleetSimulator(config).run()

    started = time.monotonic()
    for _ in result.replay():
        pass
    elapsed = time.monotonic() - started

    expected = result.clock.simulated_duration.total_seconds() / 20_000.0
    assert elapsed >= expected * 0.5, (
        f"replay finished in {elapsed:.3f}s but pacing implies at least "
        f"~{expected:.3f}s — speedup is not being honoured"
    )
