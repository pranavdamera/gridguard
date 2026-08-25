"""
Layered fault injection.

The properties asserted here are what make the layering worth having. A fault
is owned by the layer whose behaviour it changes, and the tests below check that
ownership is real rather than nominal:

* a **sensor** fault cannot change ground truth,
* a **communication** fault cannot change ground truth *or* the observation,
* an **equipment** fault changes real generation and the loss is recorded
  exactly, as the gap between potential and actual.

Without these, the batch injector's behaviour would have been reproduced with
extra structure and no extra information.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from gridguard.data.faults import GENERATION_LOSS_FAULTS, NON_LOSS_CONDITIONS, FaultType
from gridguard.domain.fault import FaultCategory
from gridguard.simulation import (
    FAULT_LAYER,
    FaultSchedule,
    FaultSpec,
    FleetSimulator,
    SimulationConfig,
)
from gridguard.simulation.faults import (
    apply_sensor_fault,
    build_layered_scenario,
    delivery_mask,
)

SITE = "gmu_fairfax"
WINDOW = {"start": "2016-06-01", "end": "2016-09-01"}


def _config(**overrides) -> SimulationConfig:
    kwargs = {"site_ids": (SITE,), **WINDOW, "seed": 7}
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)


@pytest.fixture(scope="module")
def healthy():
    return FleetSimulator(SimulationConfig(site_ids=(SITE,), **WINDOW, seed=7)).run()


@pytest.fixture(scope="module")
def faulted():
    return FleetSimulator(
        SimulationConfig(site_ids=(SITE,), **WINDOW, seed=7, inject_faults=True)
    ).run()


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_every_fault_class_declares_an_owning_layer():
    """A new class must choose a layer rather than defaulting into one."""
    assert set(FAULT_LAYER) == set(FaultType)
    assert set(FAULT_LAYER.values()) == {"equipment", "sensing", "transport"}


def test_the_two_non_loss_conditions_land_in_different_layers():
    """The distinction the old taxonomy could not express.

    Under the batch injector both a frozen sensor and a dead link end up as a
    changed number in one column. Here they are applied by different layers,
    which is what lets them be told apart at all.
    """
    assert FAULT_LAYER[FaultType.SENSOR_DROPOUT] == "sensing"
    assert FAULT_LAYER[FaultType.COMM_DROPOUT] == "transport"


def test_scenario_covers_the_whole_taxonomy(faulted):
    schedule = faulted.schedule()
    assert {f.fault_type for f in schedule} == set(FaultType)


def test_scenario_windows_do_not_overlap(faulted):
    """Overlapping faults would make per-class scoring impossible."""
    spans = sorted(((f.start, f.end) for f in faulted.schedule()), key=lambda s: s[0])
    for (_, earlier_end), (later_start, _) in zip(spans, spans[1:], strict=False):
        assert later_start > earlier_end


def test_faults_are_attached_to_real_assets(faulted):
    from gridguard.sites.registry import assets_for_site

    known = {a.asset_id for a in assets_for_site(SITE)}
    for fault in faulted.schedule():
        assert fault.asset_id in known


def test_sensor_faults_attach_to_the_instrument_not_the_array(faulted):
    sensor_faults = faulted.schedule().for_layer("sensing")
    assert sensor_faults
    for fault in sensor_faults:
        assert "pyranometer" in fault.asset_id


# ---------------------------------------------------------------------------
# Layer isolation — the point of the whole design
# ---------------------------------------------------------------------------


def _only(fault_type: FaultType, clock_index, asset: str) -> FaultSchedule:
    """A schedule containing exactly one fault, centred in the run."""
    mid = len(clock_index) // 2
    return FaultSchedule(
        (
            FaultSpec(
                fault_type=fault_type,
                asset_id=asset,
                start=clock_index[mid].to_pydatetime(),
                end=clock_index[mid + 96].to_pydatetime(),
                severity=1.0,
            ),
        )
    )


def _run_with(schedule: FaultSchedule):
    """Run one site with an explicit schedule, bypassing the scenario builder."""
    from gridguard.simulation.environment import EnvironmentLayer
    from gridguard.simulation.equipment import EquipmentLayer
    from gridguard.simulation.sensing import SensorLayer
    from gridguard.simulation.transport import TransportLayer
    from gridguard.sites.registry import assets_for_site, get_site

    config = _config()
    clock = config.clock()
    tree = assets_for_site(SITE)

    environment = EnvironmentLayer(get_site(SITE), config).run(clock)
    equipment = EquipmentLayer(tree, config, schedule).run(environment)
    observation = SensorLayer(tree, config, schedule).run(environment, equipment)
    delivery = TransportLayer(config, schedule).run(observation)
    return environment, equipment, observation, delivery


def test_a_sensor_fault_cannot_change_ground_truth():
    """The pyranometer freezing does not stop the sun or the inverter."""
    config = _config()
    clock_index = config.clock().index()

    _, clean_equipment, _, _ = _run_with(FaultSchedule())
    environment, equipment, observation, _ = _run_with(
        _only(FaultType.SENSOR_DROPOUT, clock_index, f"{SITE}_pyranometer_1")
    )

    np.testing.assert_array_equal(clean_equipment.site_actual_kw, equipment.site_actual_kw)
    # And it does change what is reported — the guarantee is not vacuous.
    assert (observation.active_sensor_fault != "").any()


def test_a_sensor_fault_changes_the_reading_not_the_irradiance():
    config = _config()
    clock_index = config.clock().index()
    schedule = _only(FaultType.SENSOR_DROPOUT, clock_index, f"{SITE}_pyranometer_1")

    environment, _, observation, _ = _run_with(schedule)
    fault = schedule.faults[0]
    window = fault.mask(environment.event_time)

    # The instrument reports one frozen value across the window ...
    reported = observation.observed_irradiance_wm2[window]
    assert len(np.unique(np.round(reported, 6))) == 1
    # ... while the sun kept moving.
    assert environment.poa_irradiance_wm2[window].std() > 0


def test_a_communication_fault_changes_neither_truth_nor_observation():
    """The strongest isolation claim: the reading was correct, it was lost."""
    config = _config()
    clock_index = config.clock().index()

    _, clean_equipment, clean_observation, clean_delivery = _run_with(FaultSchedule())
    _, equipment, observation, delivery = _run_with(
        _only(FaultType.COMM_DROPOUT, clock_index, f"{SITE}_array_1")
    )

    np.testing.assert_array_equal(clean_equipment.site_actual_kw, equipment.site_actual_kw)
    np.testing.assert_array_equal(
        clean_observation.observed_ac_power_kw, observation.observed_ac_power_kw
    )
    np.testing.assert_array_equal(
        clean_observation.observed_irradiance_wm2, observation.observed_irradiance_wm2
    )
    # Only delivery differs.
    assert delivery.delivery_ratio < clean_delivery.delivery_ratio


def test_undelivered_records_are_flagged_missing_not_zeroed():
    """Absence must never be representable as an unflagged zero."""
    from gridguard.domain.telemetry import QualityFlag

    config = _config()
    clock_index = config.clock().index()
    schedule = _only(FaultType.COMM_DROPOUT, clock_index, f"{SITE}_array_1")
    _, _, observation, delivery = _run_with(schedule)

    lost = ~delivery.delivered
    assert lost.any()
    assert (delivery.quality[lost] & int(QualityFlag.MISSING)).all()
    # The observation behind the lost records is intact, not zeroed.
    assert observation.observed_ac_power_kw[lost].max() > 0


def test_an_equipment_fault_does_change_ground_truth():
    """The counterpart: equipment faults are supposed to move the plant."""
    config = _config()
    clock_index = config.clock().index()

    _, clean_equipment, _, _ = _run_with(FaultSchedule())
    _, equipment, _, _ = _run_with(_only(FaultType.COMPLETE_OUTAGE, clock_index, f"{SITE}_array_1"))
    assert equipment.site_actual_kw.sum() < clean_equipment.site_actual_kw.sum()


# ---------------------------------------------------------------------------
# Equipment faults and recorded loss
# ---------------------------------------------------------------------------


def test_complete_outage_takes_the_asset_offline(faulted):
    from gridguard.simulation.equipment import EquipmentState

    asset = faulted.sites[SITE].equipment.assets[0]
    outage = asset.active_fault == FaultType.COMPLETE_OUTAGE.value
    assert outage.any()
    assert (asset.state[outage] == EquipmentState.OFFLINE.value).all()
    assert np.allclose(asset.actual_ac_power_kw[outage], 0.0)


def test_lost_energy_is_recorded_not_modelled(faulted):
    """Loss is potential minus actual, both of which the layer kept."""
    sim = faulted.sites[SITE]
    lost = sim.equipment.site_potential_kw - sim.equipment.site_actual_kw
    assert (lost >= -1e-9).all()
    assert lost.sum() > 0

    summary = faulted.fault_summary()
    interval_hours = faulted.config.interval_minutes / 60.0
    assert np.isclose(summary["lost_kwh"].sum(), lost.sum() * interval_hours, rtol=1e-6)


def test_every_generation_loss_class_actually_costs_energy(faulted):
    """A fault scheduled at night costs nothing and tests nothing.

    Two classes silently had this problem before windows were anchored to the
    generating part of the day: shading cost exactly zero, and a 45% partial
    outage over twelve hours cost 9 kWh.
    """
    summary = faulted.fault_summary()
    loss_classes = summary[summary["fault_type"].isin({f.value for f in GENERATION_LOSS_FAULTS})]
    assert not loss_classes.empty

    free = loss_classes[loss_classes["lost_kwh"] <= 0.0]
    assert free.empty, (
        "these generation-loss faults cost nothing, so they exercise no detector:\n"
        f"{free[['fault_type', 'start', 'intervals', 'lost_kwh']]}"
    )


def test_clipping_actually_engages(faulted):
    """A scenario that includes clipping but never clips tests nothing.

    A nameplate-relative cap engaged at some sites and not others, because
    achievable peak varies with tilt and latitude. The cap is a percentile of
    the array's own potential for exactly this reason.
    """
    asset = faulted.sites[SITE].equipment.assets[0]
    clipped = asset.active_fault == FaultType.CLIPPING.value
    assert clipped.any(), "clipping never engaged"
    assert (asset.actual_ac_power_kw[clipped] < asset.potential_ac_power_kw[clipped]).all()


def test_sensor_and_comm_faults_cost_no_generation(faulted):
    """Neither touches the plant, so neither can cost energy.

    Clipping is deliberately excluded: it *does* withhold real energy, by
    design rather than by fault, which is why it is a non-loss *condition* but
    not a zero-cost one.
    """
    summary = faulted.fault_summary()
    untouched = summary[summary["fault_type"].isin({"sensor_dropout", "comm_dropout"})]
    assert len(untouched) == 2
    assert (untouched["lost_kwh"] == 0.0).all()


def test_clipping_is_a_non_loss_condition_that_still_withholds_energy(faulted):
    """Not a contradiction, and worth stating precisely.

    ``is_generation_loss`` means "the detector should flag this", which is False
    for clipping. It is not a claim that no energy was withheld.
    """
    assert FaultType.CLIPPING in NON_LOSS_CONDITIONS
    summary = faulted.fault_summary()
    clipping = summary[summary["fault_type"] == "clipping"].iloc[0]
    assert not clipping["is_generation_loss"]
    assert clipping["lost_kwh"] > 0.0


# ---------------------------------------------------------------------------
# Ground truth completeness
# ---------------------------------------------------------------------------


def test_truth_frame_carries_all_three_fault_columns(faulted):
    truth = faulted.truth()
    for column in ("active_fault", "active_sensor_fault", "active_comm_fault", "delivered"):
        assert column in truth.columns


def test_fault_summary_reports_category_and_layer(faulted):
    summary = faulted.fault_summary()
    assert set(summary["category"]) <= {c.value for c in FaultCategory}
    assert set(summary["layer"]) == {"equipment", "sensing", "transport"}


def test_comm_dropout_removes_rows_from_the_delivered_view(faulted, healthy):
    assert len(faulted.canonical()) < len(healthy.canonical())
    undelivered = int((~faulted.sites[SITE].delivery.delivered).sum())
    assert len(healthy.canonical()) - len(faulted.canonical()) == undelivered


# ---------------------------------------------------------------------------
# Determinism and validation
# ---------------------------------------------------------------------------


def test_injected_runs_are_reproducible():
    a = FleetSimulator(_config(inject_faults=True)).run()
    b = FleetSimulator(_config(inject_faults=True)).run()
    pd.testing.assert_frame_equal(a.canonical(), b.canonical())
    pd.testing.assert_frame_equal(a.fault_summary(), b.fault_summary())


def test_injection_is_off_by_default():
    assert not SimulationConfig(site_ids=(SITE,)).inject_faults
    assert len(FleetSimulator(_config()).run().schedule()) == 0


def test_fault_windows_must_be_timezone_aware():
    with pytest.raises(ValueError, match="timezone-aware"):
        FaultSpec(
            fault_type=FaultType.COMPLETE_OUTAGE,
            asset_id="s_array_1",
            start=datetime(2016, 6, 1),
            end=datetime(2016, 6, 2),
        )


def test_fault_rejects_reversed_window():
    with pytest.raises(ValueError, match="precedes"):
        FaultSpec(
            fault_type=FaultType.COMPLETE_OUTAGE,
            asset_id="s_array_1",
            start=datetime(2016, 6, 2, tzinfo=UTC),
            end=datetime(2016, 6, 1, tzinfo=UTC),
        )


def test_fault_rejects_out_of_range_severity():
    with pytest.raises(ValueError, match="severity"):
        FaultSpec(
            fault_type=FaultType.PARTIAL_OUTAGE,
            asset_id="s_array_1",
            start=datetime(2016, 6, 1, tzinfo=UTC),
            end=datetime(2016, 6, 2, tzinfo=UTC),
            severity=1.5,
        )


def test_layer_functions_refuse_faults_they_do_not_own():
    """A misrouted fault is an error, not a silent no-op."""
    spec = FaultSpec(
        fault_type=FaultType.COMPLETE_OUTAGE,
        asset_id="s_array_1",
        start=datetime(2016, 6, 1, tzinfo=UTC),
        end=datetime(2016, 6, 2, tzinfo=UTC),
    )
    index = pd.date_range("2016-06-01", periods=4, freq="15min", tz=UTC)

    with pytest.raises(ValueError, match="not a sensor fault"):
        apply_sensor_fault(spec, index, np.ones(4))
    with pytest.raises(ValueError, match="not a communication fault"):
        delivery_mask((spec,), index)


def test_scenario_returns_empty_for_a_run_too_short_to_separate_windows():
    index = pd.date_range("2016-06-01", periods=48, freq="15min", tz=UTC)
    schedule = build_layered_scenario({"array": "s_array_1"}, index)
    assert len(schedule) == 0


def test_schedule_can_be_filtered_by_site_and_layer(faulted):
    schedule = faulted.schedule()
    assert len(schedule.for_site(SITE)) == len(schedule)
    assert len(schedule.for_site("nowhere")) == 0
    assert len(schedule.for_layer("equipment")) == 6
