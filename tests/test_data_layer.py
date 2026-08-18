"""
Tests for the data layer: canonical schema, resampling, missing values,
timezone handling, fault injection, provenance, and PVDAQ channel resolution.

The PVDAQ tests run against small recorded fixtures rather than the live OEDI
data lake, so the suite passes in CI with no network access. What they verify is
the part that actually breaks: channel selection and unit handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridguard.data.faults import (
    GENERATION_LOSS_FAULTS,
    NON_LOSS_CONDITIONS,
    FaultEvent,
    FaultType,
    apply_fault,
    apply_faults,
    build_evaluation_scenario,
)
from gridguard.data.oedi import OEDIError, detect_utc_offset_hours, resolve_channels
from gridguard.data.provenance import DatasetProvenance, load_provenance, save_provenance
from gridguard.data.schema import (
    CANONICAL_COLUMNS,
    SchemaError,
    clip_physical_ranges,
    drop_unusable_rows,
    resample_to_interval,
    validate_canonical,
)

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------


def _canonical(n: int = 96, freq: str = "15min") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2016-06-15", periods=n, freq=freq),
            "site_id": "test",
            "ac_power_kw": np.linspace(0, 100, n),
            "irradiance_wm2": np.linspace(0, 900, n),
            "temperature_c": 20.0,
            "wind_speed_ms": 2.0,
            "data_mode": "real",
        }
    )


def test_validate_accepts_a_canonical_frame():
    out = validate_canonical(_canonical())
    assert list(out.columns)[: len(CANONICAL_COLUMNS)] == CANONICAL_COLUMNS


def test_validate_rejects_missing_columns():
    frame = _canonical().drop(columns=["irradiance_wm2"])
    with pytest.raises(SchemaError, match="missing required"):
        validate_canonical(frame)


def test_validate_rejects_timezone_aware_timestamps():
    """Timestamps must be naive local standard time; the offset lives in provenance."""
    frame = _canonical()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")
    with pytest.raises(SchemaError, match="timezone-naive"):
        validate_canonical(frame)


def test_validate_rejects_unknown_data_mode():
    frame = _canonical()
    frame["data_mode"] = "guesswork"
    with pytest.raises(SchemaError, match="data_mode"):
        validate_canonical(frame)


def test_validate_rejects_duplicates_in_strict_mode():
    frame = pd.concat([_canonical(4), _canonical(4)], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_canonical(frame, strict=True)


def test_validate_drops_duplicates_in_lenient_mode():
    frame = pd.concat([_canonical(4), _canonical(4)], ignore_index=True)
    out = validate_canonical(frame, strict=False)
    assert len(out) == 4


def test_validate_sorts_by_timestamp():
    frame = _canonical(8).sample(frac=1.0, random_state=0)
    assert validate_canonical(frame)["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Resampling and missing values
# ---------------------------------------------------------------------------


def test_resample_reduces_one_minute_data_to_fifteen():
    frame = _canonical(n=60, freq="1min")
    out = resample_to_interval(frame, 15)
    assert len(out) == 4
    assert (out["timestamp"].diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_resample_averages_within_the_interval():
    frame = _canonical(n=15, freq="1min")
    frame["ac_power_kw"] = 10.0
    out = resample_to_interval(frame, 15)
    assert out["ac_power_kw"].iloc[0] == pytest.approx(10.0)


def test_short_gaps_are_interpolated():
    frame = _canonical(n=60, freq="1min")
    frame.loc[20:21, "ac_power_kw"] = np.nan
    out = resample_to_interval(frame, 15)
    assert out["ac_power_kw"].notna().all()


def test_long_gaps_are_left_missing_not_invented():
    """A communications outage is a real operating condition, not a value to impute."""
    frame = _canonical(n=240, freq="1min")
    frame.loc[60:170, "ac_power_kw"] = np.nan  # nearly two hours
    out = resample_to_interval(frame, 15)
    assert out["ac_power_kw"].isna().any()


def test_resample_rejects_multiple_sites():
    frame = pd.concat([_canonical(4), _canonical(4).assign(site_id="other")])
    with pytest.raises(ValueError, match="single site"):
        resample_to_interval(frame, 15)


def test_drop_unusable_rows_removes_missing_power():
    frame = _canonical(10)
    frame.loc[3, "ac_power_kw"] = np.nan
    assert len(drop_unusable_rows(frame)) == 9


def test_clip_removes_instrument_offsets():
    """Pyranometers read slightly negative at night; that is not generation."""
    frame = _canonical(10)
    frame.loc[0, "irradiance_wm2"] = -0.4
    frame.loc[1, "ac_power_kw"] = -0.2
    out = clip_physical_ranges(frame)
    assert out["irradiance_wm2"].min() >= 0
    assert out["ac_power_kw"].min() >= 0


def test_clip_flags_implausible_power_as_missing():
    frame = _canonical(10)
    frame.loc[5, "ac_power_kw"] = 10_000.0
    out = clip_physical_ranges(frame, capacity_kw=100.0)
    assert pd.isna(out.loc[5, "ac_power_kw"])


def test_clip_does_not_cap_legitimate_over_nameplate_excursions():
    frame = _canonical(10)
    frame.loc[5, "ac_power_kw"] = 105.0
    out = clip_physical_ranges(frame, capacity_kw=100.0)
    assert out.loc[5, "ac_power_kw"] == 105.0


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------


def test_utc_offset_is_derived_from_published_columns():
    """PVDAQ publishes both local and UTC, so the offset is observed, not assumed."""
    local = pd.date_range("2016-06-15", periods=10, freq="1min")
    frame = pd.DataFrame({"measured_on": local, "utc_measured_on": local + pd.Timedelta(hours=5)})
    assert detect_utc_offset_hours(frame) == -5


def test_constant_offset_across_a_dst_boundary_confirms_standard_time():
    """A single offset spanning March and July means no daylight-saving shift."""
    local = pd.to_datetime(["2016-01-15 12:00", "2016-07-15 12:00"])
    frame = pd.DataFrame({"measured_on": local, "utc_measured_on": local + pd.Timedelta(hours=5)})
    assert detect_utc_offset_hours(frame) == -5


def test_missing_utc_column_returns_none():
    frame = pd.DataFrame({"measured_on": pd.date_range("2016-06-15", periods=3, freq="1min")})
    assert detect_utc_offset_hours(frame) is None


def test_inconsistent_offsets_return_none():
    local = pd.to_datetime(["2016-01-15 12:00", "2016-07-15 12:00"])
    frame = pd.DataFrame(
        {
            "measured_on": local,
            "utc_measured_on": [local[0] + pd.Timedelta(hours=5), local[1] + pd.Timedelta(hours=4)],
        }
    )
    assert detect_utc_offset_hours(frame) is None


# ---------------------------------------------------------------------------
# PVDAQ channel resolution
# ---------------------------------------------------------------------------


def _metrics() -> pd.DataFrame:
    """A metrics table shaped like the real NIST system 4902 one.

    Includes the traps that make name-based selection wrong: irradiance
    published in raw millivolts, an "AC power" row that is actually reactive
    energy, and wind speed misfiled under "AC other".
    """
    return pd.DataFrame(
        [
            # instantaneous AC power from the inverter — the right choice
            dict(
                metric_id=82607,
                sensor_name="InvPAC_kW_Avg",
                common_name="AC power",
                units="kW",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="INVERTER",
            ),
            # cumulative energy, not power
            dict(
                metric_id=82636,
                sensor_name="PwrMtrEdel_kWh_Max",
                common_name="AC power",
                units="kWh",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="METER",
            ),
            # reactive energy despite the name
            dict(
                metric_id=82638,
                sensor_name="PwrMtrEdel_kVARh_Max",
                common_name="AC power",
                units="kVARh",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="METER",
            ),
            # raw pyranometer millivolts — unusable without a calibration constant
            dict(
                metric_id=82593,
                sensor_name="Pyra1_mV_Avg",
                common_name="Irradiance GHI",
                units="mV",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
            dict(
                metric_id=82594,
                sensor_name="Pyra2_mV_Avg",
                common_name="Irradiance POA",
                units="mV",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
            # calibrated W/m^2 — the right choice
            dict(
                metric_id=82595,
                sensor_name="RefCell1_Wm2_Avg",
                common_name="Irradiance POA",
                units="W/m^2",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
            dict(
                metric_id=82596,
                sensor_name="AmbTemp_C_Avg",
                common_name="Temperature ambient",
                units="C",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
            dict(
                metric_id=82621,
                sensor_name="SEWSModuleTemp_C_Avg",
                common_name="Temperature module",
                units="C",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
            # wind speed misfiled under "AC other" upstream
            dict(
                metric_id=82665,
                sensor_name="WindSpeedAve_ms",
                common_name="AC other",
                units="m/s",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
            dict(
                metric_id=82668,
                sensor_name="WindSpeed_ms_Max",
                common_name="AC other",
                units="m/s",
                calc_scale=1.0,
                calc_offset=0.0,
                source_type="OTHER",
            ),
        ]
    )


def test_resolves_instantaneous_power_not_cumulative_energy():
    channels = resolve_channels(_metrics())
    assert channels.ac_power.metric_id == 82607
    assert channels.ac_power.units == "kW"


def test_rejects_raw_millivolt_irradiance_in_favour_of_calibrated():
    """Converting mV needs a calibration constant the dataset does not publish."""
    channels = resolve_channels(_metrics())
    assert channels.irradiance.metric_id == 82595
    assert channels.irradiance.units == "W/m^2"
    assert channels.irradiance_kind == "plane-of-array"


def test_resolves_wind_despite_wrong_common_name():
    channels = resolve_channels(_metrics())
    assert channels.wind_speed.metric_id == 82665  # the averaging channel, not the max


def test_resolves_ambient_and_module_temperature_separately():
    channels = resolve_channels(_metrics())
    assert channels.temperature.metric_id == 82596
    assert channels.module_temperature.metric_id == 82621


def test_raises_when_no_calibrated_irradiance_exists():
    metrics = _metrics()
    metrics = metrics[metrics["units"] != "W/m^2"]
    with pytest.raises(OEDIError, match="W/m\\^2"):
        resolve_channels(metrics)


def test_raises_when_no_instantaneous_power_exists():
    metrics = _metrics()
    metrics = metrics[metrics["units"] != "kW"]
    with pytest.raises(OEDIError, match="AC power"):
        resolve_channels(metrics)


def test_channel_scale_and_offset_are_applied():
    metrics = _metrics()
    metrics.loc[metrics["metric_id"] == 82595, "calc_scale"] = 2.0
    metrics.loc[metrics["metric_id"] == 82595, "calc_offset"] = 1.0
    channels = resolve_channels(metrics)
    result = channels.irradiance.apply(pd.Series([10.0, 20.0]))
    assert list(result) == [21.0, 41.0]


def test_resolution_is_recorded_for_traceability():
    note = resolve_channels(_metrics()).as_note()
    assert "InvPAC_kW_Avg" in note
    assert "RefCell1_Wm2_Avg" in note


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def _daylight_frame(n: int = 96) -> pd.DataFrame:
    frame = _canonical(n)
    frame["irradiance_wm2"] = 600.0
    frame["ac_power_kw"] = 100.0
    frame["data_mode"] = "synthetic"
    return frame


def test_complete_outage_zeroes_generation():
    frame = _daylight_frame()
    event = FaultEvent(
        FaultType.COMPLETE_OUTAGE,
        frame["timestamp"].iloc[10],
        frame["timestamp"].iloc[20],
        severity=1.0,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    window = out.iloc[10:21]
    assert (window["ac_power_kw"] == 0).all()
    assert window["is_injected_fault"].all()
    assert window["is_generation_loss"].all()


def test_partial_outage_scales_generation():
    frame = _daylight_frame()
    event = FaultEvent(
        FaultType.PARTIAL_OUTAGE,
        frame["timestamp"].iloc[10],
        frame["timestamp"].iloc[20],
        severity=0.4,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert out["ac_power_kw"].iloc[15] == pytest.approx(60.0)


def test_baseline_is_retained_for_exact_loss_accounting():
    """Ground-truth lost energy needs the pre-fault series, not a model of it."""
    frame = _daylight_frame()
    event = FaultEvent(
        FaultType.PARTIAL_OUTAGE,
        frame["timestamp"].iloc[10],
        frame["timestamp"].iloc[20],
        severity=0.5,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert out["ac_power_baseline_kw"].iloc[15] == pytest.approx(100.0)
    assert out["ac_power_kw"].iloc[15] == pytest.approx(50.0)


def test_sensor_dropout_freezes_irradiance_but_not_generation():
    """A data-quality fault: nothing is actually lost."""
    frame = _daylight_frame()
    frame["irradiance_wm2"] = np.linspace(100, 900, len(frame))
    event = FaultEvent(
        FaultType.SENSOR_DROPOUT,
        frame["timestamp"].iloc[30],
        frame["timestamp"].iloc[50],
        severity=1.0,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert out["irradiance_wm2"].iloc[30:51].nunique() == 1
    assert out["ac_power_kw"].iloc[40] == pytest.approx(100.0)
    assert not out["is_generation_loss"].iloc[40]


def test_comm_dropout_blanks_power_without_losing_energy():
    frame = _daylight_frame()
    event = FaultEvent(
        FaultType.COMM_DROPOUT,
        frame["timestamp"].iloc[10],
        frame["timestamp"].iloc[20],
        severity=1.0,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert out["ac_power_kw"].iloc[10:21].isna().all()
    assert not out["is_generation_loss"].iloc[10:21].any()


def test_clipping_is_not_a_generation_loss():
    """Inverter clipping is healthy design behaviour, scored as a negative."""
    frame = _daylight_frame()
    event = FaultEvent(
        FaultType.CLIPPING,
        frame["timestamp"].iloc[10],
        frame["timestamp"].iloc[20],
        severity=0.25,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert out["ac_power_kw"].iloc[15] == pytest.approx(75.0)
    assert not out["is_generation_loss"].iloc[15]


def test_gradual_degradation_ramps():
    frame = _daylight_frame(n=200)
    event = FaultEvent(
        FaultType.GRADUAL_DEGRADATION,
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[199],
        severity=0.5,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert out["ac_power_kw"].iloc[10] > out["ac_power_kw"].iloc[190]


def test_faults_do_not_apply_at_night():
    frame = _daylight_frame()
    frame["irradiance_wm2"] = 0.0
    frame["ac_power_kw"] = 0.0
    event = FaultEvent(
        FaultType.COMPLETE_OUTAGE,
        frame["timestamp"].iloc[10],
        frame["timestamp"].iloc[20],
        severity=1.0,
    )
    out = apply_fault(frame, event, capacity_kw=100.0)
    assert not out["is_injected_fault"].any()


def test_taxonomy_partitions_into_loss_and_non_loss():
    assert GENERATION_LOSS_FAULTS.isdisjoint(NON_LOSS_CONDITIONS)
    assert GENERATION_LOSS_FAULTS | NON_LOSS_CONDITIONS == set(FaultType)


def test_evaluation_scenario_covers_every_fault_class():
    timestamps = pd.Series(pd.date_range("2022-01-01", periods=96 * 200, freq="15min"))
    scenario = build_evaluation_scenario(timestamps)
    assert {e.fault_type for e in scenario} == set(FaultType)


def test_evaluation_scenario_is_deterministic():
    timestamps = pd.Series(pd.date_range("2022-01-01", periods=96 * 200, freq="15min"))
    first = build_evaluation_scenario(timestamps, seed=7)
    second = build_evaluation_scenario(timestamps, seed=7)
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


def test_short_window_declines_to_inject():
    timestamps = pd.Series(pd.date_range("2022-01-01", periods=96 * 5, freq="15min"))
    assert build_evaluation_scenario(timestamps) == []


def test_layered_faults_never_produce_negative_power():
    frame = _daylight_frame()
    events = [
        FaultEvent(
            FaultType.PARTIAL_OUTAGE,
            frame["timestamp"].iloc[10],
            frame["timestamp"].iloc[40],
            severity=0.6,
        ),
        FaultEvent(
            FaultType.PERSISTENT_DERATE,
            frame["timestamp"].iloc[20],
            frame["timestamp"].iloc[50],
            severity=0.5,
        ),
    ]
    out = apply_faults(frame, events, capacity_kw=100.0)
    assert (out["ac_power_kw"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_rejects_unknown_data_mode():
    with pytest.raises(ValueError, match="data_mode"):
        DatasetProvenance(site_id="x", data_mode="invented", dataset="d")


def test_provenance_round_trips(tmp_path):
    record = DatasetProvenance(
        site_id="nist_ground",
        data_mode="real",
        dataset="NREL PVDAQ (OEDI)",
        source_url="https://openei.org/wiki/PVDAQ",
        processing_notes=["resampled"],
        known_limitations=["no fault labels"],
    )
    path = save_provenance([record], tmp_path / "provenance.json")
    loaded = load_provenance(path)
    assert len(loaded) == 1
    assert loaded[0].site_id == "nist_ground"
    assert loaded[0].is_real
    assert loaded[0].known_limitations == ["no fault labels"]


def test_loading_absent_provenance_returns_empty(tmp_path):
    assert load_provenance(tmp_path / "nothing.json") == []


def test_provenance_stamps_retrieval_time():
    record = DatasetProvenance(site_id="x", data_mode="synthetic", dataset="sim")
    assert record.retrieved_at
