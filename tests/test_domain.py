"""
Domain model tests.

Two things here are load-bearing beyond ordinary unit coverage:

* **Capacity reconciliation.** Introducing asset structure must not change any
  number the project already publishes.
* **Time round-tripping.** ``event_time`` and ``timestamp`` must be exact
  inverses. A shim that is not an exact round trip silently rewrites history,
  and every model, calibration and manifest in the repository is indexed by the
  local-time column.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from gridguard.data.faults import FaultType, fault_category
from gridguard.data.schema import (
    DOMAIN_COLUMNS,
    has_domain_columns,
    upgrade_to_domain_schema,
)
from gridguard.domain import (
    Asset,
    AssetKind,
    AssetTree,
    Evidence,
    FaultCategory,
    FaultHypothesis,
    InvalidIdError,
    QualityFlag,
    TelemetryRecord,
    asset_id,
    event_time_to_local,
    local_to_event_time,
    validate_id,
)
from gridguard.sites.registry import (
    assets_for_site,
    get_site,
    list_site_ids,
    load_assets,
    primary_asset_id,
)

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def test_asset_id_is_derived_and_readable():
    assert asset_id("nist_roof", "inverter", 2) == "nist_roof_inverter_2"


@pytest.mark.parametrize("bad", ["NIST_roof", "nist-roof", "nist__roof", "_nist", "nist_", ""])
def test_validate_id_rejects_malformed(bad):
    with pytest.raises(InvalidIdError):
        validate_id(bad)


def test_asset_id_rejects_zero_ordinal():
    with pytest.raises(ValueError, match="ordinal"):
        asset_id("nist_roof", "inverter", 0)


def test_stable_seed_survives_the_process_hash_salt():
    """Determinism must not depend on PYTHONHASHSEED.

    The site simulator was once seeded from ``hash()`` and produced different
    "deterministic" data on every run. This is the guard against that returning.
    """
    import subprocess
    import sys

    code = (
        "from gridguard.domain.ids import stable_seed;"
        "print(stable_seed('nist_roof', 'inverter'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"stable_seed varied with PYTHONHASHSEED: {runs}"


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def test_every_site_has_assets():
    trees = load_assets()
    assert set(trees) == set(list_site_ids())


def test_asset_capacity_reconciles_with_the_site_registry():
    """The number the project already publishes must not move."""
    mismatches = {}
    for site_id in list_site_ids():
        site = get_site(site_id)
        tree = assets_for_site(site_id)
        if abs(tree.total_capacity_kw - site.capacity_kw) > 1e-6:
            mismatches[site_id] = (site.capacity_kw, tree.total_capacity_kw)
    assert not mismatches, f"array capacities do not sum to registry capacity: {mismatches}"


def test_every_site_has_exactly_one_primary_array():
    for site_id in list_site_ids():
        assert primary_asset_id(site_id).startswith(site_id)


def test_sensors_are_assets_not_site_properties():
    """A frozen pyranometer must be attributable to a thing, not to a site."""
    for site_id in list_site_ids():
        tree = assets_for_site(site_id)
        assert tree.of_kind(AssetKind.PYRANOMETER), f"{site_id} has no irradiance sensor"


def test_generating_assets_must_declare_capacity():
    with pytest.raises(ValueError, match="capacity_kw"):
        Asset(asset_id="s_array_1", site_id="s", kind=AssetKind.ARRAY)


def test_sensing_assets_must_not_declare_capacity():
    with pytest.raises(ValueError, match="do not generate"):
        Asset(asset_id="s_meter_1", site_id="s", kind=AssetKind.METER, capacity_kw=5.0)


def test_asset_tree_rejects_a_dangling_parent():
    with pytest.raises(ValueError, match="not an asset"):
        AssetTree(
            site_id="s",
            assets=(
                Asset(
                    asset_id="s_inverter_1",
                    site_id="s",
                    kind=AssetKind.INVERTER,
                    parent_id="s_array_9",
                ),
            ),
        )


def test_asset_tree_rejects_duplicate_ids():
    a = Asset(asset_id="s_array_1", site_id="s", kind=AssetKind.ARRAY, capacity_kw=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        AssetTree(site_id="s", assets=(a, a))


def test_total_capacity_does_not_double_count_strings():
    """Strings live under arrays; counting both would double the silicon."""
    tree = AssetTree(
        site_id="s",
        assets=(
            Asset(asset_id="s_array_1", site_id="s", kind=AssetKind.ARRAY, capacity_kw=100.0),
            Asset(
                asset_id="s_string_1",
                site_id="s",
                kind=AssetKind.STRING,
                parent_id="s_array_1",
                capacity_kw=50.0,
            ),
        ),
    )
    assert tree.total_capacity_kw == 100.0


def test_inverters_hang_off_arrays_in_the_real_registry():
    tree = assets_for_site("nist_roof")
    inverters = tree.of_kind(AssetKind.INVERTER)
    assert inverters
    for inverter in inverters:
        assert inverter.parent_id is not None
        assert tree.get(inverter.parent_id).kind == AssetKind.ARRAY


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_event_time_round_trips_exactly():
    """The shim must be lossless in both directions."""
    local = pd.Series(pd.date_range("2016-06-01", periods=500, freq="15min"))
    for offset in (-5, -8, 0, 5):
        event = local_to_event_time(local, offset)
        back = event_time_to_local(event, offset)
        pd.testing.assert_series_equal(local, back, check_names=False)


def test_local_to_event_time_produces_utc():
    event = local_to_event_time(pd.Series(pd.date_range("2016-06-01", periods=3, freq="h")), -5)
    assert str(event.dt.tz) == "UTC"
    # Noon local standard at UTC-5 is 17:00 UTC.
    noon = local_to_event_time(pd.Series([pd.Timestamp("2016-06-01 12:00")]), -5)
    assert noon.iloc[0] == pd.Timestamp("2016-06-01 17:00", tz="UTC")


def test_local_to_event_time_refuses_tz_aware_input():
    aware = pd.Series(pd.date_range("2016-06-01", periods=3, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="naive local standard time"):
        local_to_event_time(aware, -5)


# ---------------------------------------------------------------------------
# Telemetry records
# ---------------------------------------------------------------------------


def test_telemetry_record_rejects_naive_times():
    with pytest.raises(ValueError, match="timezone-aware"):
        TelemetryRecord(
            asset_id="s_array_1",
            site_id="s",
            channel="ac_power_kw",
            value=1.0,
            unit="kW",
            event_time=datetime(2016, 6, 1, 12, 0),
        )


def test_missing_value_must_be_flagged_missing():
    """Absence must never be representable as an unflagged None."""
    with pytest.raises(ValueError, match="MISSING"):
        TelemetryRecord(
            asset_id="s_array_1",
            site_id="s",
            channel="ac_power_kw",
            value=None,
            unit="kW",
            event_time=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
        )


def test_arrival_lag_is_measurable():
    event = datetime(2016, 6, 1, 12, 0, tzinfo=UTC)
    record = TelemetryRecord(
        asset_id="s_array_1",
        site_id="s",
        channel="ac_power_kw",
        value=1.0,
        unit="kW",
        event_time=event,
        ingest_time=event + timedelta(seconds=42),
    )
    assert record.arrival_lag_seconds == 42.0


def test_arrival_lag_is_unknown_not_zero_when_never_ingested():
    """None and 0.0 mean different things and must not be conflated."""
    record = TelemetryRecord(
        asset_id="s_array_1",
        site_id="s",
        channel="ac_power_kw",
        value=1.0,
        unit="kW",
        event_time=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
    )
    assert record.arrival_lag_seconds is None


def test_quality_flags_compose():
    flags = QualityFlag.STALE | QualityFlag.OUT_OF_ORDER
    assert flags & QualityFlag.STALE
    assert not flags & QualityFlag.FROZEN
    assert flags.is_measured


def test_interpolated_samples_are_not_measured():
    assert not QualityFlag.INTERPOLATED.is_measured
    assert not QualityFlag.MISSING.is_measured
    assert QualityFlag.OK.is_measured


# ---------------------------------------------------------------------------
# Schema upgrade
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_frame() -> pd.DataFrame:
    """A frame in exactly the shape the committed parquet files carry."""
    n = 48
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2016-06-01", periods=n, freq="15min"),
            "site_id": "nist_roof",
            "ac_power_kw": 10.0,
            "irradiance_wm2": 500.0,
            "temperature_c": 20.0,
            "wind_speed_ms": 3.0,
            "data_mode": "real",
        }
    )


def test_legacy_frames_lack_domain_columns(legacy_frame):
    assert not has_domain_columns(legacy_frame)


def test_upgrade_adds_every_domain_column(legacy_frame):
    upgraded = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    assert has_domain_columns(upgraded)
    for column in DOMAIN_COLUMNS:
        assert column in upgraded.columns


def test_upgrade_preserves_the_original_columns_untouched(legacy_frame):
    """Nothing the existing pipeline reads may change."""
    upgraded = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    for column in legacy_frame.columns:
        pd.testing.assert_series_equal(legacy_frame[column], upgraded[column], check_names=False)


def test_upgrade_event_time_matches_the_local_timestamp(legacy_frame):
    upgraded = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    back = event_time_to_local(upgraded["event_time"], -5)
    pd.testing.assert_series_equal(upgraded["timestamp"], back, check_names=False)


def test_upgrade_reports_zero_arrival_lag_for_batch_data(legacy_frame):
    """A file did not travel; claiming otherwise would invent a network."""
    upgraded = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    lag = (upgraded["ingest_time"] - upgraded["event_time"]).dt.total_seconds()
    assert (lag == 0).all()


def test_upgrade_attributes_rows_to_the_primary_array(legacy_frame):
    upgraded = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    assert (upgraded["asset_id"] == primary_asset_id("nist_roof")).all()


def test_upgrade_is_idempotent(legacy_frame):
    """Calling twice must not overwrite real ingest times with synthesised ones."""
    once = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    once = once.assign(ingest_time=once["event_time"] + pd.Timedelta(seconds=30))
    twice = upgrade_to_domain_schema(once, utc_offset_hours=-5)
    pd.testing.assert_series_equal(once["ingest_time"], twice["ingest_time"])


def test_upgrade_needs_an_asset_for_multi_site_frames(legacy_frame):
    from gridguard.data.schema import SchemaError

    mixed = pd.concat([legacy_frame, legacy_frame.assign(site_id="nist_ground")])
    with pytest.raises(SchemaError, match="explicit asset_id"):
        upgrade_to_domain_schema(mixed, utc_offset_hours=-5)


def test_upgrade_quality_defaults_to_ok(legacy_frame):
    upgraded = upgrade_to_domain_schema(legacy_frame, utc_offset_hours=-5)
    assert (upgraded["quality"] == int(QualityFlag.OK)).all()


# ---------------------------------------------------------------------------
# Fault categories
# ---------------------------------------------------------------------------


def test_every_fault_class_has_a_category():
    for fault_type in FaultType:
        assert isinstance(fault_category(fault_type), FaultCategory)


def test_categories_separate_the_two_non_loss_conditions():
    """The whole reason for the category axis.

    ``sensor_dropout`` and ``comm_dropout`` are both non-loss conditions and are
    indistinguishable under the old taxonomy. They have completely different
    causes and completely different operator responses.
    """
    assert fault_category(FaultType.SENSOR_DROPOUT) == FaultCategory.SENSOR
    assert fault_category(FaultType.COMM_DROPOUT) == FaultCategory.COMMUNICATION


def test_fault_type_string_values_are_unchanged():
    """Committed artifacts and API responses must stay wire-compatible."""
    assert {f.value for f in FaultType} == {
        "complete_outage",
        "partial_outage",
        "persistent_derate",
        "shading",
        "gradual_degradation",
        "sensor_dropout",
        "comm_dropout",
        "clipping",
    }


def test_unknown_fault_type_is_an_error_not_a_default():
    with pytest.raises(KeyError, match="No category declared"):
        fault_category("meteor_strike")


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


def _hypothesis(**overrides) -> FaultHypothesis:
    kwargs = {
        "fault_type": "complete_outage",
        "category": FaultCategory.EQUIPMENT,
        "asset_id": "nist_roof_inverter_1",
        "site_id": "nist_roof",
        "confidence": 0.8,
        "confidence_basis": "signature rates measured on the injection set",
    }
    kwargs.update(overrides)
    return FaultHypothesis(**kwargs)


def test_hypothesis_requires_a_stated_confidence_basis():
    """An unexplained confidence value is the thing the project forbids."""
    with pytest.raises(ValueError, match="confidence_basis"):
        _hypothesis(confidence_basis="   ")


def test_hypothesis_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match="confidence"):
        _hypothesis(confidence=1.4)


def test_hypothesis_separates_supporting_from_opposing_evidence():
    """An engine that can only accumulate support will rank contradicted classes."""
    hypothesis = _hypothesis(
        evidence=(
            Evidence("power_zero", "Output is zero in daylight.", supports=True, weight=0.9),
            Evidence("neighbours_down", "All neighbours equally down.", supports=False, weight=0.6),
        )
    )
    assert len(hypothesis.supporting) == 1
    assert len(hypothesis.opposing) == 1
    assert "power_zero" in hypothesis.explain()
    assert "- neighbours_down" in hypothesis.explain()


def test_evidence_weight_is_bounded():
    with pytest.raises(ValueError, match="weight"):
        Evidence("x", "y", supports=True, weight=1.5)
