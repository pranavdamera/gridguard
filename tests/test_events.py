"""Tests for anomaly event grouping."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.anomaly.detect import detect_anomalies
from gridguard.anomaly.events import (
    explain_event_text,
    group_anomaly_events,
)
from gridguard.features.engineer import get_X_y


@pytest.fixture()
def anomaly_df_fixture():
    """Minimal anomaly_df with known anomaly structure for deterministic tests."""
    idx = pd.date_range("2023-06-01", periods=200, freq="15min")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ac_power_kw": np.abs(rng.normal(3, 1, 200)),
            "irradiance_wm2": np.abs(rng.normal(500, 200, 200)),
            "temperature_c": rng.normal(25, 5, 200),
            "wind_speed_ms": np.abs(rng.normal(3, 1, 200)),
        }
    )
    # Train with weather_only features — matches the anomaly detection pipeline
    model = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    X, y = get_X_y(df, mode="weather_only")
    model.fit(X, y)
    return detect_anomalies(df, model, freeze=False)


def test_group_returns_dataframe(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    assert isinstance(events, pd.DataFrame)


def test_required_columns_present(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    required = {
        "event_id",
        "start_time",
        "end_time",
        "duration_minutes",
        "interval_count",
        "total_lost_kwh",
        "max_residual_sigma",
        "mean_actual_kw",
        "mean_predicted_kw",
        "severity",
        "explanation",
    }
    assert required.issubset(set(events.columns))


def test_explanation_is_nonempty_string(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    if not events.empty:
        assert events["explanation"].apply(lambda s: isinstance(s, str) and len(s) > 0).all()


def test_severity_values_valid(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    if not events.empty:
        assert set(events["severity"].unique()).issubset({"low", "medium", "high"})


def test_event_ids_are_unique(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    if not events.empty:
        assert events["event_id"].is_unique


def test_start_before_end(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    if not events.empty:
        assert (events["start_time"] <= events["end_time"]).all()


def test_duration_positive(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    if not events.empty:
        assert (events["duration_minutes"] > 0).all()


def test_sorted_by_start_time_descending(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    if len(events) > 1:
        assert events["start_time"].is_monotonic_decreasing


def test_empty_when_no_anomalies():
    """If is_anomaly is always False, result should be empty."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="15min"),
            "is_anomaly": [False] * 10,
            "ac_power_kw": [5.0] * 10,
            "predicted_kw": [5.0] * 10,
            "residual_sigma": [0.0] * 10,
            "lost_energy_kwh": [0.0] * 10,
        }
    )
    events = group_anomaly_events(df)
    assert events.empty


def test_gap_creates_separate_events():
    """Two anomaly blocks separated by a gap > GAP_THRESHOLD_MINUTES → 2 events."""
    ts1 = pd.date_range("2023-01-01 10:00", periods=4, freq="15min")
    ts2 = pd.date_range("2023-01-01 14:00", periods=4, freq="15min")  # 2-hour gap
    timestamps = ts1.tolist() + ts2.tolist()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "is_anomaly": [True] * 8,
            "ac_power_kw": [2.0] * 8,
            "predicted_kw": [5.0] * 8,
            "residual_sigma": [-3.0] * 8,
            "lost_energy_kwh": [0.75] * 8,
        }
    )
    events = group_anomaly_events(df)
    assert len(events) == 2


def test_consecutive_intervals_merged_into_one_event():
    """Eight consecutive anomaly intervals → 1 event."""
    timestamps = pd.date_range("2023-01-01 10:00", periods=8, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "is_anomaly": [True] * 8,
            "ac_power_kw": [2.0] * 8,
            "predicted_kw": [5.0] * 8,
            "residual_sigma": [-3.0] * 8,
            "lost_energy_kwh": [0.75] * 8,
        }
    )
    events = group_anomaly_events(df)
    assert len(events) == 1
    assert events.iloc[0]["interval_count"] == 8
    assert abs(events.iloc[0]["total_lost_kwh"] - 6.0) < 0.01


def test_total_lost_kwh_sums_correctly():
    timestamps = pd.date_range("2023-01-01 10:00", periods=4, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "is_anomaly": [True] * 4,
            "ac_power_kw": [2.0] * 4,
            "predicted_kw": [5.0] * 4,
            "residual_sigma": [-3.0] * 4,
            "lost_energy_kwh": [1.5] * 4,
        }
    )
    events = group_anomaly_events(df)
    assert abs(events.iloc[0]["total_lost_kwh"] - 6.0) < 0.01


def test_severity_is_driven_by_shortfall_fraction():
    """Severity reflects the fraction of expected generation lost, not raw kWh.

    An absolute threshold cannot work across a fleet spanning 60 kW to 500 kW:
    10 kWh is a rounding error for one and a total outage for the other.
    """
    from gridguard.anomaly.events import _classify_severity

    # Same absolute loss, very different meaning.
    assert _classify_severity(10.0, 0.60) == "high"
    assert _classify_severity(10.0, 0.30) == "medium"
    assert _classify_severity(10.0, 0.05) == "low"

    # Boundaries.
    assert _classify_severity(5.0, 0.50) == "high"
    assert _classify_severity(5.0, 0.20) == "medium"
    assert _classify_severity(5.0, 0.19) == "low"


def test_energetically_trivial_events_are_never_escalated():
    """A near-total shortfall of almost no energy is not a critical event."""
    from gridguard.anomaly.events import _classify_severity

    assert _classify_severity(0.1, 0.95) == "low"


def test_explain_event_text_contains_key_info():
    event = {
        "start_time": pd.Timestamp("2023-06-15 09:00"),
        "duration_minutes": 195,
        "total_lost_kwh": 23.6,
        "severity": "high",
        "mean_predicted_kw": 45.2,
        "mean_actual_kw": 13.5,
    }
    text = explain_event_text(event)
    assert "23.6 kWh" in text
    assert "high" in text
    assert "2023-06-15" in text
    assert "45.2 kW" in text
    assert "13.5 kW" in text
