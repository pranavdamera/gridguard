"""Tests for anomaly event grouping."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.anomaly.events import group_anomaly_events, _empty_events_df, GAP_THRESHOLD_MINUTES
from gridguard.anomaly.detect import detect_anomalies
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
    # Train a fast model
    from gridguard.features.engineer import build_features
    model = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    X, y = get_X_y(df)
    model.fit(X, y)
    return detect_anomalies(df, model, freeze=False)


def test_group_returns_dataframe(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    assert isinstance(events, pd.DataFrame)


def test_required_columns_present(anomaly_df_fixture):
    events = group_anomaly_events(anomaly_df_fixture)
    required = {
        "event_id", "start_time", "end_time", "duration_minutes",
        "interval_count", "total_lost_kwh", "max_residual_sigma",
        "mean_actual_kw", "mean_predicted_kw", "severity",
    }
    assert required.issubset(set(events.columns))


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


def test_severity_thresholds():
    """Check severity classification boundaries."""
    from gridguard.anomaly.events import _classify_severity

    assert _classify_severity(0.5) == "low"
    assert _classify_severity(1.0) == "medium"
    assert _classify_severity(5.0) == "medium"
    assert _classify_severity(10.0) == "high"
    assert _classify_severity(50.0) == "high"
