"""Tests for anomaly detection."""

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.anomaly.detect import compute_daily_loss, detect_anomalies
from gridguard.features.engineer import get_X_y


@pytest.fixture()
def trained_model_and_df():
    """Train a weather-only linear model on synthetic data for anomaly tests.

    Uses mode="weather_only" — matching the anomaly detection pipeline.
    Lag features must NOT be used here; see docs/methodology.md.
    """
    from tests._helpers import make_frame

    df = make_frame(start="2022-01-01", end="2022-03-31", seed=99)

    model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
    X, y = get_X_y(df, mode="weather_only")
    model.fit(X, y)
    return model, df


def test_detect_anomalies_columns(trained_model_and_df):
    model, df = trained_model_and_df
    result = detect_anomalies(df, model)
    for col in ["predicted_kw", "residual_kw", "is_anomaly", "lost_energy_kwh"]:
        assert col in result.columns, f"Missing column: {col}"


def test_anomalies_only_at_daylight(trained_model_and_df):
    model, df = trained_model_and_df
    result = detect_anomalies(df, model)
    # Anomalies should only occur when irradiance > 50
    night_anomalies = result[(result["irradiance_wm2"] <= 50) & result["is_anomaly"]]
    assert len(night_anomalies) == 0, "Anomalies flagged during night"


def test_lost_energy_non_negative(trained_model_and_df):
    model, df = trained_model_and_df
    result = detect_anomalies(df, model)
    assert (result["lost_energy_kwh"] >= 0).all()


def test_lost_energy_zero_for_normal(trained_model_and_df):
    model, df = trained_model_and_df
    result = detect_anomalies(df, model)
    normal = result[~result["is_anomaly"]]
    assert (normal["lost_energy_kwh"] == 0).all()


def test_predicted_kw_non_negative(trained_model_and_df):
    model, df = trained_model_and_df
    result = detect_anomalies(df, model)
    assert (result["predicted_kw"] >= 0).all()


def test_higher_threshold_fewer_anomalies(trained_model_and_df):
    model, df = trained_model_and_df
    r1 = detect_anomalies(df, model, threshold_sigma=1.5)
    r2 = detect_anomalies(df, model, threshold_sigma=3.0)
    assert r1["is_anomaly"].sum() >= r2["is_anomaly"].sum()


def test_compute_daily_loss_structure(trained_model_and_df):
    model, df = trained_model_and_df
    result = detect_anomalies(df, model)
    daily = compute_daily_loss(result)
    assert "date" in daily.columns
    assert "lost_energy_kwh" in daily.columns
    assert "anomaly_count" in daily.columns
    assert (daily["lost_energy_kwh"] >= 0).all()


def test_injected_faults_detected():
    """Injected fault days should have higher anomaly rates when model was not trained on them.

    Trains on one random seed, evaluates on another so the model is genuinely
    surprised by the fault-day patterns (different fault days chosen by each seed).
    """
    from tests._helpers import make_frame

    # Train on seed=0 — fault days are different from the test seed
    # mode="weather_only" — matches the anomaly detection pipeline
    df_train = make_frame(start="2022-01-01", end="2022-03-31", seed=0, inject_scenario=True)
    model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
    X, y = get_X_y(df_train, mode="weather_only")
    model.fit(X, y)

    # Evaluate on seed=99 — has its own (different) fault days
    df_test = make_frame(start="2022-01-01", end="2022-03-31", seed=99, inject_scenario=True)

    if "is_injected_fault" not in df_test.columns:
        pytest.skip("Injected fault labels not present")

    # freeze=False: compute residual stats from test df (no saved artifact dependency)
    result = detect_anomalies(df_test, model, threshold_sigma=1.5, freeze=False)
    result["date"] = result["timestamp"].dt.date
    fault_days = result[result["is_injected_fault"]]["date"].unique()
    normal_days = result[~result["is_injected_fault"]]["date"].unique()

    fault_anom_rate = result[result["date"].isin(fault_days)]["is_anomaly"].mean()
    normal_anom_rate = result[result["date"].isin(normal_days)]["is_anomaly"].mean()

    assert (
        fault_anom_rate > normal_anom_rate
    ), f"Fault detection rate ({fault_anom_rate:.2%}) not higher than normal ({normal_anom_rate:.2%})"


# ---------------------------------------------------------------------------
# No-lag guarantee tests — these are the credibility proofs
# ---------------------------------------------------------------------------


def test_weather_only_features_exclude_lag_columns():
    """ANOMALY_FEATURE_COLS must never include lag features.

    This is the core correctness property of the anomaly detection pipeline.
    A lag-aware model would learn that a degraded system producing low output
    has low lags → predicts low → residual is small → fault is missed.
    """
    from gridguard.anomaly.detect import ANOMALY_FEATURE_COLS

    assert (
        "ac_power_lag1" not in ANOMALY_FEATURE_COLS
    ), "ac_power_lag1 must NOT be in the anomaly feature set"
    assert (
        "ac_power_lag4" not in ANOMALY_FEATURE_COLS
    ), "ac_power_lag4 must NOT be in the anomaly feature set"


def test_detect_anomalies_does_not_pass_lag_columns_to_model(trained_model_and_df):
    """Verify detect_anomalies only passes weather/time columns to model.predict."""
    model, df = trained_model_and_df

    seen_columns: list[list[str]] = []
    original_predict = model.predict

    def recording_predict(X):
        seen_columns.append(list(X.columns))
        return original_predict(X)

    model.predict = recording_predict
    try:
        detect_anomalies(df, model, freeze=False)
    finally:
        model.predict = original_predict

    assert seen_columns, "model.predict should have been called"
    all_cols = {c for batch in seen_columns for c in batch}
    assert "ac_power_lag1" not in all_cols, "lag1 must not reach model.predict"
    assert "ac_power_lag4" not in all_cols, "lag4 must not reach model.predict"
