"""Tests for anomaly detection."""

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.anomaly.detect import compute_daily_loss, detect_anomalies
from gridguard.features.engineer import get_X_y


@pytest.fixture()
def trained_model_and_df():
    """Train a fast linear model on synthetic data for use in anomaly tests."""
    from gridguard.ingestion.download import _generate_synthetic

    df = _generate_synthetic(start="2022-01-01", end="2022-03-31", seed=99)

    # Use a fast linear model to avoid slow RF/XGB in tests
    model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
    X, y = get_X_y(df)
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


def test_injected_faults_detected(trained_model_and_df):
    """Injected fault days in synthetic data should have higher anomaly counts."""
    model, df = trained_model_and_df
    result = detect_anomalies(df, model, threshold_sigma=1.5)

    if "is_injected_fault" not in result.columns:
        pytest.skip("Injected fault labels not present")

    result["date"] = result["timestamp"].dt.date
    fault_days = result[result["is_injected_fault"]]["date"].unique()
    normal_days = result[~result["is_injected_fault"]]["date"].unique()

    fault_anom_rate = result[result["date"].isin(fault_days)]["is_anomaly"].mean()
    normal_anom_rate = result[result["date"].isin(normal_days)]["is_anomaly"].mean()

    assert fault_anom_rate > normal_anom_rate, (
        f"Fault detection rate ({fault_anom_rate:.2%}) not higher than normal ({normal_anom_rate:.2%})"
    )
