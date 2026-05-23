"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from gridguard.features.engineer import (
    FEATURE_COLS,
    LAG_AWARE_FEATURES,
    WEATHER_ONLY_FEATURES,
    build_features,
    get_feature_cols,
    get_X_y,
    train_test_split_by_date,
)


@pytest.fixture()
def sample_df():
    """Minimal 2-day DataFrame with all required raw columns."""
    idx = pd.date_range("2023-06-01", periods=200, freq="15min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "ac_power_kw": np.abs(rng.normal(3, 1, 200)),
            "irradiance_wm2": np.abs(rng.normal(400, 200, 200)),
            "temperature_c": rng.normal(25, 5, 200),
            "wind_speed_ms": np.abs(rng.normal(3, 1, 200)),
        }
    )


def test_build_features_columns(sample_df):
    out = build_features(sample_df)
    for col in FEATURE_COLS:
        assert col in out.columns, f"Missing feature column: {col}"


def test_cyclic_encoding_range(sample_df):
    out = build_features(sample_df)
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()
    assert out["doy_sin"].between(-1, 1).all()


def test_get_X_y_no_nan(sample_df):
    X, y = get_X_y(sample_df)
    assert not X.isna().any().any(), "X contains NaN"
    assert not y.isna().any(), "y contains NaN"
    assert len(X) == len(y)


def test_get_X_y_feature_count(sample_df):
    X, _ = get_X_y(sample_df)
    assert set(X.columns) == set(FEATURE_COLS)


def test_temporal_split_no_leakage(sample_df):
    train, test = train_test_split_by_date(sample_df, "2023-06-02")
    assert train["timestamp"].max() < pd.Timestamp("2023-06-02")
    assert test["timestamp"].min() >= pd.Timestamp("2023-06-02")
    assert len(train) + len(test) == len(sample_df)


def test_irradiance_squared_correct(sample_df):
    out = build_features(sample_df)
    expected = sample_df["irradiance_wm2"].values ** 2
    np.testing.assert_allclose(out["irradiance_sq"].values, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Feature-set separation tests
# ---------------------------------------------------------------------------


def test_weather_only_excludes_lag_features():
    """WEATHER_ONLY_FEATURES must not contain actual-power lag columns."""
    assert "ac_power_lag1" not in WEATHER_ONLY_FEATURES
    assert "ac_power_lag4" not in WEATHER_ONLY_FEATURES


def test_lag_aware_includes_lag_features():
    """LAG_AWARE_FEATURES must include both lag columns."""
    assert "ac_power_lag1" in LAG_AWARE_FEATURES
    assert "ac_power_lag4" in LAG_AWARE_FEATURES


def test_weather_only_is_subset_of_lag_aware():
    """Every weather-only feature should also appear in lag-aware."""
    assert set(WEATHER_ONLY_FEATURES).issubset(set(LAG_AWARE_FEATURES))


def test_feature_cols_alias_is_lag_aware():
    """FEATURE_COLS backward-compat alias must equal LAG_AWARE_FEATURES."""
    assert FEATURE_COLS == LAG_AWARE_FEATURES


def test_get_feature_cols_weather_only():
    cols = get_feature_cols("weather_only")
    assert "ac_power_lag1" not in cols
    assert "ac_power_lag4" not in cols
    assert "irradiance_wm2" in cols


def test_get_feature_cols_lag_aware():
    cols = get_feature_cols("lag_aware")
    assert "ac_power_lag1" in cols
    assert "ac_power_lag4" in cols


def test_get_X_y_weather_only_mode(sample_df):
    """X from weather_only mode must not contain lag columns."""
    X, y = get_X_y(sample_df, mode="weather_only")
    assert "ac_power_lag1" not in X.columns
    assert "ac_power_lag4" not in X.columns
    assert "irradiance_wm2" in X.columns
    assert len(X) == len(y)


def test_get_X_y_lag_aware_mode(sample_df):
    """X from lag_aware mode must include lag columns."""
    X, y = get_X_y(sample_df, mode="lag_aware")
    assert "ac_power_lag1" in X.columns
    assert "ac_power_lag4" in X.columns
    assert len(X) == len(y)


def test_weather_only_returns_fewer_features(sample_df):
    X_weather, _ = get_X_y(sample_df, mode="weather_only")
    X_lag, _ = get_X_y(sample_df, mode="lag_aware")
    assert X_weather.shape[1] < X_lag.shape[1]


def test_get_X_y_default_is_lag_aware(sample_df):
    """Default mode should be lag_aware (backward compatibility)."""
    X_default, _ = get_X_y(sample_df)
    X_lag, _ = get_X_y(sample_df, mode="lag_aware")
    assert list(X_default.columns) == list(X_lag.columns)
