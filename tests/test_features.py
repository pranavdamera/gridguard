"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from gridguard.features.engineer import (
    FEATURE_COLS,
    build_features,
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
