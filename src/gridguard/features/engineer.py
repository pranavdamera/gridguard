"""
Feature engineering for solar generation forecasting.

Design principles:
  - All features are physically meaningful (interpretable by domain experts).
  - No target leakage at inference: lag features are ac_power from prior intervals,
    not from the current or future interval.
  - Returns a clean (X, y) pair ready for sklearn/xgboost.

Feature modes
-------------
Two distinct feature sets serve different modelling purposes:

  WEATHER_ONLY_FEATURES  — "what should a healthy system produce?"
    Use this mode to train the expected-generation model used in anomaly detection.
    Contains only weather, time, and site features. Never includes actual-power
    lag features because persistent degradation would be hidden in those lags —
    a degraded system produces low power, so lag1 = low, model learns low is normal.

  LAG_AWARE_FEATURES  — "what will the system produce in the next interval?"
    Use this mode for short-horizon operational forecasting where recent observed
    output is a strong signal (e.g., cloud-ramp prediction). Includes ac_power_lag1
    and ac_power_lag4. Do NOT use for anomaly detection baselines.

Feature groups:
  1. Time features  — hour, day-of-year, month, sin/cos encodings
  2. Weather        — irradiance, temperature, wind speed, irradiance²
  3. Rolling stats  — 1-hour rolling mean irradiance (smooths cloud transients)
  4. Lag features   — previous 15-min and 1-hour ac_power (LAG_AWARE only)

TODO: Add capacity factor (ac_power / system_capacity_kw) as a normalised target.
TODO: Add clearsky ratio feature using pvlib for more physically grounded baseline.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------

#: Weather + time features only. Use for anomaly detection / expected-generation models.
#: Lag features are intentionally excluded to prevent masking persistent degradation.
WEATHER_ONLY_FEATURES: list[str] = [
    # Time (cyclic)
    "hour",
    "day_of_year",
    "month",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    # Weather
    "irradiance_wm2",
    "irradiance_sq",
    "temperature_c",
    "wind_speed_ms",
    # Rolling irradiance (cloud smoothing)
    "irradiance_roll1h",
]

#: Full feature set including recent-output lags. Use for short-horizon forecasting.
#: Do NOT use for anomaly detection baselines; see module docstring.
LAG_AWARE_FEATURES: list[str] = WEATHER_ONLY_FEATURES + [
    "ac_power_lag1",  # previous 15-min interval
    "ac_power_lag4",  # 1 hour ago
]

#: Backward-compatible alias. Points to LAG_AWARE_FEATURES.
FEATURE_COLS = LAG_AWARE_FEATURES

FeatureMode = Literal["weather_only", "lag_aware"]

TARGET_COL = "ac_power_kw"


def get_feature_cols(mode: FeatureMode = "lag_aware") -> list[str]:
    """Return the feature column list for the given mode."""
    if mode == "weather_only":
        return WEATHER_ONLY_FEATURES
    return LAG_AWARE_FEATURES


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
    """Add all engineered feature columns to df in-place. Returns df.

    Always builds both weather and lag columns. Use get_feature_cols() to select
    the appropriate subset when building X for training or inference.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    ts = df["timestamp"]

    # Time
    df["hour"] = ts.dt.hour + ts.dt.minute / 60
    df["day_of_year"] = ts.dt.day_of_year
    df["month"] = ts.dt.month

    # Cyclic encodings — prevent model treating hour 23 as "far" from hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Weather interactions
    df["irradiance_sq"] = df["irradiance_wm2"] ** 2

    # Rolling irradiance (4 × 15min = 1 hour)
    df["irradiance_roll1h"] = (
        df["irradiance_wm2"].rolling(window=4, min_periods=1).mean()
    )

    if include_lags and TARGET_COL in df.columns:
        df["ac_power_lag1"] = df[TARGET_COL].shift(1).fillna(0)
        df["ac_power_lag4"] = df[TARGET_COL].shift(4).fillna(0)
    else:
        df["ac_power_lag1"] = 0.0
        df["ac_power_lag4"] = 0.0

    return df


def get_X_y(
    df: pd.DataFrame,
    mode: FeatureMode = "lag_aware",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with no NaN rows.

    Args:
        df:   Raw DataFrame with at minimum timestamp and ac_power_kw columns.
        mode: "lag_aware" (default) includes ac_power lag features — use for
              short-horizon forecasting. "weather_only" excludes lag features —
              use for anomaly detection / expected-generation models to avoid
              masking persistent degradation.
    """
    include_lags = mode == "lag_aware"
    df = build_features(df, include_lags=include_lags)
    cols = get_feature_cols(mode)
    mask = df[cols].notna().all(axis=1) & df[TARGET_COL].notna()
    X = df.loc[mask, cols]
    y = df.loc[mask, TARGET_COL]
    return X, y


def train_test_split_by_date(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split — train on data before split_date, test on data after.

    Always use temporal splits for time series. Random splits leak future info.
    """
    split = pd.Timestamp(split_date)
    train = df[df["timestamp"] < split].copy()
    test = df[df["timestamp"] >= split].copy()
    return train, test
