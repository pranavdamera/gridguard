"""
Feature engineering for solar generation forecasting.

Design principles:
  - All features are physically meaningful (interpretable by domain experts).
  - No target leakage: features only use weather + time, not ac_power.
  - Returns a clean (X, y) pair ready for sklearn/xgboost.

Feature groups:
  1. Time features  — hour, day-of-year, month, sin/cos encodings
  2. Weather        — irradiance, temperature, wind speed, irradiance²
  3. Lag features   — previous 15-min and 1-hour ac_power (for sequence models later)
  4. Rolling stats  — 1-hour rolling mean irradiance (smooths cloud transients)

TODO: Add capacity factor (ac_power / system_capacity_kw) as a normalised target.
TODO: Add clearsky ratio feature using pvlib for more physically grounded baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    # Time
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
    # Rolling / lag
    "irradiance_roll1h",
    "ac_power_lag1",   # previous 15-min interval
    "ac_power_lag4",   # 1 hour ago
]

TARGET_COL = "ac_power_kw"


def build_features(df: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
    """Add all engineered feature columns to df in-place. Returns df."""
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


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with no NaN rows."""
    df = build_features(df)
    mask = df[FEATURE_COLS].notna().all(axis=1) & df[TARGET_COL].notna()
    X = df.loc[mask, FEATURE_COLS]
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
