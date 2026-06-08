"""
Anomaly detection: flag intervals where actual generation is meaningfully
below the model's expected generation.

Method: residual thresholding with training-set calibration
  residual = actual - predicted
  An interval is anomalous if residual < -k * sigma(residuals on TRAINING data)
  where sigma is estimated per-hour bucket (irradiance-conditioned).

Calibration:
  Residual statistics are computed from the TRAINING split during `run_training()`
  and saved to artifacts/models/residual_stats.json.  detect_anomalies() loads
  this file by default so test/live data statistics never contaminate thresholds.
  Pass residual_stats=None and freeze=False only in offline batch analysis where
  you want to re-calibrate from the current dataset.

Why per-hour sigma?
  Absolute residuals are much larger at peak irradiance than at dawn/dusk.
  Normalising by hour of day prevents false positives at peak and misses at margins.

Output columns added to df:
  predicted_kw      — model forecast
  residual_kw       — actual minus predicted
  residual_sigma    — normalised residual (z-score)
  is_anomaly        — bool flag
  lost_energy_kwh   — estimated lost generation this interval (15-min = /4 hours)

TODO: Replace sigma thresholding with a proper conformal prediction interval
      for better coverage guarantees.
TODO: Add DBSCAN spatial clustering to group consecutive anomaly intervals into events.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.config import settings
from gridguard.features.engineer import WEATHER_ONLY_FEATURES, build_features

# Anomaly detection always uses weather-only features.
# Lag features must never appear here — a degraded system produces low lags,
# which would cause a lag-aware model to predict low output as "normal" and
# hide persistent underperformance. See docs/methodology.md for details.
ANOMALY_FEATURE_COLS = WEATHER_ONLY_FEATURES

logger = logging.getLogger(__name__)

INTERVAL_HOURS = 0.25  # 15-minute intervals
_RESIDUAL_STATS_FILENAME = "residual_stats.json"


def detect_anomalies(
    df: pd.DataFrame,
    model,
    threshold_sigma: float | None = None,
    residual_stats: dict | None = None,
    freeze: bool = True,
) -> pd.DataFrame:
    """
    Flag anomalous intervals in df.

    Args:
        df:              DataFrame with raw columns (timestamp, ac_power_kw, …)
        model:           Fitted forecasting model
        threshold_sigma: Alert if residual < -threshold_sigma * sigma (default from settings)
        residual_stats:  Pre-computed per-hour (mean, std) dict.
                         - If None and freeze=True (default): loads from artifacts/models/residual_stats.json
                         - If None and freeze=False: computes from df itself (use only for offline analysis)
                         - If provided: used as-is
        freeze:          When True and residual_stats is None, loads the saved training-set stats.
                         Set False only when you intentionally want to recalibrate from df.

    Returns:
        df with anomaly columns added.
    """
    threshold_sigma = threshold_sigma or settings.anomaly_threshold_sigma
    df = build_features(df.copy(), include_lags=False)

    present_cols = [c for c in ANOMALY_FEATURE_COLS if c in df.columns]
    df["predicted_kw"] = np.clip(model.predict(df[present_cols]), 0, None)

    # Only evaluate during daylight (irradiance > 50 W/m² avoids dawn noise)
    daylight = df["irradiance_wm2"] > 50

    df["residual_kw"] = df["ac_power_kw"] - df["predicted_kw"]
    df["residual_sigma"] = np.nan

    if residual_stats is None:
        if freeze:
            residual_stats = _load_residual_stats()
        if residual_stats is None:
            # Fallback: compute from df (warn so the caller knows)
            logger.warning(
                "No saved residual_stats found. Computing from current data — "
                "anomaly thresholds may be contaminated by test data. "
                "Run 'make train' to generate a proper calibration artifact."
            )
            residual_stats = _compute_residual_stats(df[daylight])

    df.loc[daylight, "residual_sigma"] = df.loc[daylight].apply(
        lambda row: _normalise_residual(row["residual_kw"], row["hour"], residual_stats),
        axis=1,
    )

    df["is_anomaly"] = daylight & (df["residual_sigma"] < -threshold_sigma)

    # Estimated lost energy: deficit × interval duration
    df["lost_energy_kwh"] = np.where(
        df["is_anomaly"],
        np.maximum(0, df["predicted_kw"] - df["ac_power_kw"]) * INTERVAL_HOURS,
        0.0,
    )

    logger.info(
        "Anomaly detection complete: %d / %d daylight intervals flagged (%.1f%%)",
        df["is_anomaly"].sum(),
        daylight.sum(),
        100 * df["is_anomaly"].sum() / max(1, daylight.sum()),
    )
    return df


def compute_daily_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate anomaly stats by calendar day."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    agg = (
        df.groupby("date")
        .agg(
            actual_kwh=("ac_power_kw", lambda s: s.sum() * INTERVAL_HOURS),
            predicted_kwh=("predicted_kw", lambda s: s.sum() * INTERVAL_HOURS),
            lost_energy_kwh=("lost_energy_kwh", "sum"),
            anomaly_count=("is_anomaly", "sum"),
        )
        .reset_index()
    )
    agg["loss_pct"] = (
        agg["lost_energy_kwh"] / agg["predicted_kwh"].replace(0, np.nan) * 100
    ).round(1)
    return agg


# ---------------------------------------------------------------------------
# Residual stats helpers
# ---------------------------------------------------------------------------


def compute_and_save_residual_stats(
    train_df: pd.DataFrame,
    model,
    model_dir: Path | None = None,
) -> dict:
    """Compute per-hour residual statistics from the TRAINING split and save to JSON.

    Called by run_training() after the model is fitted.  The resulting JSON
    is the calibration artifact — detect_anomalies() loads it to avoid using
    test-data statistics.

    Returns the stats dict so callers can inspect it immediately.
    """
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_feat = build_features(train_df.copy(), include_lags=False)
    present_cols = [c for c in ANOMALY_FEATURE_COLS if c in train_feat.columns]
    train_feat["predicted_kw"] = np.clip(model.predict(train_feat[present_cols]), 0, None)
    train_feat["residual_kw"] = train_feat["ac_power_kw"] - train_feat["predicted_kw"]

    daylight = train_feat["irradiance_wm2"] > 50
    stats = _compute_residual_stats(train_feat[daylight])

    out_path = model_dir / _RESIDUAL_STATS_FILENAME
    # JSON keys must be strings; convert int hour keys
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)

    logger.info("Saved residual calibration stats → %s", out_path)
    return stats


def _load_residual_stats(model_dir: Path | None = None) -> dict | None:
    """Load residual stats JSON from the artifact directory. Returns None if not found."""
    path = Path(model_dir or settings.model_dir) / _RESIDUAL_STATS_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    # Restore int keys
    return {int(k): v for k, v in raw.items()}


def _compute_residual_stats(df: pd.DataFrame) -> dict:
    """Compute per-integer-hour residual mean and std."""
    df = df.copy()
    df["hour_int"] = df["timestamp"].dt.hour
    stats = df.groupby("hour_int")["residual_kw"].agg(["mean", "std"]).to_dict("index")
    return stats


def _normalise_residual(residual: float, hour: float, stats: dict) -> float:
    hour_int = int(hour)
    s = stats.get(hour_int, {"mean": 0, "std": 1})
    std = s["std"] if s["std"] > 1e-6 else 1.0
    return (residual - s["mean"]) / std
