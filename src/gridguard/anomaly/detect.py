"""
Anomaly detection: flag intervals where generation falls outside the calibrated
range of healthy behaviour.

Two detectors, one default
--------------------------
``conformal`` (default)
    Flags an interval when actual generation falls below a split-conformal
    lower bound calibrated per irradiance bucket. Carries a distribution-free
    coverage guarantee: at most ``alpha`` of healthy intervals should breach it.
    See :mod:`gridguard.anomaly.conformal`.

``sigma`` (benchmark)
    The original detector: flag when the residual falls below ``-k`` per-hour
    standard deviations. Retained deliberately — it is the baseline the
    conformal detector has to beat, and reporting both keeps that comparison
    honest rather than asserting the upgrade was worthwhile.

Both are calibrated on **healthy training intervals only**. Using evaluation-
period data to set thresholds would let a degraded test period widen the very
bounds meant to detect it.

Feature discipline
------------------
Detection always runs on the weather-only model. Lag features must never appear
here: a system that has been degraded for a week produces low output, so its
lagged power is low, so a lag-aware model predicts low output and calls the
degradation normal. The failure is silent and total, which is exactly why the
constraint is enforced in code rather than left to convention.

Output columns
--------------
``predicted_kw``       Expected generation from the weather-only model.
``expected_lower_kw``  Lower edge of the calibrated healthy range.
``expected_upper_kw``  Upper edge (context; over-production is not a fault).
``residual_kw``        actual - predicted.
``residual_sigma``     Residual in per-hour standard deviations (sigma detector).
``is_anomaly``         Whether the interval breached the lower bound.
``lost_energy_kwh``    Estimated energy shortfall for the interval.
``detection_method``   Which detector produced the flag.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.anomaly.conformal import ConformalCalibration
from gridguard.config import settings
from gridguard.features.engineer import WEATHER_ONLY_FEATURES, build_features

logger = logging.getLogger(__name__)

#: Detection uses weather-only features. See the module docstring.
ANOMALY_FEATURE_COLS = WEATHER_ONLY_FEATURES

INTERVAL_HOURS = 0.25  # 15-minute intervals
DAYLIGHT_IRRADIANCE_WM2 = 50  # below this there is no generation to assess

_RESIDUAL_STATS_FILENAME = "residual_stats.json"
_CONFORMAL_FILENAME = "conformal_calibration.json"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_anomalies(
    df: pd.DataFrame,
    model,
    *,
    method: str | None = None,
    calibration: ConformalCalibration | None = None,
    residual_stats: dict | None = None,
    threshold_sigma: float | None = None,
    model_dir: Path | None = None,
    freeze: bool = True,
) -> pd.DataFrame:
    """Flag anomalous intervals.

    Args:
        df:              Raw canonical telemetry (timestamp, ac_power_kw, weather).
        model:           Fitted weather-only expected-generation model.
        method:          ``"conformal"`` (default) or ``"sigma"``.
        calibration:     Pre-loaded conformal calibration. Loaded from the
                         artifact directory when omitted.
        residual_stats:  Pre-computed per-hour residual statistics for the sigma
                         detector. Loaded from the artifact directory when omitted.
        threshold_sigma: Sigma-detector threshold. Defaults to settings.
        model_dir:       Artifact directory for calibration lookup.
        freeze:          When True (default), use saved training-split
                         calibration. Set False only for offline analysis where
                         recalibrating from the supplied frame is intended.

    Returns:
        ``df`` with the anomaly columns described in the module docstring.
    """
    method = (method or settings.anomaly_method).lower()
    if method not in ("conformal", "sigma"):
        raise ValueError(f"Unknown detection method '{method}'. Use 'conformal' or 'sigma'.")

    model_dir = Path(model_dir or settings.model_dir)
    out = build_features(df.copy(), include_lags=False)

    present = [c for c in ANOMALY_FEATURE_COLS if c in out.columns]
    out["predicted_kw"] = np.clip(model.predict(out[present]), 0, None)
    out["residual_kw"] = out["ac_power_kw"] - out["predicted_kw"]

    daylight = out["irradiance_wm2"] > DAYLIGHT_IRRADIANCE_WM2

    # Sigma statistics are always computed: even when conformal is driving
    # detection, residual_sigma is a useful severity signal for ranking events.
    if residual_stats is None and freeze:
        residual_stats = _load_residual_stats(model_dir)
    if residual_stats is None:
        if freeze:
            logger.warning(
                "No saved residual_stats found in %s. Computing from the supplied data — "
                "thresholds may be contaminated by evaluation data. Run the artifact build "
                "to generate a proper calibration.",
                model_dir,
            )
        residual_stats = _compute_residual_stats(out[daylight])

    out["residual_sigma"] = np.nan
    if daylight.any():
        out.loc[daylight, "residual_sigma"] = _normalise_residuals(
            out.loc[daylight, "residual_kw"].to_numpy(),
            out.loc[daylight, "hour"].to_numpy(),
            residual_stats,
        )

    if method == "conformal":
        if calibration is None and freeze:
            calibration = ConformalCalibration.load(model_dir / _CONFORMAL_FILENAME)
        if calibration is None:
            logger.warning(
                "No conformal calibration found in %s; falling back to the sigma detector. "
                "Run the artifact build to generate one.",
                model_dir,
            )
            method = "sigma"

    if method == "conformal":
        irradiance = out["irradiance_wm2"].to_numpy(dtype=float)
        lower = calibration.lower_bounds_for_array(irradiance)
        upper = calibration.upper_bounds_for_array(irradiance)
        out["expected_lower_kw"] = np.clip(out["predicted_kw"].to_numpy() + lower, 0, None)
        out["expected_upper_kw"] = np.clip(out["predicted_kw"].to_numpy() + upper, 0, None)
        out["is_anomaly"] = daylight & (out["ac_power_kw"] < out["expected_lower_kw"])
    else:
        threshold_sigma = threshold_sigma or settings.anomaly_threshold_sigma
        sigma_by_hour = _sigma_lookup(out["hour"].to_numpy(), residual_stats)
        out["expected_lower_kw"] = np.clip(
            out["predicted_kw"].to_numpy() - threshold_sigma * sigma_by_hour, 0, None
        )
        out["expected_upper_kw"] = np.clip(
            out["predicted_kw"].to_numpy() + threshold_sigma * sigma_by_hour, 0, None
        )
        out["is_anomaly"] = daylight & (out["residual_sigma"] < -threshold_sigma)

    out["detection_method"] = method
    out["is_anomaly"] = out["is_anomaly"].fillna(False).astype(bool)

    out["lost_energy_kwh"] = np.where(
        out["is_anomaly"],
        np.maximum(0.0, out["predicted_kw"] - out["ac_power_kw"]) * INTERVAL_HOURS,
        0.0,
    )

    flagged = int(out["is_anomaly"].sum())
    total_daylight = int(daylight.sum())
    logger.info(
        "Anomaly detection (%s): %d / %d daylight intervals flagged (%.2f%%).",
        method,
        flagged,
        total_daylight,
        100 * flagged / max(1, total_daylight),
    )
    return out


def compute_daily_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate anomaly statistics by calendar day."""
    out = df.copy()
    out["date"] = out["timestamp"].dt.date
    agg = (
        out.groupby("date")
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
# Calibration fitting
# ---------------------------------------------------------------------------


def fit_calibrations(
    healthy_train_df: pd.DataFrame,
    model,
    *,
    alpha: float | None = None,
    model_dir: Path | None = None,
    save: bool = True,
) -> tuple[ConformalCalibration, dict]:
    """Fit and persist both calibrations from healthy intervals.

    Returns ``(conformal_calibration, residual_stats)``.

    Both detectors are calibrated from the same intervals so that the comparison
    between them isolates the method rather than the data.

    **Pass held-out data.** Split conformal assumes the calibration residuals are
    exchangeable with future residuals, which in-sample residuals are not: a
    model's errors on its own training data are biased small, so the resulting
    bounds are too tight and real coverage falls below the stated guarantee.
    Callers should pass a validation split the model never saw — the build
    pipeline uses the chronological tail of the training window, which is also
    the closest available data to the evaluation period.
    """
    alpha = alpha if alpha is not None else settings.conformal_alpha
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(healthy_train_df.copy(), include_lags=False)
    present = [c for c in ANOMALY_FEATURE_COLS if c in features.columns]
    features["predicted_kw"] = np.clip(model.predict(features[present]), 0, None)
    features["residual_kw"] = features["ac_power_kw"] - features["predicted_kw"]

    daylight = features["irradiance_wm2"] > DAYLIGHT_IRRADIANCE_WM2
    calibration_rows = features[daylight]
    if calibration_rows.empty:
        raise ValueError("No daylight intervals available to calibrate the detector.")

    conformal = ConformalCalibration.fit(
        residuals=calibration_rows["residual_kw"].to_numpy(),
        irradiance_wm2=calibration_rows["irradiance_wm2"].to_numpy(),
        alpha=alpha,
        timestamps=calibration_rows["timestamp"],
    )
    residual_stats = _compute_residual_stats(calibration_rows)

    if save:
        conformal.save(model_dir / _CONFORMAL_FILENAME)
        _save_residual_stats(residual_stats, model_dir)

    return conformal, residual_stats


def compute_and_save_residual_stats(
    train_df: pd.DataFrame,
    model,
    model_dir: Path | None = None,
) -> dict:
    """Compute and persist per-hour residual statistics for the sigma detector."""
    _, residual_stats = fit_calibrations(train_df, model, model_dir=model_dir, save=True)
    return residual_stats


# ---------------------------------------------------------------------------
# Residual statistics helpers
# ---------------------------------------------------------------------------


def _save_residual_stats(stats: dict, model_dir: Path) -> Path:
    path = Path(model_dir) / _RESIDUAL_STATS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in stats.items()}, indent=2) + "\n")
    logger.info("Saved residual calibration stats -> %s", path)
    return path


def _load_residual_stats(model_dir: Path | None = None) -> dict | None:
    path = Path(model_dir or settings.model_dir) / _RESIDUAL_STATS_FILENAME
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def _compute_residual_stats(df: pd.DataFrame) -> dict:
    """Per-integer-hour residual mean and standard deviation."""
    frame = df.copy()
    frame["hour_int"] = frame["timestamp"].dt.hour
    return frame.groupby("hour_int")["residual_kw"].agg(["mean", "std"]).to_dict("index")


def _sigma_lookup(hours: np.ndarray, stats: dict) -> np.ndarray:
    """Per-hour residual standard deviation, vectorised with a safe fallback."""
    out = np.ones(len(hours), dtype=float)
    for i, hour in enumerate(hours):
        entry = stats.get(int(hour)) if np.isfinite(hour) else None
        std = entry.get("std") if entry else None
        out[i] = std if std and std > 1e-6 else 1.0
    return out


def _normalise_residuals(residuals: np.ndarray, hours: np.ndarray, stats: dict) -> np.ndarray:
    """Convert residuals to per-hour z-scores."""
    means = np.zeros(len(hours), dtype=float)
    for i, hour in enumerate(hours):
        entry = stats.get(int(hour)) if np.isfinite(hour) else None
        mean = entry.get("mean") if entry else None
        means[i] = mean if mean is not None and np.isfinite(mean) else 0.0
    return (residuals - means) / _sigma_lookup(hours, stats)
