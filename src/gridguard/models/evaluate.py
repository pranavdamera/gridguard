"""
Model evaluation: overall and stratified metrics.

Overall metrics:
  MAE   — mean absolute error in kW (easy to communicate to stakeholders)
  RMSE  — penalises large errors more; sensitive to anomaly spikes
  MAPE  — % error; computed on daylight hours only (avoids div-by-zero)
  R²    — how much variance the model explains

Stratified metrics: same metrics broken down by —
  hour_of_day    — identify if model underperforms at dawn/dusk
  season         — winter vs summer performance gap
  irradiance_bin — performance at low, medium, high irradiance
  site_id        — if site_id column present (multi-site mode)

Reports are saved as CSV to artifacts/reports/ for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gridguard.features.engineer import FEATURE_COLS, build_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-split metric computation
# ---------------------------------------------------------------------------


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Return a dict of metrics for a single model on X, y."""
    y_pred = np.clip(model.predict(X), 0, None)  # generation can't be negative
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    r2 = r2_score(y, y_pred)

    # MAPE on daylight hours only (irradiance > threshold avoids div-by-zero)
    daylight_mask = y > 0.1
    if daylight_mask.sum() > 0:
        mape = (
            np.mean(
                np.abs((y[daylight_mask] - y_pred[daylight_mask]) / y[daylight_mask])
            )
            * 100
        )
    else:
        mape = float("nan")

    return {
        "mae_kw": round(mae, 4),
        "rmse_kw": round(rmse, 4),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4),
        "n_samples": int(len(y)),
    }


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute metrics from plain arrays (used inside stratified groupby)."""
    if len(y_true) == 0:
        return {"mae_kw": np.nan, "rmse_kw": np.nan, "mape_pct": np.nan, "r2": np.nan, "n_samples": 0}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    daylight = y_true > 0.1
    mape = (
        float(np.mean(np.abs((y_true[daylight] - y_pred[daylight]) / y_true[daylight])) * 100)
        if daylight.sum() > 0
        else np.nan
    )
    return {
        "mae_kw": round(float(mae), 4),
        "rmse_kw": round(float(rmse), 4),
        "mape_pct": round(float(mape), 2) if not np.isnan(mape) else np.nan,
        "r2": round(float(r2), 4) if not np.isnan(r2) else np.nan,
        "n_samples": int(len(y_true)),
    }


def compare_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return a comparison DataFrame with one row per model, sorted by RMSE."""
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows).sort_values("rmse_kw")
    return df


def add_predictions(df: pd.DataFrame, model, model_name: str = "predicted") -> pd.DataFrame:
    """Append predicted generation column to df."""
    df = build_features(df.copy())
    present_cols = [c for c in FEATURE_COLS if c in df.columns]
    preds = np.clip(model.predict(df[present_cols]), 0, None)
    df[f"{model_name}_kw"] = preds
    return df


# ---------------------------------------------------------------------------
# Stratified evaluation helpers
# ---------------------------------------------------------------------------


def _add_strata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate df with strata columns used for breakdowns."""
    df = df.copy()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["_hour"] = ts.dt.hour
        month = ts.dt.month
        df["_season"] = pd.cut(
            month,
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Fall"],
            right=True,
        )

    if "irradiance_wm2" in df.columns:
        df["_irr_bin"] = pd.cut(
            df["irradiance_wm2"],
            bins=[-1, 50, 300, 600, 1200],
            labels=["Night/Dawn", "Low", "Medium", "High"],
        )

    return df


def stratified_metrics(
    df: pd.DataFrame,
    model,
    strata: str = "hour_of_day",
) -> pd.DataFrame:
    """Compute metrics broken down by a single stratum.

    Args:
        df:     DataFrame with raw columns (timestamp, ac_power_kw, irradiance_wm2, …)
        model:  Fitted model
        strata: One of "hour_of_day", "season", "irradiance_bin", "site_id"

    Returns:
        DataFrame with one row per stratum value, sorted by the stratum.
    """
    df = build_features(df.copy())
    df = _add_strata_columns(df)

    present_cols = [c for c in FEATURE_COLS if c in df.columns]
    df["_pred"] = np.clip(model.predict(df[present_cols]), 0, None)

    strata_col_map = {
        "hour_of_day": "_hour",
        "season": "_season",
        "irradiance_bin": "_irr_bin",
        "site_id": "site_id",
    }
    if strata not in strata_col_map:
        raise ValueError(f"Unknown strata '{strata}'. Choose from {list(strata_col_map)}")

    col = strata_col_map[strata]
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found — ensure df has the required fields.")

    rows = []
    for val, group in df.groupby(col, observed=True):
        m = _metrics_from_arrays(group["ac_power_kw"].values, group["_pred"].values)
        rows.append({strata: val, **m})

    return pd.DataFrame(rows)


def full_stratified_report(
    df: pd.DataFrame,
    model,
    model_name: str = "model",
    report_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute all stratifications and save CSVs to report_dir.

    Returns dict[strata_name → metrics_df].
    """
    report_dir = Path(report_dir or "artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    strata_list = ["hour_of_day", "season", "irradiance_bin"]
    if "site_id" in df.columns:
        strata_list.append("site_id")

    reports: dict[str, pd.DataFrame] = {}
    for strata in strata_list:
        try:
            sdf = stratified_metrics(df, model, strata=strata)
            reports[strata] = sdf
            out = report_dir / f"{model_name}_{strata}.csv"
            sdf.to_csv(out, index=False)
            logger.info("Saved stratified report: %s", out)
        except Exception as exc:
            logger.warning("Could not compute strata '%s': %s", strata, exc)

    return reports


# Kept for backwards-compatibility (imported in tests)
def get_X_y(df: pd.DataFrame):
    from gridguard.features.engineer import get_X_y as _get_X_y
    return _get_X_y(df)
