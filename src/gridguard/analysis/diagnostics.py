"""
Per-site research diagnostics.

These four analyses were the only content in the retired Streamlit dashboard
that did not already exist elsewhere in the project. They are reproduced here as
tested functions so the capability survives the surface being removed.

What was actually unique, and what was not
------------------------------------------
Most of the dashboard duplicated the API: actual-vs-predicted charts, the event
queue, model metrics and the fleet map all have equivalents in ``api/main.py``
and the web application. Two capabilities existed in the package already but
were surfaced nowhere else — :func:`gridguard.anomaly.detect.compute_daily_loss`
and :func:`gridguard.explainability.shap_explain.global_feature_importance` —
so they are wired up here rather than reimplemented.

Genuinely absent from the rest of the project, and therefore new code:

``power_curve``
    The power-versus-irradiance characteristic, with detected anomalies
    overlaid. This is the standard PV diagnostic scatter and nothing else in the
    project renders it. Underperformance shows up as points sitting below the
    main locus at a given irradiance, which is a different and more direct view
    than a time series.

``clear_sky_projection``
    A forward projection of generation under clear-sky conditions. The dashboard
    computed this from hand-rolled solar geometry — declination, hour angle and
    a cosine-of-zenith approximation written out inline. That duplicated, less
    accurate than, and silently divergent from the pvlib Ineichen model the rest
    of the project already uses, so it is rebuilt on
    :func:`gridguard.data.synthetic.clear_sky_poa`'s approach instead. The
    numbers will not match the dashboard's, and should not: these are better.

The projection is a **physical upper bound under clear skies**, not a
weather-aware forecast. It answers "what could this site produce tomorrow if
nothing were in the way", which is a ceiling to compare against, not a
prediction of what it will produce.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gridguard.sites.registry import Site

logger = logging.getLogger(__name__)

#: Irradiance below which an interval is night, not underperformance.
DAYLIGHT_THRESHOLD_WM2 = 50.0


# ---------------------------------------------------------------------------
# Power curve
# ---------------------------------------------------------------------------


def power_curve(detected: pd.DataFrame, *, daylight_only: bool = True) -> pd.DataFrame:
    """Power-versus-irradiance pairs, with anomaly labels.

    Args:
        detected: output of :func:`gridguard.anomaly.detect.detect_anomalies`.
        daylight_only: drop night intervals, where the pairs pile up at the
            origin and carry no diagnostic information.

    Returns:
        Columns ``irradiance_wm2``, ``ac_power_kw``, ``is_anomaly``,
        ``timestamp``.
    """
    required = {"irradiance_wm2", "ac_power_kw"}
    missing = required - set(detected.columns)
    if missing:
        raise ValueError(f"power_curve needs columns {sorted(missing)}")

    out = detected.copy()
    if daylight_only:
        out = out[out["irradiance_wm2"] > DAYLIGHT_THRESHOLD_WM2]

    keep = [c for c in ("timestamp", "irradiance_wm2", "ac_power_kw", "is_anomaly") if c in out]
    out = out[keep].copy()
    if "is_anomaly" not in out.columns:
        out["is_anomaly"] = False
    out["is_anomaly"] = out["is_anomaly"].fillna(False).astype(bool)
    return out.reset_index(drop=True)


def power_curve_summary(curve: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Bin a power curve by irradiance and summarise each bin.

    Gives the report a table rather than only a picture: for each irradiance
    band, the median output of healthy intervals, the median output of flagged
    intervals, and how far apart they are. A detector that is working separates
    those two medians; one that is firing at random does not.
    """
    if curve.empty:
        return pd.DataFrame(
            columns=[
                "irradiance_bin",
                "n_total",
                "n_anomalous",
                "healthy_median_kw",
                "anomalous_median_kw",
                "median_gap_kw",
            ]
        )

    binned = curve.copy()
    binned["irradiance_bin"] = pd.cut(binned["irradiance_wm2"], bins=n_bins)

    rows = []
    for label, group in binned.groupby("irradiance_bin", observed=True):
        healthy = group.loc[~group["is_anomaly"], "ac_power_kw"]
        flagged = group.loc[group["is_anomaly"], "ac_power_kw"]
        healthy_median = float(healthy.median()) if len(healthy) else float("nan")
        flagged_median = float(flagged.median()) if len(flagged) else float("nan")
        rows.append(
            {
                "irradiance_bin": str(label),
                "n_total": int(len(group)),
                "n_anomalous": int(group["is_anomaly"].sum()),
                "healthy_median_kw": healthy_median,
                "anomalous_median_kw": flagged_median,
                "median_gap_kw": healthy_median - flagged_median,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance(model, X: pd.DataFrame, *, max_samples: int = 2000) -> pd.DataFrame:
    """Global feature importance by mean absolute SHAP value.

    The dashboard used XGBoost's built-in ``feature_importances_``, which is
    gain-based: it reports how much each feature improved the split criterion
    during training. Mean absolute SHAP measures how much each feature actually
    moves predictions on the data supplied, which is the question a reader of
    the report is asking, and it is consistent with the per-event attribution
    the API already returns. The project depends on ``shap`` either way.

    Falls back to gain-based importance if SHAP is unavailable, and says so in
    the ``method`` column rather than silently returning different numbers under
    the same name.
    """
    from gridguard.explainability.shap_explain import (
        SHAP_AVAILABLE,
        compute_shap_values,
        global_feature_importance,
    )

    sample = X.sample(min(max_samples, len(X)), random_state=0) if len(X) > max_samples else X

    if SHAP_AVAILABLE:
        shap_values, names = compute_shap_values(model, sample)
        out = global_feature_importance(shap_values, names)
        out["method"] = "mean_abs_shap"
        out = out.rename(columns={"mean_abs_shap": "importance"})
        return out[["feature", "importance", "method"]]

    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"{type(model).__name__} exposes neither SHAP support nor "
            "feature_importances_; no importance can be computed."
        )
    logger.warning("shap unavailable — falling back to gain-based feature importance.")
    out = pd.DataFrame(
        {"feature": list(X.columns), "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    out["method"] = "xgboost_gain"
    return out


# ---------------------------------------------------------------------------
# Clear-sky projection
# ---------------------------------------------------------------------------


def clear_sky_projection(
    site: Site,
    model,
    *,
    start: pd.Timestamp,
    periods: int = 96,
    freq: str = "15min",
    wind_speed_ms: float = 3.5,
) -> pd.DataFrame:
    """Project generation forward under clear skies.

    Uses the same pvlib Ineichen clear-sky model and plane-of-array
    transposition as the rest of the project, so the projection is consistent
    with the physics baseline rather than being a second, divergent solar model.

    Ambient temperature is the one input with no physical model here; it is
    carried from a seasonal climatology so the projection is not silently
    assuming a fixed temperature. That approximation is the projection's main
    limitation and is reported in the ``assumptions`` attribute of the result.

    Args:
        site: the site to project for. Needs tilt and azimuth for a meaningful
            plane-of-array figure; falls back to the project defaults if absent.
        model: a fitted weather-only model.
        start: first interval of the projection.
        periods: number of intervals.
        freq: interval spacing.

    Returns:
        Columns ``timestamp``, ``irradiance_wm2``, ``temperature_c``,
        ``wind_speed_ms``, ``projected_kw``. Carries an ``attrs["assumptions"]``
        dict describing what was assumed.
    """
    from gridguard.data.synthetic import clear_sky_poa
    from gridguard.features.engineer import get_X_y

    index = pd.date_range(start, periods=periods, freq=freq)
    _, poa = clear_sky_poa(index, site)

    day_of_year = index.day_of_year.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    temperature = (
        16.0
        + 11.0 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 80)))
        + 5.0 * np.sin(np.radians(15.0 * (hour - 14)))
    )

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "site_id": site.site_id,
            "irradiance_wm2": np.round(poa, 2),
            "temperature_c": np.round(temperature, 2),
            "wind_speed_ms": np.full(periods, wind_speed_ms),
            # get_X_y needs a target column present; it is not read for
            # prediction and the projected values overwrite nothing.
            "ac_power_kw": np.zeros(periods),
        }
    )

    X, _ = get_X_y(frame, mode="weather_only")
    frame["projected_kw"] = np.clip(model.predict(X), 0.0, None)

    out = frame[
        ["timestamp", "irradiance_wm2", "temperature_c", "wind_speed_ms", "projected_kw"]
    ].copy()
    out.attrs["assumptions"] = {
        "irradiance": "pvlib Ineichen clear-sky, transposed to the array plane (isotropic).",
        "tilt_deg": site.tilt_deg,
        "azimuth_deg": site.azimuth_deg,
        "temperature": "Seasonal + diurnal climatology, not a forecast.",
        "wind_speed_ms": wind_speed_ms,
        "interpretation": (
            "A clear-sky ceiling, not a weather-aware forecast. Actual output "
            "will be at or below this whenever there is any cloud."
        ),
    }
    return out
