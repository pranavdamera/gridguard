"""Tests for the offline research diagnostics.

These cover the analyses migrated out of the retired Streamlit dashboard. The
dashboard had no tests at all, which is part of why it was retired; the
capability is only worth preserving if it is pinned down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridguard.analysis import (
    clear_sky_projection,
    feature_importance,
    power_curve,
    power_curve_summary,
)
from gridguard.sites.registry import get_site


@pytest.fixture
def fitted_model(detected_frame):
    """A cheap weather-only model, matching the convention in conftest.py."""
    from sklearn.ensemble import RandomForestRegressor

    from gridguard.features.engineer import get_X_y

    X, y = get_X_y(detected_frame, mode="weather_only")
    model = RandomForestRegressor(n_estimators=25, max_depth=8, random_state=0, n_jobs=1)
    model.fit(X, y)
    return model, X


@pytest.fixture
def detected_frame() -> pd.DataFrame:
    """A detected-style frame with a clean irradiance/power relationship."""
    rng = np.random.default_rng(0)
    n = 600
    index = pd.date_range("2016-11-01", periods=n, freq="15min")
    irradiance = np.clip(rng.uniform(0, 1000, n), 0, None)
    power = irradiance * 0.2 + rng.normal(0, 2, n)

    # A block of genuine underperformance, flagged.
    flagged = np.zeros(n, dtype=bool)
    flagged[200:260] = True
    power[200:260] *= 0.4

    return pd.DataFrame(
        {
            "timestamp": index,
            "site_id": "nist_roof",
            "irradiance_wm2": irradiance,
            "temperature_c": rng.uniform(5, 30, n),
            "wind_speed_ms": rng.uniform(0, 6, n),
            "ac_power_kw": np.clip(power, 0, None),
            "predicted_kw": np.clip(irradiance * 0.2, 0, None),
            "lost_energy_kwh": np.where(flagged, 3.0, 0.0),
            "is_anomaly": flagged,
        }
    )


# ---------------------------------------------------------------------------
# power_curve
# ---------------------------------------------------------------------------


def test_power_curve_drops_night_intervals(detected_frame):
    curve = power_curve(detected_frame)
    assert (curve["irradiance_wm2"] > 50).all()
    assert len(curve) < len(detected_frame)


def test_power_curve_keeps_night_when_asked(detected_frame):
    curve = power_curve(detected_frame, daylight_only=False)
    assert len(curve) == len(detected_frame)


def test_power_curve_requires_its_inputs():
    with pytest.raises(ValueError, match="irradiance_wm2"):
        power_curve(pd.DataFrame({"ac_power_kw": [1.0]}))


def test_power_curve_tolerates_a_frame_with_no_anomaly_column(detected_frame):
    curve = power_curve(detected_frame.drop(columns="is_anomaly"))
    assert not curve["is_anomaly"].any()


def test_power_curve_summary_separates_flagged_from_healthy(detected_frame):
    """The whole point of the plot: flagged intervals sit below healthy ones.

    If this stops holding, either the detector or the diagnostic is broken —
    and a scatter nobody tests would not have told us which.
    """
    summary = power_curve_summary(power_curve(detected_frame))
    populated = summary[summary["n_anomalous"] > 0].dropna(subset=["median_gap_kw"])

    assert not populated.empty, "fixture should produce flagged intervals in some bins"
    assert (populated["median_gap_kw"] > 0).all(), (
        "flagged intervals should have lower median output than healthy ones "
        f"in the same irradiance band:\n{populated}"
    )


def test_power_curve_summary_handles_empty_input():
    summary = power_curve_summary(pd.DataFrame(columns=["irradiance_wm2", "ac_power_kw"]))
    assert summary.empty
    assert "median_gap_kw" in summary.columns


# ---------------------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------------------


def test_feature_importance_ranks_irradiance_first(fitted_model):
    """Power is driven by irradiance here by construction, so it must lead."""
    model, X = fitted_model

    importance = feature_importance(model, X)
    assert list(importance.columns) == ["feature", "importance", "method"]
    assert importance["importance"].is_monotonic_decreasing
    assert "irradiance" in importance["feature"].iloc[0]


def test_feature_importance_rejects_models_it_cannot_explain(fitted_model):
    _, X = fitted_model

    class Opaque:
        pass

    import gridguard.explainability.shap_explain as se

    original = se.SHAP_AVAILABLE
    se.SHAP_AVAILABLE = False
    try:
        with pytest.raises(TypeError, match="feature_importances_"):
            feature_importance(Opaque(), X)
    finally:
        se.SHAP_AVAILABLE = original


# ---------------------------------------------------------------------------
# clear_sky_projection
# ---------------------------------------------------------------------------


def test_clear_sky_projection_is_a_daily_arc(fitted_model):
    """Night is zero, midday is not — the basic shape of a solar day."""
    model, _ = fitted_model
    site = get_site("nist_roof")
    projection = clear_sky_projection(site, model, start=pd.Timestamp("2016-06-15"), periods=96)

    assert len(projection) == 96
    assert (projection["projected_kw"] >= 0).all()

    midnight = projection.iloc[0]["projected_kw"]
    midday = projection.iloc[48]["projected_kw"]
    assert midday > midnight
    assert projection["irradiance_wm2"].max() > 100


def test_clear_sky_projection_carries_its_assumptions(fitted_model):
    """A ceiling reported without its assumptions is a number that misleads."""
    model, _ = fitted_model

    projection = clear_sky_projection(
        get_site("nist_roof"), model, start=pd.Timestamp("2016-06-15"), periods=8
    )
    assumptions = projection.attrs["assumptions"]
    assert "pvlib" in assumptions["irradiance"]
    assert "not a forecast" in assumptions["temperature"].lower()
    assert "not a weather-aware forecast" in assumptions["interpretation"]


def test_clear_sky_projection_uses_pvlib_not_hand_rolled_geometry():
    """Guards against the dashboard's inline solar geometry creeping back in.

    Summer solar noon must deliver materially more plane-of-array irradiance
    than the winter equivalent at this latitude. The retired implementation
    approximated this; the shared pvlib model is the single source of truth for
    it now, and the two must not silently diverge again.
    """
    from gridguard.data.synthetic import _clear_sky_poa

    site = get_site("nist_roof")
    summer = pd.date_range("2016-06-21 12:00", periods=1, freq="15min")
    winter = pd.date_range("2016-12-21 12:00", periods=1, freq="15min")

    _, summer_poa = _clear_sky_poa(summer, site)
    _, winter_poa = _clear_sky_poa(winter, site)

    assert summer_poa[0] > 0
    assert winter_poa[0] >= 0
    assert summer_poa[0] > winter_poa[0]
