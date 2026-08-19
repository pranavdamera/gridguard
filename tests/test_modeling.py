"""
Tests for conformal calibration, the physics baseline, and leakage discipline.

The leakage tests matter most. They encode the single constraint that makes the
detector work at all: the expected-generation model must never see lagged power,
because a system degraded for days produces low output, so its lagged power is
low, so a lag-aware model predicts low output and calls the degradation normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridguard.anomaly.conformal import (
    BUCKET_ORDER,
    GLOBAL_BUCKET,
    MIN_BUCKET_SAMPLES,
    ConformalCalibration,
    compute_conformal_bound,
    compute_conformal_interval,
    irradiance_bucket,
)
from gridguard.features.engineer import (
    LAG_AWARE_FEATURES,
    WEATHER_ONLY_FEATURES,
    get_feature_cols,
    get_X_y,
    train_test_split_by_date,
)
from gridguard.models.physics import PhysicsAssumptions, PhysicsBaseline, PhysicsResidualModel
from gridguard.sites.registry import get_site

# ---------------------------------------------------------------------------
# Leakage discipline
# ---------------------------------------------------------------------------


def test_weather_only_excludes_every_power_lag():
    """The constraint the whole detector rests on."""
    assert not any("ac_power" in feature for feature in WEATHER_ONLY_FEATURES)


def test_lag_aware_is_a_strict_superset():
    assert set(WEATHER_ONLY_FEATURES) < set(LAG_AWARE_FEATURES)
    assert "ac_power_lag1" in LAG_AWARE_FEATURES


def test_get_feature_cols_respects_mode():
    assert get_feature_cols("weather_only") == WEATHER_ONLY_FEATURES
    assert get_feature_cols("lag_aware") == LAG_AWARE_FEATURES


def test_weather_only_X_has_no_lag_columns(synthetic_frame):
    X, _ = get_X_y(synthetic_frame, mode="weather_only")
    assert not [c for c in X.columns if "lag" in c]


def test_lag_features_drag_the_prediction_toward_degraded_output(synthetic_frame):
    """The concrete mechanism the weather-only rule exists to prevent.

    Hold the weather fixed and feed the model the lagged power of a system that
    has been running at half output for days. A lag-aware model follows those
    lags downward — it predicts the degraded output as the expected output,
    which is how a persistent fault becomes invisible to a residual detector.
    The weather-only model cannot do this: those columns are not in its feature
    set at all.

    On the effect size: it is directionally clear but modest here (a few percent),
    because irradiance explains most of the variance in this simulation, leaving
    lagged power little incremental information to contribute. The hazard scales
    with how much output variance the weather channels *fail* to explain — soiling,
    partial shading, inverter derating, sensor drift — which is larger on real
    installations than in simulation. So this test asserts the direction and the
    weather-only model's structural immunity, not a magnitude the data here would
    not support.
    """
    from sklearn.ensemble import RandomForestRegressor

    healthy = synthetic_frame[synthetic_frame["timestamp"] < "2022-05-01"]

    X_lag, y_lag = get_X_y(healthy, mode="lag_aware")
    lag_model = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=0, n_jobs=1)
    lag_model.fit(X_lag, y_lag)

    X_weather, y_weather = get_X_y(healthy, mode="weather_only")
    weather_model = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=0, n_jobs=1)
    weather_model.fit(X_weather, y_weather)

    # Evaluate on generating intervals only; lags are meaningless at night.
    daylight = X_lag["irradiance_wm2"] > 300
    X_lag_eval = X_lag[daylight].copy()
    X_weather_eval = X_weather[daylight].copy()

    healthy_lag_prediction = lag_model.predict(X_lag_eval).mean()
    healthy_weather_prediction = weather_model.predict(X_weather_eval).mean()

    # Same weather, but the system has been degraded for long enough that its
    # recent history reflects the fault.
    X_lag_degraded = X_lag_eval.copy()
    X_lag_degraded["ac_power_lag1"] *= 0.5
    X_lag_degraded["ac_power_lag4"] *= 0.5
    degraded_lag_prediction = lag_model.predict(X_lag_degraded).mean()

    lag_drop = (healthy_lag_prediction - degraded_lag_prediction) / healthy_lag_prediction
    assert lag_drop > 0.005, (
        "Lag-aware expectation should fall when recent output is degraded "
        f"(fell {lag_drop:.2%}); that fall is what hides a persistent fault."
    )

    # The weather-only model is structurally immune: degraded history is not an
    # input it has.
    assert "ac_power_lag1" not in X_weather_eval.columns
    assert weather_model.predict(X_weather_eval).mean() == pytest.approx(healthy_weather_prediction)


def test_weather_only_residual_exposes_a_persistent_fault(synthetic_frame):
    """End to end: a sustained 50% derate stays visible in weather-only residuals."""
    from sklearn.ensemble import RandomForestRegressor

    healthy = synthetic_frame[synthetic_frame["timestamp"] < "2022-05-01"]
    degraded = synthetic_frame[synthetic_frame["timestamp"] >= "2022-05-01"].copy()
    degraded["ac_power_kw"] *= 0.5

    X_train, y_train = get_X_y(healthy, mode="weather_only")
    model = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=0, n_jobs=1)
    model.fit(X_train, y_train)

    X_eval, y_eval = get_X_y(degraded, mode="weather_only")
    daylight = X_eval["irradiance_wm2"] > 300
    residual = y_eval[daylight].to_numpy() - model.predict(X_eval[daylight])

    # The shortfall should be large and consistently negative, not noise.
    assert residual.mean() < 0
    assert (residual < 0).mean() > 0.9


def test_temporal_split_never_reorders_time(synthetic_frame):
    train, test = train_test_split_by_date(synthetic_frame, "2022-04-01")
    assert train["timestamp"].max() < test["timestamp"].min()


def test_temporal_split_partitions_all_rows(synthetic_frame):
    train, test = train_test_split_by_date(synthetic_frame, "2022-04-01")
    assert len(train) + len(test) == len(synthetic_frame)


# ---------------------------------------------------------------------------
# Conformal bounds
# ---------------------------------------------------------------------------


def test_bucketing_is_monotonic_in_irradiance():
    assert irradiance_bucket(0) == "night"
    assert irradiance_bucket(100) == "low"
    assert irradiance_bucket(400) == "medium"
    assert irradiance_bucket(700) == "high"
    assert irradiance_bucket(1000) == "peak"


def test_coverage_guarantee_holds_on_calibration_data():
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(2000)
    for alpha in (0.01, 0.05, 0.10, 0.20):
        bound = compute_conformal_bound(residuals, alpha=alpha)
        coverage = float((residuals >= bound).mean())
        assert coverage >= 1 - alpha, f"alpha={alpha}: coverage {coverage:.3f}"


def test_conformal_is_calibrated_where_a_sigma_rule_is_not():
    """The point of conformal: you get the false-alarm rate you asked for.

    PV residuals are heavy-tailed. On such data a "2 sigma" threshold is badly
    miscalibrated, because the outliers inflate the standard deviation that
    defines it — the bound drifts far into the tail and covers ~99% instead of
    the ~95% the choice of k implies. A detector calibrated that way is not
    conservative in a useful way; it is silently missing real faults.

    The conformal bound targets 95% by construction and lands there, with no
    distributional assumption at all.
    """
    rng = np.random.default_rng(1)
    residuals = rng.standard_t(df=2, size=5000)
    target = 0.95

    conformal_coverage = float((residuals >= compute_conformal_bound(residuals, alpha=0.05)).mean())
    sigma_coverage = float((residuals >= -2 * residuals.std()).mean())

    assert conformal_coverage >= target
    assert abs(conformal_coverage - target) < 0.01, "conformal should land on its target"
    assert sigma_coverage - target > 0.03, (
        f"2-sigma should be visibly miscalibrated on heavy tails "
        f"(got {sigma_coverage:.4f} against a {target} target)"
    )


def test_bound_is_minus_infinity_when_data_cannot_support_alpha():
    """Better an honest 'no finite bound' than a tight one invented from 5 points."""
    assert compute_conformal_bound(np.arange(5.0), alpha=0.01) == float("-inf")


def test_interval_brackets_the_residuals():
    rng = np.random.default_rng(2)
    residuals = rng.standard_normal(1000)
    lower, upper = compute_conformal_interval(residuals, alpha=0.1)
    assert lower < 0 < upper
    assert float(((residuals >= lower) & (residuals <= upper)).mean()) >= 0.9


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError, match="alpha"):
        compute_conformal_bound(np.zeros(10), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        compute_conformal_interval(np.zeros(10), alpha=1.0)


def test_nan_residuals_are_ignored():
    residuals = np.array([1.0, 2.0, np.nan, -1.0, 0.5] * 100)
    assert np.isfinite(compute_conformal_bound(residuals, alpha=0.1))


# ---------------------------------------------------------------------------
# Mondrian calibration
# ---------------------------------------------------------------------------


def _calibration_data(n: int = 4000, seed: int = 3):
    """Residuals whose spread grows with irradiance, as real PV residuals do."""
    rng = np.random.default_rng(seed)
    irradiance = rng.uniform(0, 1100, n)
    residuals = rng.normal(0, 1 + irradiance / 200)
    return residuals, irradiance


def test_fit_produces_bounds_per_populated_bucket():
    residuals, irradiance = _calibration_data()
    calibration = ConformalCalibration.fit(residuals, irradiance, alpha=0.05)
    assert GLOBAL_BUCKET in calibration.lower_bounds
    for bucket in BUCKET_ORDER:
        if calibration.sample_counts.get(bucket, 0) >= MIN_BUCKET_SAMPLES:
            assert bucket in calibration.lower_bounds


def test_bounds_widen_with_irradiance():
    """A single global bound would be too tight at noon and too loose at dawn."""
    residuals, irradiance = _calibration_data()
    calibration = ConformalCalibration.fit(residuals, irradiance, alpha=0.05)
    assert calibration.lower_bound_for(1000) < calibration.lower_bound_for(100)


def test_sparse_buckets_fall_back_to_the_global_bound():
    """A bound from a handful of points is noise wearing the costume of one."""
    rng = np.random.default_rng(4)
    irradiance = np.concatenate([rng.uniform(0, 100, 2000), rng.uniform(1000, 1100, 5)])
    residuals = rng.normal(0, 1, len(irradiance))
    calibration = ConformalCalibration.fit(residuals, irradiance, alpha=0.05)
    assert "peak" not in calibration.lower_bounds
    assert calibration.lower_bound_for(1050) == calibration.lower_bounds[GLOBAL_BUCKET]


def test_coverage_holds_out_of_sample():
    """The check that matters: the guarantee survives unseen data."""
    train_residuals, train_irradiance = _calibration_data(seed=5)
    test_residuals, test_irradiance = _calibration_data(seed=6)

    calibration = ConformalCalibration.fit(train_residuals, train_irradiance, alpha=0.05)
    coverage = calibration.coverage_on(test_residuals, test_irradiance)
    assert coverage["overall"] >= 0.92  # 0.95 target, small-sample slack


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError, match="same shape"):
        ConformalCalibration.fit(np.zeros(10), np.zeros(5))


def test_calibration_round_trips(tmp_path):
    residuals, irradiance = _calibration_data()
    calibration = ConformalCalibration.fit(residuals, irradiance, alpha=0.05)
    path = calibration.save(tmp_path / "conformal.json")

    loaded = ConformalCalibration.load(path)
    assert loaded.alpha == calibration.alpha
    assert loaded.lower_bounds == calibration.lower_bounds
    assert loaded.lower_bound_for(800) == calibration.lower_bound_for(800)


def test_loading_absent_calibration_returns_none(tmp_path):
    assert ConformalCalibration.load(tmp_path / "nothing.json") is None


def test_serialised_calibration_states_its_guarantee():
    residuals, irradiance = _calibration_data()
    payload = ConformalCalibration.fit(residuals, irradiance, alpha=0.05).to_dict()
    assert "95%" in payload["guarantee"]
    assert "exchangeability" in payload["guarantee"].lower()


# ---------------------------------------------------------------------------
# Physics baseline
# ---------------------------------------------------------------------------


def test_physics_predicts_zero_without_sun(demo_site):
    model = PhysicsBaseline(demo_site, calibrate=False)
    X = pd.DataFrame({"irradiance_wm2": [0.0], "temperature_c": [15.0], "wind_speed_ms": [2.0]})
    assert model.predict(X)[0] == pytest.approx(0.0)


def test_physics_output_rises_with_irradiance(demo_site):
    model = PhysicsBaseline(demo_site, calibrate=False)
    X = pd.DataFrame(
        {
            "irradiance_wm2": [200.0, 600.0, 1000.0],
            "temperature_c": [20.0, 20.0, 20.0],
            "wind_speed_ms": [2.0, 2.0, 2.0],
        }
    )
    predictions = model.predict(X)
    assert predictions[0] < predictions[1] < predictions[2]


def test_physics_derates_with_temperature(demo_site):
    """Warmer cells produce less; this is the sign of the effect, not its size."""
    model = PhysicsBaseline(demo_site, calibrate=False)
    X = pd.DataFrame(
        {
            "irradiance_wm2": [800.0, 800.0],
            "temperature_c": [10.0, 40.0],
            "wind_speed_ms": [2.0, 2.0],
        }
    )
    cool, hot = model.predict(X)
    assert hot < cool


def test_physics_clips_at_the_inverter_rating(demo_site):
    model = PhysicsBaseline(demo_site, calibrate=False)
    X = pd.DataFrame({"irradiance_wm2": [2000.0], "temperature_c": [20.0], "wind_speed_ms": [2.0]})
    assert model.predict(X)[0] <= demo_site.capacity_kw


def test_physics_never_predicts_negative(demo_site):
    model = PhysicsBaseline(demo_site, calibrate=False)
    X = pd.DataFrame(
        {"irradiance_wm2": [0.0, 5.0], "temperature_c": [60.0, 60.0], "wind_speed_ms": [0.0, 0.0]}
    )
    assert (model.predict(X) >= 0).all()


def test_physics_tolerates_missing_weather(demo_site):
    """Weather channels can drop out while power does not; do not propagate NaN."""
    model = PhysicsBaseline(demo_site, calibrate=False)
    X = pd.DataFrame(
        {"irradiance_wm2": [800.0], "temperature_c": [np.nan], "wind_speed_ms": [np.nan]}
    )
    assert np.isfinite(model.predict(X)[0])


def test_physics_requires_its_input_columns(demo_site):
    model = PhysicsBaseline(demo_site, calibrate=False)
    with pytest.raises(ValueError, match="needs columns"):
        model.predict(pd.DataFrame({"irradiance_wm2": [500.0]}))


def test_physics_fits_exactly_one_parameter(synthetic_frame, demo_site):
    X, y = get_X_y(synthetic_frame, mode="weather_only")
    model = PhysicsBaseline(demo_site).fit(X, y)
    assert model.explain()["free_parameters"] == 1
    assert 0.3 <= model.derate_ <= 1.8


def test_physics_explains_its_assumptions(demo_site):
    detail = PhysicsBaseline(demo_site, calibrate=False).explain()
    assert "gamma_pdc" in detail["assumptions"]
    assert "generic" in detail["assumption_note"].lower()


def test_physics_assumptions_are_overridable(demo_site):
    strong = PhysicsBaseline(
        demo_site, assumptions=PhysicsAssumptions(system_losses=0.0), calibrate=False
    )
    weak = PhysicsBaseline(
        demo_site, assumptions=PhysicsAssumptions(system_losses=0.5), calibrate=False
    )
    X = pd.DataFrame({"irradiance_wm2": [600.0], "temperature_c": [20.0], "wind_speed_ms": [2.0]})
    assert strong.predict(X)[0] > weak.predict(X)[0]


def test_hybrid_beats_bare_physics_on_its_training_distribution(synthetic_frame, demo_site):
    X, y = get_X_y(synthetic_frame, mode="weather_only")
    physics = PhysicsBaseline(demo_site).fit(X, y)
    hybrid = PhysicsResidualModel(demo_site).fit(X, y)

    physics_mae = float(np.mean(np.abs(y - physics.predict(X))))
    hybrid_mae = float(np.mean(np.abs(y - hybrid.predict(X))))
    assert hybrid_mae < physics_mae


def test_hybrid_never_predicts_negative(synthetic_frame, demo_site):
    X, y = get_X_y(synthetic_frame, mode="weather_only")
    hybrid = PhysicsResidualModel(demo_site).fit(X, y)
    assert (hybrid.predict(X) >= 0).all()


def test_physics_uses_real_registry_geometry():
    """The measured sites carry published tilt and azimuth, so physics is grounded."""
    site = get_site("nist_ground")
    detail = PhysicsBaseline(site, calibrate=False).explain()
    assert detail["tilt_deg"] == 20.0
    assert detail["azimuth_deg"] == 180.0
    assert "metadata" in detail["capacity_basis"].lower()
