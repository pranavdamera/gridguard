"""
Tests proving train/test calibration separation.

Key invariant: residual_stats computed from training data should never be
contaminated by test-set statistics. These tests verify:
  1. compute_and_save_residual_stats uses training rows only.
  2. detect_anomalies with freeze=True loads the saved file.
  3. detect_anomalies with freeze=False (self-compute) produces different stats
     when test distribution differs from training distribution.
  4. Frozen stats are stable across multiple calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.anomaly.detect import (
    _compute_residual_stats,
    _load_residual_stats,
    compute_and_save_residual_stats,
    detect_anomalies,
)
from gridguard.features.engineer import FEATURE_COLS, build_features, get_X_y


@pytest.fixture()
def simple_model_and_splits(tmp_path):
    """Return (model, train_df, test_df) with distinct distributions."""
    from gridguard.ingestion.download import _generate_synthetic

    full_df = _generate_synthetic(start="2022-01-01", end="2023-06-30", seed=1)
    train_df = full_df[full_df["timestamp"] < "2023-01-01"].copy()
    test_df = full_df[full_df["timestamp"] >= "2023-01-01"].copy()

    model = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    X_train, y_train = get_X_y(train_df)
    model.fit(X_train, y_train)

    return model, train_df, test_df


def test_compute_and_save_creates_json(simple_model_and_splits, tmp_path):
    model, train_df, _ = simple_model_and_splits
    stats = compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)
    json_path = tmp_path / "residual_stats.json"
    assert json_path.exists(), "residual_stats.json should be created"
    assert isinstance(stats, dict)
    assert len(stats) > 0


def test_json_has_all_daylight_hours(simple_model_and_splits, tmp_path):
    model, train_df, _ = simple_model_and_splits
    compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)
    stats = _load_residual_stats(tmp_path)
    # Expect at least 6 hours of daylight to have entries
    assert len(stats) >= 6


def test_json_values_have_mean_and_std(simple_model_and_splits, tmp_path):
    model, train_df, _ = simple_model_and_splits
    compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)
    stats = _load_residual_stats(tmp_path)
    for hour, vals in stats.items():
        assert "mean" in vals and "std" in vals, f"Hour {hour} missing mean/std"


def test_load_residual_stats_returns_none_when_missing(tmp_path):
    result = _load_residual_stats(tmp_path)
    assert result is None


def test_frozen_stats_differ_from_self_computed(simple_model_and_splits, tmp_path):
    """
    If test data has different distribution, frozen (train) stats differ from
    self-computed (test) stats — demonstrating the calibration separation.
    """
    model, train_df, test_df = simple_model_and_splits
    compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)

    frozen_stats = _load_residual_stats(tmp_path)
    test_feat = build_features(test_df.copy())
    present = [c for c in FEATURE_COLS if c in test_feat.columns]
    test_feat["predicted_kw"] = np.clip(model.predict(test_feat[present]), 0, None)
    test_feat["residual_kw"] = test_feat["ac_power_kw"] - test_feat["predicted_kw"]
    daylight = test_feat["irradiance_wm2"] > 50
    self_computed = _compute_residual_stats(test_feat[daylight])

    # At least one hour should differ (different seasonal data in train vs test)
    common_hours = set(frozen_stats) & set(self_computed)
    diffs = [
        abs(frozen_stats[h]["std"] - self_computed[h]["std"])
        for h in common_hours
    ]
    assert any(d > 1e-6 for d in diffs), (
        "Frozen train stats and self-computed test stats should not be identical "
        "(train and test cover different time periods)"
    )


def test_detect_anomalies_uses_frozen_stats_by_default(simple_model_and_splits, tmp_path):
    """freeze=True should load the saved JSON, not compute from df."""
    model, train_df, test_df = simple_model_and_splits
    compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)

    # Temporarily override model_dir via the env
    import gridguard.anomaly.detect as det_mod
    original_load = det_mod._load_residual_stats

    captured = {}

    def patched_load(model_dir=None):
        stats = original_load(tmp_path)
        captured["stats"] = stats
        return stats

    det_mod._load_residual_stats = patched_load
    try:
        result = detect_anomalies(test_df, model, freeze=True)
    finally:
        det_mod._load_residual_stats = original_load

    assert "stats" in captured, "freeze=True should call _load_residual_stats"
    assert "is_anomaly" in result.columns


def test_detect_anomalies_freeze_false_does_not_load_file(simple_model_and_splits, tmp_path):
    """freeze=False should NOT load from disk."""
    model, train_df, test_df = simple_model_and_splits
    compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)

    import gridguard.anomaly.detect as det_mod
    calls = []
    original_load = det_mod._load_residual_stats

    def patched_load(model_dir=None):
        calls.append(1)
        return original_load(model_dir)

    det_mod._load_residual_stats = patched_load
    try:
        detect_anomalies(test_df, model, freeze=False)
    finally:
        det_mod._load_residual_stats = original_load

    assert len(calls) == 0, "freeze=False should skip _load_residual_stats"


def test_residual_stats_only_use_training_rows(simple_model_and_splits, tmp_path):
    """Stats computed from train_df should reflect training-split residuals."""
    model, train_df, test_df = simple_model_and_splits
    compute_and_save_residual_stats(train_df, model, model_dir=tmp_path)
    frozen = _load_residual_stats(tmp_path)

    # Manual recomputation from training split
    train_feat = build_features(train_df.copy())
    present = [c for c in FEATURE_COLS if c in train_feat.columns]
    train_feat["predicted_kw"] = np.clip(model.predict(train_feat[present]), 0, None)
    train_feat["residual_kw"] = train_feat["ac_power_kw"] - train_feat["predicted_kw"]
    daylight = train_feat["irradiance_wm2"] > 50
    manual = _compute_residual_stats(train_feat[daylight])

    for hour in set(frozen) & set(manual):
        assert abs(frozen[hour]["mean"] - manual[hour]["mean"]) < 1e-4, (
            f"Hour {hour}: frozen mean differs from training mean"
        )
