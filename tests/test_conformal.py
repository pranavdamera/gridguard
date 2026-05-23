"""Tests for conformal lower prediction bound utility."""

import numpy as np
import pytest

from gridguard.anomaly.conformal import compute_conformal_bound, flag_conformal_anomalies


@pytest.fixture()
def normal_residuals():
    """Standard-normal calibration residuals (n=500)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(500)


def test_compute_conformal_bound_returns_scalar(normal_residuals):
    bound = compute_conformal_bound(normal_residuals, alpha=0.1)
    assert isinstance(bound, float)


def test_coverage_at_least_one_minus_alpha(normal_residuals):
    """At least (1-alpha) of calibration residuals must be >= bound."""
    alpha = 0.1
    bound = compute_conformal_bound(normal_residuals, alpha=alpha)
    coverage = (normal_residuals >= bound).mean()
    # Allow a tiny slack for the finite-sample correction
    assert coverage >= (1 - alpha) - 0.02, (
        f"Coverage {coverage:.3f} is below (1 - {alpha}) = {1 - alpha}"
    )


def test_tighter_bound_at_lower_alpha(normal_residuals):
    """Lower alpha (higher coverage requirement) → smaller (more negative) bound."""
    bound_10 = compute_conformal_bound(normal_residuals, alpha=0.10)
    bound_20 = compute_conformal_bound(normal_residuals, alpha=0.20)
    assert bound_10 <= bound_20, (
        "alpha=0.10 requires higher coverage, so the bound should be <= alpha=0.20"
    )


def test_bound_is_negative_for_normal_residuals(normal_residuals):
    """For symmetric residuals around 0, the lower bound should be negative."""
    bound = compute_conformal_bound(normal_residuals, alpha=0.1)
    assert bound < 0


def test_empty_residuals_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compute_conformal_bound(np.array([]), alpha=0.1)


def test_invalid_alpha_raises(normal_residuals):
    with pytest.raises(ValueError, match="alpha"):
        compute_conformal_bound(normal_residuals, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        compute_conformal_bound(normal_residuals, alpha=1.0)


def test_flag_conformal_anomalies_shape(normal_residuals):
    """Output shape must match evaluation residuals, not calibration residuals."""
    rng = np.random.default_rng(7)
    eval_residuals = rng.standard_normal(100)
    flags = flag_conformal_anomalies(eval_residuals, normal_residuals, alpha=0.1)
    assert flags.shape == (100,)
    assert flags.dtype == bool


def test_clearly_anomalous_residuals_flagged(normal_residuals):
    """Residuals far below the calibration lower tail must be flagged."""
    eval_residuals = np.array([-10.0, -8.0, 5.0, 2.0])
    flags = flag_conformal_anomalies(eval_residuals, normal_residuals, alpha=0.1)
    assert flags[0], "-10.0 should be flagged as anomalous"
    assert flags[1], "-8.0 should be flagged as anomalous"
    assert not flags[2], "5.0 should not be flagged"
    assert not flags[3], "2.0 should not be flagged"


def test_flag_false_positive_rate(normal_residuals):
    """On calibration data itself, false positive rate should be ~alpha."""
    alpha = 0.1
    flags = flag_conformal_anomalies(normal_residuals, normal_residuals, alpha=alpha)
    fpr = flags.mean()
    # Should be close to alpha (the split-conformal guarantee is approximate on self)
    assert fpr <= alpha + 0.05, f"FPR {fpr:.3f} exceeds alpha {alpha} by too much"


def test_single_calibration_point_returns_scalar():
    """With one calibration point the bound is the point itself (degenerate but valid)."""
    bound = compute_conformal_bound(np.array([1.0]), alpha=0.5)
    assert isinstance(bound, float)
    assert bound <= 1.0
