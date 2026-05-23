"""
Conformal lower prediction bound for solar generation anomaly detection.

Motivation
----------
The default anomaly detector uses a per-hour sigma threshold: flag an interval
if residual < -k * sigma(training residuals). This is fast and interpretable,
but it provides no formal coverage guarantee — the threshold is heuristic.

Conformal prediction offers a distribution-free alternative: given a calibration
set of residuals from a held-out split, we can compute a lower bound q such that
at least (1 - alpha) fraction of future residuals will exceed q, with no
distributional assumptions beyond exchangeability.

Usage
-----
1. Compute calibration residuals from the TRAINING split only (same principle
   as frozen residual stats in detect.py — never use test data here).
2. Call compute_conformal_bound(cal_residuals, alpha=0.1) to get the lower bound.
3. Flag any evaluation interval where actual - predicted < bound.

This module is a clean foundation. The current production detector (detect.py)
still uses sigma thresholding as the default, which is documented in the
detect.py module. The conformal approach is the principled next step.

References
----------
Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification." arXiv:2107.07511.

Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift."
NeurIPS 2019.
"""

from __future__ import annotations

import numpy as np


def compute_conformal_bound(
    calibration_residuals: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute the lower conformal bound at miscoverage level alpha.

    Given calibration residuals (actual − predicted) from the training split,
    returns a threshold q such that at least ceil((n+1)*(1-alpha))/n fraction
    of calibration residuals are >= q. This is the finite-sample conformal
    quantile with the standard +1/n correction.

    Args:
        calibration_residuals: 1-D array of residuals from the calibration
            (training) split. Must use the same model and feature set that will
            be used at evaluation time. Never pass test-set residuals.
        alpha: Miscoverage rate in (0, 1). alpha=0.1 targets 90% coverage.
            Lower alpha → tighter bound → fewer false positives.

    Returns:
        Scalar lower bound q. Flag an interval as anomalous if its residual < q.
    """
    residuals = np.asarray(calibration_residuals, dtype=float)
    if len(residuals) == 0:
        raise ValueError("calibration_residuals must be non-empty")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(residuals)
    # Finite-sample correction: quantile level accounts for the +1 unseen point
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    # Lower bound = the (1 - level)-th quantile of the residual distribution
    # i.e., the left tail where residuals are most negative
    q = float(np.quantile(residuals, 1.0 - level))
    return q


def flag_conformal_anomalies(
    residuals: np.ndarray,
    calibration_residuals: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Return a boolean mask of conformal anomalies.

    An interval is anomalous if its residual falls below the conformal lower
    bound derived from calibration_residuals at miscoverage level alpha.

    Args:
        residuals:             1-D array of evaluation residuals (actual − predicted).
        calibration_residuals: 1-D array of training-split residuals used to set
                               the lower bound. Must NOT overlap with the evaluation set.
        alpha:                 Miscoverage rate. alpha=0.1 → ~90% of calibration
                               intervals are correctly labeled non-anomalous.

    Returns:
        Boolean array of the same length as residuals; True = anomalous.
    """
    bound = compute_conformal_bound(calibration_residuals, alpha=alpha)
    return np.asarray(residuals) < bound
