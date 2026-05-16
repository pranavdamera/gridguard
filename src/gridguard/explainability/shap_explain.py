"""
SHAP-based explainability for the XGBoost forecasting model.

SHAP (SHapley Additive exPlanations) answers: "Why did the model predict X
for this interval?" — critical for operator trust in an alert system.

Key outputs:
  1. Global feature importance (mean |SHAP| across all samples)
  2. Per-sample SHAP values for anomalous intervals
  3. Waterfall-style breakdown for a single interval

Why SHAP over simple feature importance?
  Tree feature importance counts splits — it ignores feature interaction magnitude.
  SHAP values are game-theoretically grounded: each feature's contribution is
  consistent and sums exactly to (prediction − baseline).

TODO: Add SHAP dependence plots (irradiance vs SHAP) to show non-linearity.
TODO: Extend to RandomForest via TreeExplainer (works the same way).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability features disabled. Run: pip install shap")


def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for XGBoost model.

    Returns:
        shap_values: array of shape (n_samples, n_features)
        feature_names: list of feature column names
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap package is required. Install with: pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, list(X.columns)


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of features sorted by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_anomaly(
    model,
    X_anomaly: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top_n SHAP contributors for each anomalous interval.

    Negative SHAP values on a positive target mean the feature is pulling
    the prediction down — useful for explaining "why is output low today?".

    Returns a DataFrame with columns: timestamp_idx, feature, shap_value
    """
    if not SHAP_AVAILABLE:
        return pd.DataFrame({"feature": X_anomaly.columns, "shap_value": [0.0] * len(X_anomaly.columns)})

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_anomaly)

    rows = []
    for i in range(len(X_anomaly)):
        sample_sv = sv[i]
        top_idx = np.argsort(np.abs(sample_sv))[::-1][:top_n]
        for j in top_idx:
            rows.append(
                {
                    "sample_idx": X_anomaly.index[i],
                    "feature": X_anomaly.columns[j],
                    "shap_value": round(sample_sv[j], 4),
                    "feature_value": round(float(X_anomaly.iloc[i, j]), 4),
                }
            )
    return pd.DataFrame(rows)
