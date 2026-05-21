"""
Model zoo: persistence baseline → linear regression → RandomForest → XGBoost.

All models share the same sklearn-compatible .fit(X, y) / .predict(X) API,
making it easy to compare them in evaluate.py.

Progression:
  Persistence  — naive "tomorrow = today" benchmark; any real model must beat this.
  Linear       — checks if relationships are roughly linear; fast and interpretable.
  RandomForest — handles non-linearity and interactions; no feature scaling needed.
  XGBoost      — typically best performance; supports SHAP natively.

TODO: Add a LightGBM variant for comparison.
TODO: Add a simple LSTM baseline in pytorch/ for sequence modelling practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------


class PersistenceModel(BaseEstimator, RegressorMixin):
    """Predict next value = current value (lag-1 persistence).

    This is the standard naive benchmark for energy time-series. A model
    that doesn't clearly beat persistence isn't worth deploying.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceModel:
        # Nothing to learn — persistence uses the lag-1 feature directly
        self._lag_col = "ac_power_lag1"
        if self._lag_col not in X.columns:
            raise ValueError("PersistenceModel needs 'ac_power_lag1' in X")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self._lag_col].values


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def get_model(name: str) -> BaseEstimator:
    """Return an unfitted sklearn-compatible model by name."""
    models = {
        "persistence": PersistenceModel(),
        "linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=30,  # pass eval_set to .fit()
            random_state=42,
            verbosity=0,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models)}")
    return models[name]


ALL_MODELS = ["persistence", "linear", "random_forest", "xgboost"]
