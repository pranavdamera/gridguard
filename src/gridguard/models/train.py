"""
Training pipeline: load data → features → train models → save artifacts.

Artifacts produced:
  artifacts/models/{name}.pkl          — lag-aware forecasting models
  artifacts/models/anomaly_detector.pkl — weather-only model for fault detection
  artifacts/models/residual_stats.json  — per-hour calibration from healthy training data
  data/processed/test_df.parquet       — held-out test split for evaluation

Two-model design:
  Forecasting models ({name}.pkl) — trained with lag-aware features for short-horizon
    operational forecasting. Compared on the held-out test set for the /metrics endpoint.

  anomaly_detector.pkl — trained with WEATHER_ONLY features on healthy (non-fault)
    training intervals. The key credibility property: it cannot learn that low output
    is "normal" by looking at lag features from a degraded system. See detect.py.

Train/validation/test split:
  test_df    — rows from TRAIN_TEST_SPLIT_DATE onward (never touched during training)
  val_df     — last VAL_FRAC of the training rows (for XGBoost early stopping)
  train_df   — remaining training rows (before val_df)

XGBoost early stopping uses val_df, NOT test_df. Using test_df for early stopping
makes the test set part of model selection, inflating apparent accuracy.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from gridguard.config import settings
from gridguard.features.engineer import get_X_y, train_test_split_by_date
from gridguard.models.baseline import ALL_MODELS, get_model

logger = logging.getLogger(__name__)

VAL_FRAC = 0.15  # fraction of training data reserved for validation (XGBoost early stopping)


def run_training(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    model_dir: Path | None = None,
) -> dict:
    """Train forecasting models + anomaly detector, save all artifacts.

    Returns dict with keys:
        models         — dict[name → fitted lag-aware model]
        anomaly_model  — fitted weather-only anomaly detector
        X_train, X_test, y_test, test_df, train_df
    """
    model_names = model_names or ALL_MODELS
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Temporal splits — NEVER shuffle time-series data
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split_by_date(df, settings.train_test_split_date)
    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    # Validation split within training data (chronological, last VAL_FRAC rows)
    val_cutoff = int(len(train_df) * (1 - VAL_FRAC))
    train_train_df = train_df.iloc[:val_cutoff].copy()
    train_val_df = train_df.iloc[val_cutoff:].copy()
    logger.info(
        "Train-train: %d rows | Train-val: %d rows | Test: %d rows",
        len(train_train_df),
        len(train_val_df),
        len(test_df),
    )

    # ------------------------------------------------------------------
    # Part A: Lag-aware forecasting models
    # ------------------------------------------------------------------
    X_train, y_train = get_X_y(train_train_df, mode="lag_aware")
    X_val, y_val = get_X_y(train_val_df, mode="lag_aware")
    X_test, y_test = get_X_y(test_df, mode="lag_aware")

    trained: dict[str, object] = {}

    for name in model_names:
        logger.info("Training forecasting model: %s …", name)
        model = get_model(name)

        if name == "xgboost":
            # Use validation set for early stopping — never the test set
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif name == "persistence":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        artifact_path = model_dir / f"{name}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved %s → %s", name, artifact_path)
        trained[name] = model

    # ------------------------------------------------------------------
    # Part B: Weather-only anomaly detector, trained on HEALTHY data only
    # ------------------------------------------------------------------
    logger.info("Training anomaly detector (weather-only, healthy data only) …")

    # Exclude injected fault intervals from the baseline so the model learns
    # what a *healthy* system should produce, not a degraded one.
    if "is_injected_fault" in train_train_df.columns:
        healthy_train_df = train_train_df[~train_train_df["is_injected_fault"]].copy()
        n_excluded = len(train_train_df) - len(healthy_train_df)
        logger.info(
            "  Excluded %d injected-fault intervals from anomaly baseline training.",
            n_excluded,
        )
    else:
        healthy_train_df = train_train_df

    X_train_w, y_train_w = get_X_y(healthy_train_df, mode="weather_only")
    X_val_w, y_val_w = get_X_y(train_val_df, mode="weather_only")

    # Use same XGBoost hyperparams but validate on the weather-only val set
    from xgboost import XGBRegressor

    anomaly_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    anomaly_model.fit(
        X_train_w,
        y_train_w,
        eval_set=[(X_val_w, y_val_w)],
        verbose=False,
    )
    anomaly_path = model_dir / "anomaly_detector.pkl"
    with open(anomaly_path, "wb") as f:
        pickle.dump(anomaly_model, f)
    logger.info("Saved anomaly_detector → %s", anomaly_path)

    # ------------------------------------------------------------------
    # Part C: Residual calibration from anomaly model on healthy training data
    # ------------------------------------------------------------------
    from gridguard.anomaly.detect import compute_and_save_residual_stats

    compute_and_save_residual_stats(healthy_train_df, anomaly_model, model_dir=model_dir)

    # ------------------------------------------------------------------
    # Persist test split for API and dashboard
    # ------------------------------------------------------------------
    test_df_path = Path(settings.data_processed_dir) / "test_df.parquet"
    test_df_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(test_df_path, index=False)

    return {
        "models": trained,
        "anomaly_model": anomaly_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": test_df,
        "train_df": train_df,
    }


def load_model(name: str, model_dir: Path | None = None) -> object:
    """Load a previously saved model artifact."""
    model_dir = Path(model_dir or settings.model_dir)
    path = model_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No artifact at {path} — run 'make demo-reset' first.")
    with open(path, "rb") as f:
        return pickle.load(f)
