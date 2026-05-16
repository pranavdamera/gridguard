"""
Training pipeline: load data → features → train all models → save artifacts.

Artifacts produced:
  artifacts/models/{model_name}.pkl       — fitted model
  artifacts/models/residual_stats.json    — per-hour residual calibration (training split only)
  data/processed/test_df.parquet          — test split for evaluation and dashboard

Usage:
    python scripts/run_pipeline.py

    or programmatically:
        from gridguard.models.train import run_training
        results = run_training(df)
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


def run_training(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    model_dir: Path | None = None,
) -> dict:
    """Train all models on df, save artifacts, and calibrate residual stats.

    Returns dict with keys: models, X_train, X_test, y_test, test_df, train_df.
    """
    model_names = model_names or ALL_MODELS
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split_by_date(df, settings.train_test_split_date)
    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    trained: dict[str, object] = {}
    best_model = None  # used for residual calibration; prefer xgboost

    for name in model_names:
        logger.info("Training %s …", name)
        model = get_model(name)

        if name == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        artifact_path = model_dir / f"{name}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved %s → %s", name, artifact_path)
        trained[name] = model

        if best_model is None or name == "xgboost":
            best_model = model

    # Compute and save residual calibration from the TRAINING split only.
    # This prevents test-set statistics from leaking into anomaly thresholds.
    if best_model is not None and not isinstance(best_model, type):
        from gridguard.anomaly.detect import compute_and_save_residual_stats

        compute_and_save_residual_stats(train_df, best_model, model_dir=model_dir)

    # Save test split for evaluation and dashboard
    test_df_path = Path(settings.data_processed_dir) / "test_df.parquet"
    test_df_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(test_df_path, index=False)

    return {
        "models": trained,
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
        raise FileNotFoundError(f"No artifact at {path} — run 'make train' first.")
    with open(path, "rb") as f:
        return pickle.load(f)
