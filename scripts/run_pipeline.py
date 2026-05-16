"""
Full training pipeline: data → features → train → evaluate → save.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --models xgboost random_forest
    python scripts/run_pipeline.py --skip-data   (if data/processed/raw.parquet already exists)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

from gridguard.config import settings
from gridguard.features.engineer import get_X_y, train_test_split_by_date
from gridguard.ingestion.download import load_raw_data
from gridguard.models.baseline import ALL_MODELS
from gridguard.models.evaluate import compare_models
from gridguard.models.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Run GridGuard training pipeline")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--skip-data", action="store_true", help="Skip data download if already cached")
    parser.add_argument("--source", default=settings.data_source, choices=["synthetic", "nrel"])
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    logger.info("=== Step 1: Data ingestion ===")
    if args.skip_data:
        cache = Path(settings.data_processed_dir) / "raw.parquet"
        if not cache.exists():
            logger.warning("--skip-data specified but no cache found; downloading anyway.")
            args.skip_data = False

    if not args.skip_data:
        import os
        # Remove cache to force reload
        cache = Path(settings.data_processed_dir) / "raw.parquet"
        if cache.exists():
            cache.unlink()

    df = load_raw_data(source=args.source)
    logger.info("Loaded %d rows (%s → %s)", len(df), df["timestamp"].min(), df["timestamp"].max())

    # -----------------------------------------------------------------------
    # Step 2: Train
    # -----------------------------------------------------------------------
    logger.info("=== Step 2: Training models: %s ===", args.models)
    results = run_training(df, model_names=args.models)

    # -----------------------------------------------------------------------
    # Step 3: Evaluate
    # -----------------------------------------------------------------------
    logger.info("=== Step 3: Evaluation ===")
    metrics = compare_models(results["models"], results["X_test"], results["y_test"])
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)
    print(metrics.to_string(index=False))
    print("=" * 60)

    best = metrics.iloc[0]
    logger.info(
        "Best model: %s  |  RMSE=%.4f kW  |  MAE=%.4f kW  |  R²=%.4f",
        best["model"],
        best["rmse_kw"],
        best["mae_kw"],
        best["r2"],
    )

    # -----------------------------------------------------------------------
    # Step 4: Anomaly detection summary
    # -----------------------------------------------------------------------
    logger.info("=== Step 4: Anomaly detection ===")
    from gridguard.anomaly.detect import compute_daily_loss, detect_anomalies

    best_model = results["models"][best["model"]]
    adf = detect_anomalies(results["test_df"], best_model)
    daily = compute_daily_loss(adf)
    total_loss = daily["lost_energy_kwh"].sum()
    fault_days = (daily["anomaly_count"] > 0).sum()

    logger.info(
        "Anomaly summary: %d fault days detected, %.1f kWh estimated lost energy",
        fault_days,
        total_loss,
    )

    logger.info("Pipeline complete. Run 'make api' and 'make dashboard' to start the demo.")


if __name__ == "__main__":
    main()
