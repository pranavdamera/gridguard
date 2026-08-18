"""
Full training pipeline: data → features → train → evaluate → save.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --site-id gmu_fairfax
    python scripts/run_pipeline.py --site-id gmu_fairfax --demo
    python scripts/run_pipeline.py --models xgboost random_forest
    python scripts/run_pipeline.py --skip-data   (if data/processed/raw*.parquet already exists)
"""

import argparse
import logging
from pathlib import Path

from gridguard.config import settings
from gridguard.ingestion.download import load_raw_data
from gridguard.models.baseline import ALL_MODELS
from gridguard.models.evaluate import compare_models
from gridguard.models.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run GridGuard DMV training pipeline")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip data download if already cached"
    )
    parser.add_argument("--source", default=settings.data_source, choices=["synthetic", "nrel"])
    parser.add_argument(
        "--site-id",
        default=None,
        metavar="SITE_ID",
        help="Site ID from config/sites.csv (e.g. gmu_fairfax). Uses site latitude and capacity.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate the deterministic GMU underperformance demo scenario (requires --site-id).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    logger.info("=== Step 1: Data ingestion ===")

    # Log selected site if provided
    if args.site_id:
        try:
            from gridguard.sites.registry import get_site

            site = get_site(args.site_id)
            logger.info(
                "Site: %s (lat=%.4f, cap=%.0f kW)", site.name, site.latitude, site.capacity_kw
            )
        except Exception:
            logger.info("Site ID: %s", args.site_id)

    # Determine the cache file name for --skip-data
    cache_name = (
        f"raw_{args.site_id}_demo.parquet"
        if (args.site_id and args.demo)
        else f"raw_{args.site_id}.parquet" if args.site_id else "raw.parquet"
    )
    cache = Path(settings.data_processed_dir) / cache_name

    if args.skip_data:
        if not cache.exists():
            logger.warning("--skip-data specified but %s not found; downloading anyway.", cache)
            args.skip_data = False

    if not args.skip_data and cache.exists():
        cache.unlink()

    df = load_raw_data(source=args.source, site_id=args.site_id, demo=args.demo)
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

    # Use the weather-only anomaly_detector, not the lag-aware forecast model
    anomaly_model = results["anomaly_model"]
    adf = detect_anomalies(results["test_df"], anomaly_model)
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
