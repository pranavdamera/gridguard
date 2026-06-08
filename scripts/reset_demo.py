"""
Deterministic demo reset for GridGuard.

Regenerates the exact same synthetic telemetry, model artifacts, anomaly
intervals, and grouped events every time using a fixed random seed (42).

Scripted event: GMU Fairfax Campus PV — underperformance on 2023-06-15
  Window:  09:00 – 12:15 (13 × 15-min intervals)
  Effect:  70% AC power reduction during peak irradiance hours
  Site:    gmu_fairfax  (250 kW nameplate, 38.8308° N, −77.3075° W)

Usage:
    python scripts/reset_demo.py
    # or via Makefile:
    make demo-reset
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEMO_SITE_ID = "gmu_fairfax"
DEMO_DATE = "2023-06-15"
SEED = 42


def main() -> None:
    logger.info("=" * 60)
    logger.info("GridGuard demo reset — deterministic synthetic bundle")
    logger.info("Site: %s  |  Seed: %d  |  Demo date: %s", DEMO_SITE_ID, SEED, DEMO_DATE)
    logger.info("=" * 60)

    from gridguard.config import settings
    from gridguard.ingestion.download import _generate_synthetic, load_raw_data
    from gridguard.models.baseline import ALL_MODELS
    from gridguard.models.evaluate import compare_models
    from gridguard.models.train import run_training
    from gridguard.anomaly.detect import compute_daily_loss, detect_anomalies
    from gridguard.anomaly.events import group_anomaly_events
    from gridguard.sites.registry import get_site

    model_dir = Path(settings.model_dir)
    processed_dir = Path(settings.data_processed_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate deterministic demo telemetry
    # ------------------------------------------------------------------
    logger.info("Step 1/5 — Generating synthetic telemetry …")
    site = get_site(DEMO_SITE_ID)
    logger.info(
        "  Site: %s  lat=%.4f  cap=%.0f kW",
        site.name, site.latitude, site.capacity_kw,
    )

    # Remove any existing demo cache so we regenerate fresh
    demo_cache = processed_dir / f"raw_{DEMO_SITE_ID}_demo.parquet"
    base_cache = processed_dir / f"raw_{DEMO_SITE_ID}.parquet"
    for p in [demo_cache, base_cache]:
        if p.exists():
            p.unlink()

    df = load_raw_data(source="synthetic", site_id=DEMO_SITE_ID, demo=True)
    logger.info("  Generated %d rows  (%s → %s)", len(df), df["timestamp"].min(), df["timestamp"].max())

    # Verify demo window is present
    demo_window = df[df["timestamp"].dt.date.astype(str) == DEMO_DATE]
    injected = demo_window[demo_window.get("is_injected_fault", False) == True] if "is_injected_fault" in df.columns else demo_window
    logger.info(
        "  Demo window (%s): %d rows, %.1f kW avg ac_power",
        DEMO_DATE, len(demo_window), demo_window["ac_power_kw"].mean(),
    )

    # ------------------------------------------------------------------
    # Step 2: Train models
    # ------------------------------------------------------------------
    logger.info("Step 2/5 — Training models (all: %s) …", ALL_MODELS)
    results = run_training(df, model_names=ALL_MODELS)
    logger.info("  Training complete. Test rows: %d", len(results["test_df"]))

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    logger.info("Step 3/5 — Model evaluation …")
    metrics = compare_models(results["models"], results["X_test"], results["y_test"])
    best = metrics.iloc[0]
    logger.info(
        "  Best: %s  RMSE=%.3f kW  MAE=%.3f kW  R²=%.4f",
        best["model"], best["rmse_kw"], best["mae_kw"], best["r2"],
    )
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)
    print(metrics.to_string(index=False))
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 4: Anomaly detection
    # ------------------------------------------------------------------
    logger.info("Step 4/5 — Running anomaly detection …")
    # Use the dedicated weather-only anomaly detector, NOT the lag-aware forecast model
    anomaly_model = results["anomaly_model"]
    test_df = results["test_df"]
    adf = detect_anomalies(test_df, anomaly_model)
    daily = compute_daily_loss(adf)
    fault_days = (daily["anomaly_count"] > 0).sum()
    total_loss = daily["lost_energy_kwh"].sum()
    logger.info(
        "  %d fault days detected, %.1f kWh estimated lost energy",
        fault_days, total_loss,
    )

    # Check the demo event is visible
    demo_day = daily[daily["date"].astype(str) == DEMO_DATE]
    if not demo_day.empty:
        d = demo_day.iloc[0]
        logger.info(
            "  Demo event (%s): %d anomaly intervals, %.2f kWh lost",
            DEMO_DATE, int(d["anomaly_count"]), float(d["lost_energy_kwh"]),
        )
    else:
        logger.warning("  Demo date %s not found in test set — check train/test split date.", DEMO_DATE)

    # ------------------------------------------------------------------
    # Step 5: Group events and save
    # ------------------------------------------------------------------
    logger.info("Step 5/5 — Grouping anomaly events …")
    events = group_anomaly_events(adf)
    logger.info("  %d events grouped (%d high severity)", len(events), (events["severity"] == "high").sum())

    # Persist test_df for the API to load
    test_out = processed_dir / "test_df.parquet"
    test_df.to_parquet(test_out, index=False)
    logger.info("  Saved test_df → %s", test_out)

    logger.info("=" * 60)
    logger.info("Demo reset complete.")
    logger.info("  Start API:       make api")
    logger.info("  Start dashboard: make dashboard")
    logger.info("  Start web:       make web")
    logger.info("  API docs:        http://localhost:8000/docs")
    logger.info("  Demo endpoint:   http://localhost:8000/demo/scenario")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
