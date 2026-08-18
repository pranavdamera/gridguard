"""
Build every production artifact, deterministically.

This is the one command that stands between a fresh clone and a deployable
backend. It downloads or generates the approved data, preprocesses it, trains
the models, calibrates uncertainty, evaluates everything, and writes a manifest
recording exactly what was built and how it scored.

    python scripts/build_artifacts.py                 # the shipped demo fleet
    python scripts/build_artifacts.py --mode real     # measured sites only
    python scripts/build_artifacts.py --mode synthetic
    python scripts/build_artifacts.py --sites nist_ground gmu_fairfax
    python scripts/build_artifacts.py --refresh       # ignore the data cache

Or via the Makefile:

    make build-artifacts

The deployed backend never runs this. It loads what this produced.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridguard.artifacts import MANIFEST_FILENAME, ArtifactManifest
from gridguard.config import settings
from gridguard.data.synthetic import DEMO_SITE_ID
from gridguard.pipeline import build_site
from gridguard.sites.registry import list_sites

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_artifacts")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GridGuard production artifacts")
    parser.add_argument(
        "--mode",
        choices=["all", "real", "synthetic"],
        default="all",
        help="Which fleet to build. Default: all.",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="Explicit site ids. Overrides --mode.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the telemetry cache and re-download or regenerate.",
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Skip the pvlib physics baseline and hybrid model.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=f"Artifact output directory. Default: {settings.model_dir}",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep building remaining sites when one fails.",
    )
    return parser.parse_args(argv)


def select_sites(args: argparse.Namespace) -> list[str]:
    if args.sites:
        return list(args.sites)
    mode = None if args.mode == "all" else args.mode
    return [s.site_id for s in list_sites(data_mode=mode)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    site_ids = select_sites(args)
    if not site_ids:
        logger.error("No sites selected.")
        return 1

    model_dir = Path(args.model_dir or settings.model_dir)
    manifest = ArtifactManifest()
    manifest.build_notes.append(
        f"Detection method: {settings.anomaly_method} (conformal alpha="
        f"{settings.conformal_alpha})."
    )

    logger.info("Building artifacts for %d site(s): %s", len(site_ids), ", ".join(site_ids))
    failures: list[str] = []

    for site_id in site_ids:
        try:
            result = build_site(
                site_id,
                model_dir=model_dir,
                include_physics=not args.no_physics,
                use_cache=not args.refresh,
                demo=(site_id == DEMO_SITE_ID),
            )
            manifest.sites.append(result.artifact)
            _log_site_summary(result)
        except Exception as exc:
            failures.append(site_id)
            if args.continue_on_error:
                logger.error("Failed to build %s: %s", site_id, exc, exc_info=True)
                continue
            logger.error("Failed to build %s: %s", site_id, exc, exc_info=True)
            return 1

    if failures:
        manifest.build_notes.append(f"Sites that failed to build: {', '.join(failures)}")

    manifest_path = model_dir / MANIFEST_FILENAME
    manifest.save(manifest_path)

    logger.info("=" * 70)
    logger.info(
        "Built %d/%d site(s). Manifest: %s", len(manifest.sites), len(site_ids), manifest_path
    )
    logger.info("Commit %s%s", manifest.git_commit, " (dirty)" if manifest.git_dirty else "")
    logger.info("=" * 70)
    return 1 if failures else 0


def _log_site_summary(result) -> None:
    artifact = result.artifact
    logger.info("-" * 70)
    logger.info(
        "%s (%s): %d rows, best forecast model = %s",
        artifact.site_id,
        artifact.data_mode,
        artifact.row_count,
        artifact.best_forecast_model,
    )
    for row in artifact.forecast_metrics:
        logger.info(
            "    %-16s MAE %8.3f kW   RMSE %8.3f kW   R2 %7.4f",
            row.get("model", "?"),
            row.get("mae_kw", float("nan")),
            row.get("rmse_kw", float("nan")),
            row.get("r2", float("nan")),
        )
    if artifact.physics_metrics:
        logger.info("    -- weather-only models (not comparable with the above) --")
        for name in ("physics", "physics_hybrid", "weather_only_xgboost"):
            entry = artifact.physics_metrics.get(name)
            if isinstance(entry, dict) and "mae_kw" in entry:
                logger.info(
                    "    %-16s MAE %8.3f kW   RMSE %8.3f kW   R2 %7.4f",
                    name,
                    entry["mae_kw"],
                    entry["rmse_kw"],
                    entry["r2"],
                )
    if artifact.conformal_coverage:
        logger.info(
            "    conformal out-of-sample coverage: %s",
            {k: f"{v:.3f}" for k, v in artifact.conformal_coverage.items()},
        )
    detection = artifact.detection_metrics
    if detection:
        logger.info(
            "    detection: P=%.3f R=%.3f F1=%.3f | events %s/%s | false alarms %.2f/day",
            detection.get("precision", float("nan")),
            detection.get("recall", float("nan")),
            detection.get("f1", float("nan")),
            detection.get("events_detected", "?"),
            detection.get("events_total", "?"),
            detection.get("false_alarms_per_day", float("nan")),
        )
    logger.info("    events grouped: %d", len(result.events))


if __name__ == "__main__":
    sys.exit(main())
