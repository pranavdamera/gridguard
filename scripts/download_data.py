"""
Fetch or regenerate site telemetry.

The curated measured datasets are committed to the repository, so a fresh clone
runs the real-data pipeline with no download. This script is what regenerates
them — run it to change the window, add a measured site, or refresh after an
upstream correction.

    python scripts/download_data.py --list-sites
    python scripts/download_data.py --refresh-curated
    python scripts/download_data.py --site-id nist_ground --start 2016-01-01 --end 2016-12-31
    python scripts/download_data.py --site-id gmu_fairfax          # regenerate a simulated site

No API key is required. Measured data comes from the NREL PVDAQ collection in
the Open Energy Data Initiative data lake, which is anonymously readable over
HTTPS. (The legacy PVDAQ v3 REST API this project once used has been retired.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridguard.config import settings
from gridguard.data.provenance import save_provenance
from gridguard.data.sources import load_site_data
from gridguard.sites.registry import list_sites, load_sites

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("download_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or regenerate GridGuard telemetry")
    parser.add_argument(
        "--list-sites", action="store_true", help="Print the site registry and exit"
    )
    parser.add_argument("--site-id", default=None, help="Site to fetch or generate")
    parser.add_argument(
        "--refresh-curated",
        action="store_true",
        help="Re-download every measured site and refresh data/curated/",
    )
    parser.add_argument("--start", default=None, help="ISO date (measured sites)")
    parser.add_argument("--end", default=None, help="ISO date (measured sites)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Include the scripted demonstration event (simulated sites only)",
    )
    return parser.parse_args(argv)


def print_registry() -> None:
    print(f"{'site_id':<18} {'mode':<10} {'capacity':>9}  {'lat':>8} {'lon':>9}  name")
    print("-" * 100)
    for site in list_sites():
        print(
            f"{site.site_id:<18} {site.data_mode:<10} {site.capacity_kw:>7.1f}kW  "
            f"{site.latitude:>8.4f} {site.longitude:>9.4f}  {site.name}"
        )
    print()
    print("Measured sites carry published system metadata; simulated sites are illustrative.")


def refresh_curated() -> int:
    """Re-download every measured site and refresh the committed curated copies."""
    curated_dir = Path(settings.data_curated_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)

    real_sites = list_sites(data_mode="real")
    if not real_sites:
        logger.error("No measured sites in the registry.")
        return 1

    logger.info(
        "Refreshing %d curated dataset(s) for %s to %s …",
        len(real_sites),
        settings.real_data_start,
        settings.real_data_end,
    )

    for site in real_sites:
        frame, provenance = load_site_data(
            site.site_id,
            use_cache=False,
            start=settings.real_data_start,
            end=settings.real_data_end,
        )
        frame_path = curated_dir / f"telemetry_{site.site_id}.parquet"
        frame.to_parquet(frame_path, index=False)
        save_provenance([provenance], curated_dir / f"telemetry_{site.site_id}.provenance.json")
        size_mb = frame_path.stat().st_size / 1e6
        logger.info(
            "  %-14s %6d rows  %.2f MB  %s -> %s",
            site.site_id,
            len(frame),
            size_mb,
            provenance.start[:10],
            provenance.end[:10],
        )

    total_mb = sum(p.stat().st_size for p in curated_dir.glob("*")) / 1e6
    logger.info(
        "Curated datasets refreshed (%.1f MB total). Review the diff before committing.", total_mb
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_sites:
        print_registry()
        return 0

    if args.refresh_curated:
        return refresh_curated()

    if not args.site_id:
        logger.error("Pass --site-id, --refresh-curated, or --list-sites.")
        return 1

    if args.site_id not in load_sites():
        logger.error("Unknown site '%s'. Run --list-sites.", args.site_id)
        return 1

    kwargs: dict = {"demo": args.demo}
    if args.start:
        kwargs["start"] = args.start
    if args.end:
        kwargs["end"] = args.end

    frame, provenance = load_site_data(args.site_id, use_cache=False, **kwargs)
    logger.info(
        "%s: %d rows, %s to %s (%s)",
        args.site_id,
        len(frame),
        provenance.start,
        provenance.end,
        provenance.data_mode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
