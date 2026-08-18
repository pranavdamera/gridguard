"""
Download or generate solar generation data.

Usage:
    python scripts/download_data.py --source synthetic
    python scripts/download_data.py --source synthetic --site-id gmu_fairfax
    python scripts/download_data.py --list-sites
    python scripts/download_data.py --source nrel

For NREL data, set NREL_API_KEY in your .env file first:
    https://developer.nrel.gov/signup/
"""

import argparse
import logging
import sys

from gridguard.ingestion.download import load_raw_data
from gridguard.sites.registry import load_sites

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/generate GridGuard training data")
    parser.add_argument(
        "--source",
        choices=["synthetic", "nrel"],
        default="synthetic",
        help="Data source: synthetic (default) or nrel (requires API key)",
    )
    parser.add_argument(
        "--site-id",
        default=None,
        metavar="SITE_ID",
        help="Site ID from config/sites.csv (e.g. gmu_fairfax). "
        "Uses site latitude and capacity for synthetic generation.",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="Print all available site IDs and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Inject a scripted underperformance event at 2023-06-15 09:00–12:15 for demo purposes.",
    )
    args = parser.parse_args()

    if args.list_sites:
        sites = load_sites()
        print(f"\n{'ID':<20}  {'Name':<45}  {'Lat':>7}  {'Lon':>8}  {'Cap (kW)':>9}")
        print("-" * 96)
        for sid, s in sorted(sites.items()):
            print(
                f"{sid:<20}  {s.name:<45}  {s.latitude:>7.4f}  {s.longitude:>8.4f}  {s.capacity_kw:>9.1f}"
            )
        sys.exit(0)

    df = load_raw_data(source=args.source, site_id=args.site_id, demo=args.demo)
    print(f"\nLoaded {len(df):,} rows  |  columns: {list(df.columns)}")
    print(df.head())
    print(f"\nDate range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    if "is_injected_fault" in df.columns:
        faults = df["is_injected_fault"].sum()
        print(f"Injected fault intervals: {faults:,} ({100*faults/len(df):.1f}%)")
    if args.site_id:
        print(f"Site: {args.site_id}")
