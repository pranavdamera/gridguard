"""
Site registry: load and query the local DMV site catalogue.

The source of truth is config/sites.csv relative to the repo root.
We locate it by walking up from this file's location so it works
regardless of the current working directory.

Each site row has:
  site_id, name, region, latitude, longitude, capacity_kw
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # src/gridguard/sites/registry.py → repo root
_SITES_CSV = _REPO_ROOT / "config" / "sites.csv"


@dataclass(frozen=True)
class Site:
    site_id: str
    name: str
    region: str
    latitude: float
    longitude: float
    capacity_kw: float


@lru_cache(maxsize=1)
def load_sites(csv_path: Path | None = None) -> dict[str, Site]:
    """Return dict[site_id → Site], cached after first load."""
    path = Path(csv_path) if csv_path else _SITES_CSV
    if not path.exists():
        raise FileNotFoundError(f"Sites registry not found at {path}")
    df = pd.read_csv(path)
    return {
        row["site_id"]: Site(
            site_id=row["site_id"],
            name=row["name"],
            region=row["region"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            capacity_kw=float(row["capacity_kw"]),
        )
        for _, row in df.iterrows()
    }


def get_site(site_id: str, csv_path: Path | None = None) -> Site:
    """Look up a site by ID. Raises KeyError if not found."""
    sites = load_sites(csv_path)
    if site_id not in sites:
        available = sorted(sites.keys())
        raise KeyError(f"Unknown site_id '{site_id}'. Available: {available}")
    return sites[site_id]


def list_site_ids(csv_path: Path | None = None) -> list[str]:
    return sorted(load_sites(csv_path).keys())
