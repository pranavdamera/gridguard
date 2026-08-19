"""
Site registry: the catalogue of every asset GridGuard monitors.

The source of truth is ``config/sites.csv`` relative to the repo root, located
by walking up from this file so it resolves regardless of working directory.

Two kinds of site coexist in one registry, distinguished by ``data_mode``:

``real``       Backed by measured telemetry from a public dataset. Coordinates,
               capacity, tilt and azimuth come from the upstream system
               metadata, and ``source_system_id`` links back to it.

``synthetic``  A simulated DMV campus asset used for controlled fault-injection
               demonstrations. Institution names are illustrative — no named
               institution supplied telemetry to this project.

Keeping both in one registry (rather than two files) means fleet views, the map,
and spatial neighbour logic all operate on a single uniform collection, while
``data_mode`` keeps the provenance distinction explicit everywhere it surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # src/gridguard/sites/ -> repo root
_SITES_CSV = _REPO_ROOT / "config" / "sites.csv"

#: Shown wherever synthetic data is presented to a human.
SYNTHETIC_DISCLAIMER = (
    "Simulated demonstration data. Institution names and approximate locations "
    "are used for illustrative purposes. This is not measured operational data "
    "from these institutions."
)

#: Shown wherever measured data is presented to a human.
REAL_DATA_NOTE = (
    "Measured photovoltaic telemetry from NREL PVDAQ, distributed via the "
    "Open Energy Data Initiative (OEDI) data lake."
)


@dataclass(frozen=True)
class Site:
    """One monitored PV asset."""

    site_id: str
    name: str
    region: str
    data_mode: str  # "real" | "synthetic"
    latitude: float
    longitude: float
    capacity_kw: float
    elevation_m: float = 0.0
    capacity_basis: str = ""
    tilt_deg: float | None = None
    azimuth_deg: float | None = None
    source_system_id: int | None = None
    notes: str = field(default="")

    @property
    def is_real(self) -> bool:
        return self.data_mode == "real"

    @property
    def disclaimer(self) -> str:
        """The provenance statement that must accompany this site in any UI."""
        return REAL_DATA_NOTE if self.is_real else SYNTHETIC_DISCLAIMER

    @property
    def has_mount_geometry(self) -> bool:
        """Whether tilt and azimuth are known, so a physics model can be run."""
        return self.tilt_deg is not None and self.azimuth_deg is not None


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    f = _optional_float(value)
    return int(f) if f is not None else None


@lru_cache(maxsize=1)
def load_sites(csv_path: Path | None = None) -> dict[str, Site]:
    """Return ``{site_id: Site}``, cached after first load."""
    path = Path(csv_path) if csv_path else _SITES_CSV
    if not path.exists():
        raise FileNotFoundError(f"Sites registry not found at {path}")

    df = pd.read_csv(path)

    required = {"site_id", "name", "data_mode", "latitude", "longitude", "capacity_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    sites: dict[str, Site] = {}
    for _, row in df.iterrows():
        mode = str(row["data_mode"]).strip()
        if mode not in ("real", "synthetic"):
            raise ValueError(
                f"Site '{row['site_id']}' has data_mode={mode!r}; expected 'real' or 'synthetic'."
            )
        sites[row["site_id"]] = Site(
            site_id=str(row["site_id"]),
            name=str(row["name"]),
            region=str(row.get("region", "") or ""),
            data_mode=mode,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            capacity_kw=float(row["capacity_kw"]),
            elevation_m=_optional_float(row.get("elevation_m")) or 0.0,
            capacity_basis=str(row.get("capacity_basis", "") or ""),
            tilt_deg=_optional_float(row.get("tilt_deg")),
            azimuth_deg=_optional_float(row.get("azimuth_deg")),
            source_system_id=_optional_int(row.get("source_system_id")),
            notes=str(row.get("notes", "") or ""),
        )
    return sites


def get_site(site_id: str, csv_path: Path | None = None) -> Site:
    """Look up a site by ID. Raises KeyError if not found."""
    sites = load_sites(csv_path)
    if site_id not in sites:
        raise KeyError(f"Unknown site_id '{site_id}'. Available: {sorted(sites)}")
    return sites[site_id]


def list_site_ids(csv_path: Path | None = None) -> list[str]:
    return sorted(load_sites(csv_path).keys())


def list_sites(
    data_mode: str | None = None,
    csv_path: Path | None = None,
) -> list[Site]:
    """All sites, optionally filtered to one data mode, ordered by site_id."""
    sites = sorted(load_sites(csv_path).values(), key=lambda s: s.site_id)
    if data_mode is not None:
        sites = [s for s in sites if s.data_mode == data_mode]
    return sites


def sites_as_records(csv_path: Path | None = None) -> list[dict]:
    """All sites as JSON-serialisable dicts."""
    return [
        {
            "site_id": s.site_id,
            "name": s.name,
            "region": s.region,
            "data_mode": s.data_mode,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "elevation_m": s.elevation_m,
            "capacity_kw": s.capacity_kw,
            "capacity_basis": s.capacity_basis,
            "tilt_deg": s.tilt_deg,
            "azimuth_deg": s.azimuth_deg,
            "source_system_id": s.source_system_id,
            "disclaimer": s.disclaimer,
            "notes": s.notes,
        }
        for s in sorted(load_sites(csv_path).values(), key=lambda s: s.site_id)
    ]
