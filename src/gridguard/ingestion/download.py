"""
Data ingestion: NREL PVDAQ (real) or synthetic solar generation.

Real data:  https://developer.nrel.gov/docs/solar/pvdaq-v3/
Synthetic:  Physically-motivated simulator — useful when API is slow or unavailable.

Site-aware mode:
  Pass site_id to load_raw_data() to generate data for a specific DMV site.
  Sites are defined in config/sites.csv and loaded via gridguard.sites.registry.
  Without a site_id the old behaviour is preserved (lat 38.8°N, 10 kW system).

TODO: Add support for Open Power System Data (https://open-power-system-data.org/)
TODO: Add DuckDB caching layer to avoid re-downloading on reruns
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gridguard.config import settings

logger = logging.getLogger(__name__)

# Default site parameters used when no site_id is provided (backwards-compat)
_DEFAULT_LATITUDE = 38.83  # ~Northern Virginia / DC
_DEFAULT_CAPACITY_KW = 10.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_raw_data(
    source: str | None = None,
    output_dir: Path | None = None,
    site_id: str | None = None,
) -> pd.DataFrame:
    """Download or generate raw solar generation data and save to output_dir.

    Args:
        source:     "synthetic" (default) or "nrel"
        output_dir: Directory for the parquet cache. Defaults to settings.data_processed_dir.
        site_id:    Optional site ID from config/sites.csv.  When provided, uses
                    site-specific latitude and capacity and caches under a separate
                    file so multiple sites can coexist in the same output_dir.

    Returns DataFrame with columns:
        timestamp, ac_power_kw, irradiance_wm2, temperature_c, wind_speed_ms
        [is_injected_fault — synthetic only]
        [site_id — when site_id is provided]
    """
    source = source or settings.data_source
    output_dir = Path(output_dir or settings.data_processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-site cache so different sites don't overwrite each other
    cache_name = f"raw_{site_id}.parquet" if site_id else "raw.parquet"
    cache_path = output_dir / cache_name

    if cache_path.exists():
        logger.info("Loading cached raw data from %s", cache_path)
        return pd.read_parquet(cache_path)

    if source == "nrel":
        df = _fetch_nrel_pvdaq()
    else:
        df = _generate_synthetic_dispatch(site_id=site_id)

    if site_id:
        df["site_id"] = site_id

    df.to_parquet(cache_path, index=False)
    logger.info("Saved %d rows to %s", len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# NREL PVDAQ
# ---------------------------------------------------------------------------


def _fetch_nrel_pvdaq() -> pd.DataFrame:
    """Pull 15-minute interval data from the NREL PVDAQ v3 API.

    Requires a free API key: https://developer.nrel.gov/signup/
    Set NREL_API_KEY in your .env file.

    TODO: Paginate to pull multiple years; currently pulls the most recent year.
    """
    base_url = "https://developer.nrel.gov/api/pvdaq/v3/data_file"
    params = {
        "api_key": settings.nrel_api_key,
        "system_id": settings.pvdaq_system_id,
        "aggregate": "15",  # 15-minute intervals
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
    }
    logger.info("Fetching NREL PVDAQ system_id=%d …", settings.pvdaq_system_id)
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))

    # Normalise column names — PVDAQ uses different names per system
    col_map = {
        "datetime": "timestamp",
        "ac_power": "ac_power_kw",
        "poa_irradiance": "irradiance_wm2",
        "air_temperature": "temperature_c",
        "wind_speed": "wind_speed_ms",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    _validate_schema(df)
    return df


# ---------------------------------------------------------------------------
# Synthetic data generator (site-aware)
# ---------------------------------------------------------------------------


def _generate_synthetic_dispatch(site_id: str | None) -> pd.DataFrame:
    """Route to site-specific or default synthetic generator."""
    if site_id is not None:
        from gridguard.sites.registry import get_site

        site = get_site(site_id)
        logger.info(
            "Generating synthetic data for site '%s' (lat=%.4f, cap=%.0f kW)",
            site.name,
            site.latitude,
            site.capacity_kw,
        )
        return _generate_synthetic(
            latitude=site.latitude,
            system_capacity_kw=site.capacity_kw,
        )

    logger.info(
        "Generating synthetic data with defaults (lat=%.2f, cap=%.0f kW). "
        "Use --site-id for a named site.",
        _DEFAULT_LATITUDE,
        _DEFAULT_CAPACITY_KW,
    )
    return _generate_synthetic(
        latitude=_DEFAULT_LATITUDE,
        system_capacity_kw=_DEFAULT_CAPACITY_KW,
    )


def _generate_synthetic(
    start: str = "2022-01-01",
    end: str = "2023-12-31",
    freq: str = "15min",
    latitude: float = _DEFAULT_LATITUDE,
    system_capacity_kw: float = _DEFAULT_CAPACITY_KW,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate physically-motivated 15-minute synthetic solar data.

    Args:
        latitude:           Degrees north. Affects sun elevation and seasonal swing.
        system_capacity_kw: AC nameplate capacity of the PV system.

    Model:
      1. Clear-sky irradiance from sun elevation angle at the given latitude.
      2. Cloud attenuation with log-normal noise.
      3. Temperature with diurnal + seasonal cycle calibrated to ~Northern Virginia.
      4. Panel efficiency drops with temperature (0.4 %/°C above 25°C).
      5. Injected faults: random day-long underperformance events (~5% of days).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq)
    n = len(idx)

    # --- Sun geometry (simplified, latitude-parameterised) ---
    doy = idx.day_of_year.values  # 1–365
    hour = idx.hour.values + idx.minute.values / 60
    lat_rad = np.radians(latitude)
    declination = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    hour_angle = np.radians(15 * (hour - 12))
    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, 0, 1)

    # Clear-sky irradiance (W/m²)
    ghi_clearsky = 1000 * cos_zenith

    # Cloud factor — log-normal with seasonal autocorrelation
    cloud_base = rng.lognormal(mean=0, sigma=0.4, size=n)
    cloud_factor = np.clip(cloud_base, 0, 1.5)
    irradiance = np.clip(ghi_clearsky * cloud_factor, 0, 1200)

    # Night-time clamp
    irradiance[cos_zenith < 0.05] = 0.0

    # --- Temperature (Northern Virginia climate profile) ---
    # Seasonal mean 5°C in winter, 28°C in summer; diurnal swing ~±5°C
    temp_seasonal = 16 + 11 * np.sin(np.radians(360 / 365 * (doy - 80)))
    temp_diurnal = 5 * np.sin(np.radians(15 * (hour - 14)))
    temperature = temp_seasonal + temp_diurnal + rng.normal(0, 1.5, n)

    # --- AC power ---
    efficiency = 1.0 - 0.004 * np.maximum(0, temperature - 25)
    ac_power = system_capacity_kw * (irradiance / 1000) * efficiency
    ac_power = np.clip(ac_power, 0, system_capacity_kw)

    # --- Wind speed ---
    wind_speed = np.abs(rng.normal(3.5, 1.5, n))

    # --- Inject faults (anomalies) ---
    # ~5% of days have underperformance events (soiling, partial shading, inverter trip)
    unique_days = pd.Series(idx.date).unique()
    fault_days = rng.choice(unique_days, size=int(0.05 * len(unique_days)), replace=False)
    fault_mask = pd.Series(idx.date).isin(fault_days).values
    # Fault reduces output by 40–80%
    fault_severity = rng.uniform(0.2, 0.6, size=n)
    ac_power = np.where(fault_mask, ac_power * fault_severity, ac_power)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "ac_power_kw": ac_power.round(4),
            "irradiance_wm2": irradiance.round(2),
            "temperature_c": temperature.round(2),
            "wind_speed_ms": wind_speed.round(2),
            "is_injected_fault": fault_mask,  # ground truth label for evaluation
        }
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_schema(df: pd.DataFrame) -> None:
    required = {"timestamp", "ac_power_kw", "irradiance_wm2", "temperature_c"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
