"""
NREL PVDAQ access via the OEDI data lake.

Why not the PVDAQ API?
----------------------
The legacy ``developer.nrel.gov/api/pvdaq/v3`` endpoints that earlier versions
of GridGuard targeted have been retired. The PVDAQ *data* remains public and is
distributed through the Open Energy Data Initiative (OEDI) data lake as an
anonymously-readable S3 bucket, which this module reads over plain HTTPS.

Consequences, all of them good for a public demo:

* **No API key is required.** Nothing in the real-data path needs a credential.
* Data is partitioned one file per system per day, so a curated slice can be
  assembled without downloading the whole archive.

Layout
------
``systems.csv``   ``pvdaq/csv/systems.csv``
``metadata``      ``pvdaq/csv/system_metadata/{system_id}_system_metadata.json``
``metrics``       ``pvdaq/parquet/metrics/metrics__system_{id}__part000.parquet``
``telemetry``     ``pvdaq/parquet/pvdata/system_id={id}/year={Y}/month={M}/day={D}/
                    system_{id}__date_{Y}_{MM}_{DD}.snappy.000.parquet``

Note the asymmetry in the telemetry path: directory components are unpadded
(``month=6``) while the filename is zero-padded (``date_2016_06_01``).

Channel resolution
------------------
Telemetry parquet is *long* — ``(measured_on, utc_measured_on, metric_id,
value)`` — and carries no channel names. The per-system **metrics table** is the
authority: it maps each ``metric_id`` to a sensor name, a semantic
``common_name``, declared ``units``, and a linear ``calc_scale``/``calc_offset``.

Selecting on declared units matters more than it might appear. On the NIST
systems, two of the four irradiance channels are raw pyranometer **millivolts**
and two are calibrated **W/m²**; and the channel named ``ac_power_meter_1864``
is actually reactive energy in kVARh. Picking channels by name alone silently
yields nonsense, so this module selects on ``units`` first and treats names as a
tiebreaker.
"""

from __future__ import annotations

import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OEDI_BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
PVDAQ_LANDING_PAGE = "https://openei.org/wiki/PVDAQ"
SYSTEMS_CSV_URL = f"{OEDI_BUCKET_URL}/pvdaq/csv/systems.csv"

LICENSE_NOTE = (
    "NREL PVDAQ data published via the Open Energy Data Initiative (OEDI) as a "
    "public dataset. Attribution to NREL/OEDI required; consult the dataset "
    "landing page for current terms before redistribution."
)

_REQUEST_TIMEOUT = 60
_MAX_RETRIES = 4
_BACKOFF_SECONDS = 2.0


class OEDIError(RuntimeError):
    """Raised when the OEDI data lake cannot satisfy a request."""


@dataclass(frozen=True)
class PVDAQSystem:
    """Metadata for one PVDAQ system, as published in its metadata JSON."""

    system_id: int
    public_name: str
    latitude: float | None
    longitude: float | None
    elevation_m: float | None
    dc_capacity_kw: float | None
    tilt_deg: float | None
    azimuth_deg: float | None
    mount_count: int
    location: str
    module_model: str
    module_count: int | None
    inverter_model: str

    @property
    def has_mount_geometry(self) -> bool:
        return self.tilt_deg is not None and self.azimuth_deg is not None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _get(url: str, *, expect_binary: bool = False) -> bytes | str:
    """GET with bounded exponential backoff. Raises OEDIError on give-up."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                raise OEDIError(f"Not found in OEDI data lake: {url}")
            resp.raise_for_status()
            return resp.content if expect_binary else resp.text
        except OEDIError:
            raise
        except Exception as exc:  # network flake, 5xx, timeout
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_SECONDS * (2**attempt)
                logger.debug("GET %s failed (%s); retrying in %.0fs", url, exc, delay)
                time.sleep(delay)
    raise OEDIError(f"Failed to fetch {url} after {_MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Catalogue and metadata
# ---------------------------------------------------------------------------


def list_systems() -> pd.DataFrame:
    """Return the PVDAQ system catalogue (system_id, name, lat/lon, elevation)."""
    return pd.read_csv(io.StringIO(str(_get(SYSTEMS_CSV_URL))))


def fetch_system_metadata(system_id: int) -> PVDAQSystem:
    """Fetch and parse one system's metadata document.

    Carries exactly what a physics model needs: coordinates, mount tilt and
    azimuth, module model and count, and DC nameplate.
    """
    url = f"{OEDI_BUCKET_URL}/pvdaq/csv/system_metadata/{system_id}_system_metadata.json"
    raw = json.loads(str(_get(url)))

    system = raw.get("System", {}) or {}
    site = raw.get("Site", {}) or {}
    mounts = raw.get("Mount", {}) or {}
    inverters = raw.get("Inverters", {}) or {}
    modules = raw.get("Modules", {}) or {}

    # A system may have several mounts (e.g. an east/west canopy). The first by
    # key order is reported; callers should check `mount_count` before treating
    # tilt/azimuth as describing the whole array.
    mount = next(iter(mounts.values()), {})
    inverter = next(iter(inverters.values()), {})
    module = next(iter(modules.values()), {})

    return PVDAQSystem(
        system_id=int(system.get("system_id", system_id)),
        public_name=str(system.get("public_name", f"system_{system_id}")),
        latitude=_as_float(site.get("latitude")),
        longitude=_as_float(site.get("longitude")),
        elevation_m=_as_float(site.get("elevation")),
        dc_capacity_kw=_as_float(system.get("power")),
        tilt_deg=_as_float(mount.get("tilt")),
        azimuth_deg=_as_float(mount.get("azimuth")),
        mount_count=len(mounts),
        location=str(site.get("location", "") or ""),
        module_model=str(module.get("model", "") or ""),
        module_count=_as_int(module.get("quantity")),
        inverter_model=str(inverter.get("model", "") or ""),
    )


def fetch_metrics(system_id: int) -> pd.DataFrame:
    """Fetch the per-system metric dictionary.

    Columns of interest: ``metric_id``, ``sensor_name``, ``common_name``,
    ``units``, ``calc_scale``, ``calc_offset``, ``source_type``.
    """
    url = f"{OEDI_BUCKET_URL}/pvdaq/parquet/metrics/metrics__system_{system_id}__part000.parquet"
    blob = _get(url, expect_binary=True)
    metrics = pd.read_parquet(io.BytesIO(bytes(blob)))
    if "metric_id" not in metrics.columns:
        raise OEDIError(f"Metrics table for system {system_id} has no metric_id column.")
    return metrics


def _as_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    f = _as_float(value)
    return int(f) if f is not None else None


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One selected measurement channel."""

    metric_id: int
    sensor_name: str
    common_name: str
    units: str
    scale: float
    offset: float

    def apply(self, values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce") * self.scale + self.offset

    def describe(self) -> str:
        return f"{self.sensor_name} (metric_id={self.metric_id}, {self.units})"


@dataclass(frozen=True)
class ResolvedChannels:
    """The channel chosen for each canonical measurement."""

    ac_power: Channel
    irradiance: Channel
    irradiance_kind: str  # "plane-of-array" | "global horizontal"
    temperature: Channel | None
    wind_speed: Channel | None
    module_temperature: Channel | None

    def all_channels(self) -> list[Channel]:
        found = [self.ac_power, self.irradiance]
        found += [c for c in (self.temperature, self.wind_speed, self.module_temperature) if c]
        return found

    def as_note(self) -> str:
        parts = [
            f"ac_power={self.ac_power.describe()}",
            f"irradiance={self.irradiance.describe()} [{self.irradiance_kind}]",
            f"temperature={self.temperature.describe() if self.temperature else 'unavailable'}",
            f"wind_speed={self.wind_speed.describe() if self.wind_speed else 'unavailable'}",
        ]
        return "; ".join(parts)


def _to_channel(row: pd.Series) -> Channel:
    scale = row.get("calc_scale")
    offset = row.get("calc_offset")
    return Channel(
        metric_id=int(row["metric_id"]),
        sensor_name=str(row.get("sensor_name", "") or ""),
        common_name=str(row.get("common_name", "") or ""),
        units=str(row.get("units", "") or ""),
        scale=1.0 if scale is None or pd.isna(scale) else float(scale),
        offset=0.0 if offset is None or pd.isna(offset) else float(offset),
    )


def _pick(candidates: pd.DataFrame, prefer_contains: str | None = None) -> pd.Series | None:
    """Deterministically pick one metric row, optionally preferring a name match."""
    if candidates.empty:
        return None
    ordered = candidates.sort_values("metric_id")
    if prefer_contains:
        preferred = ordered[
            ordered["sensor_name"].str.contains(prefer_contains, case=False, na=False)
        ]
        if not preferred.empty:
            return preferred.iloc[0]
    return ordered.iloc[0]


def resolve_channels(metrics: pd.DataFrame) -> ResolvedChannels:
    """Choose the canonical channel for each measurement from the metrics table.

    Selection is driven by declared ``units`` because upstream ``common_name``
    values are not always reliable — wind speed on the NIST systems is filed
    under "AC other", and several "AC power" rows are cumulative energy in kWh
    or reactive power in kVARh.

    Preferences, in order:

    * **AC power** — instantaneous power in ``kW``, from an inverter where one
      exists (a revenue meter is the fallback).
    * **Irradiance** — plane-of-array in ``W/m²`` before global horizontal.
      Channels published in raw ``mV`` are rejected outright: converting them
      needs an instrument calibration constant that the dataset does not
      publish, and guessing one would corrupt every downstream number.
    * **Temperature** — ambient air in ``C`` (module temperature is captured
      separately, for the physics model).
    * **Wind speed** — average, not maximum, in ``m/s``.
    """
    required = {"metric_id", "units"}
    if not required.issubset(metrics.columns):
        raise OEDIError(f"Metrics table missing columns {required - set(metrics.columns)}")

    m = metrics.copy()
    for col in ("sensor_name", "common_name", "units", "source_type"):
        if col not in m.columns:
            m[col] = ""
        m[col] = m[col].fillna("").astype(str)

    # --- AC power -----------------------------------------------------------
    power = m[(m["common_name"] == "AC power") & (m["units"] == "kW")]
    inverter_power = power[power["source_type"].str.upper() == "INVERTER"]
    power_row = _pick(inverter_power if not inverter_power.empty else power)
    if power_row is None:
        raise OEDIError(
            "No instantaneous AC power channel in kW found. Available AC power units: "
            f"{sorted(m[m['common_name'] == 'AC power']['units'].unique())}"
        )

    # --- irradiance ---------------------------------------------------------
    irradiance_all = m[m["common_name"].str.startswith("Irradiance")]
    calibrated = irradiance_all[irradiance_all["units"].str.replace(" ", "") == "W/m^2"]
    if calibrated.empty:
        raise OEDIError(
            "No irradiance channel published in W/m^2. Found units: "
            f"{sorted(irradiance_all['units'].unique())}. Channels in raw mV cannot be "
            "converted without the instrument calibration constant, which this dataset "
            "does not publish."
        )
    poa = calibrated[calibrated["common_name"].str.contains("POA", na=False)]
    if not poa.empty:
        irradiance_row, irradiance_kind = _pick(poa), "plane-of-array"
    else:
        irradiance_row, irradiance_kind = _pick(calibrated), "global horizontal"

    # --- ambient temperature ------------------------------------------------
    temp_row = _pick(m[(m["common_name"] == "Temperature ambient") & (m["units"] == "C")])

    # --- wind speed ---------------------------------------------------------
    # `common_name` is unreliable here, so match on units plus sensor name and
    # prefer the averaging channel over the max.
    wind_candidates = m[
        (m["units"].str.lower() == "m/s")
        & (m["sensor_name"].str.contains("wind", case=False, na=False))
    ]
    wind_row = _pick(wind_candidates, prefer_contains="ave")

    # --- module temperature (optional, for the physics model) ---------------
    module_row = _pick(m[(m["common_name"] == "Temperature module") & (m["units"] == "C")])

    return ResolvedChannels(
        ac_power=_to_channel(power_row),
        irradiance=_to_channel(irradiance_row),
        irradiance_kind=irradiance_kind,
        temperature=_to_channel(temp_row) if temp_row is not None else None,
        wind_speed=_to_channel(wind_row) if wind_row is not None else None,
        module_temperature=_to_channel(module_row) if module_row is not None else None,
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def daily_parquet_url(system_id: int, day: date) -> str:
    """Build the OEDI object URL for one system-day of telemetry."""
    return (
        f"{OEDI_BUCKET_URL}/pvdaq/parquet/pvdata/"
        f"system_id={system_id}/year={day.year}/month={day.month}/day={day.day}/"
        f"system_{system_id}__date_{day.year}_{day.month:02d}_{day.day:02d}.snappy.000.parquet"
    )


def fetch_day(
    system_id: int,
    day: date,
    metric_ids: set[int] | None = None,
) -> pd.DataFrame | None:
    """Fetch one day of telemetry in long form, filtered to ``metric_ids``.

    Returns None when the day is absent. Missing days are normal — instruments
    go offline — so an absent object is reported rather than raised, letting the
    caller assemble whatever coverage actually exists.
    """
    url = daily_parquet_url(system_id, day)
    try:
        blob = _get(url, expect_binary=True)
    except OEDIError:
        logger.debug("No data for system %d on %s", system_id, day)
        return None

    frame = pd.read_parquet(io.BytesIO(bytes(blob)))
    if metric_ids is not None and "metric_id" in frame.columns:
        frame = frame[frame["metric_id"].isin(metric_ids)]
    return frame


def fetch_range(
    system_id: int,
    start: date,
    end: date,
    *,
    metric_ids: set[int] | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch a contiguous date range in long form, skipping unpublished days.

    Days are fetched concurrently because each is an independent object and a
    curated range can span hundreds of files. Filtering to ``metric_ids`` inside
    the worker keeps peak memory proportional to the channels actually needed
    rather than to all ~100 published per system.
    """
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    logger.info(
        "Fetching PVDAQ system %d: %s to %s (%d days) from the OEDI data lake …",
        system_id,
        start,
        end,
        len(days),
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(lambda d: fetch_day(system_id, d, metric_ids), days))

    present = [f for f in frames if f is not None and not f.empty]
    if not present:
        raise OEDIError(
            f"No PVDAQ data available for system {system_id} between {start} and {end}. "
            "Check the system id and that the range falls within its operating period."
        )

    missing = len(days) - len(present)
    if missing:
        logger.info("  %d/%d days had no published data and were skipped.", missing, len(days))

    combined = pd.concat(present, ignore_index=True)
    logger.info("  Retrieved %d samples across %d days.", len(combined), len(present))
    return combined


def pivot_channels(long_frame: pd.DataFrame, channels: ResolvedChannels) -> pd.DataFrame:
    """Pivot long telemetry to one column per resolved channel.

    Applies each channel's ``calc_scale``/``calc_offset`` and returns a frame
    indexed by ``measured_on`` with canonical column names.
    """
    wanted = {c.metric_id: c for c in channels.all_channels()}
    subset = long_frame[long_frame["metric_id"].isin(wanted)]
    if subset.empty:
        raise OEDIError("None of the resolved channels appear in the telemetry.")

    wide = subset.pivot_table(
        index="measured_on",
        columns="metric_id",
        values="value",
        aggfunc="mean",
    ).sort_index()

    name_for = {
        channels.ac_power.metric_id: "ac_power_kw",
        channels.irradiance.metric_id: "irradiance_wm2",
    }
    if channels.temperature:
        name_for[channels.temperature.metric_id] = "temperature_c"
    if channels.wind_speed:
        name_for[channels.wind_speed.metric_id] = "wind_speed_ms"
    if channels.module_temperature:
        name_for[channels.module_temperature.metric_id] = "module_temperature_c"

    out = pd.DataFrame(index=wide.index)
    for metric_id, channel in wanted.items():
        if metric_id in wide.columns:
            out[name_for[metric_id]] = channel.apply(wide[metric_id])

    out.index.name = "timestamp"
    return out.reset_index()


def detect_utc_offset_hours(long_frame: pd.DataFrame) -> int | None:
    """Derive the fixed local-standard-time offset from the published columns.

    PVDAQ publishes both ``measured_on`` (local) and ``utc_measured_on``, so the
    offset is observable rather than assumed. A single constant offset across a
    range that spans a daylight-saving boundary confirms the local column is
    standard time with no seasonal shift.
    """
    if "utc_measured_on" not in long_frame.columns:
        return None
    local = pd.to_datetime(long_frame["measured_on"])
    utc = pd.to_datetime(long_frame["utc_measured_on"])
    offsets = ((local - utc).dt.total_seconds() / 3600).round().dropna().unique()
    if len(offsets) != 1:
        logger.warning("Multiple UTC offsets present in one range: %s", sorted(offsets))
        return None
    return int(offsets[0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: A clear-sky index at the 95th percentile should land near 1.0 for a
#: calibrated irradiance sensor observed over several days.
PLAUSIBLE_CSI_RANGE = (0.55, 1.35)


def clearsky_reference(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
    utc_offset_hours: int = -5,
    tilt_deg: float | None = None,
    azimuth_deg: float | None = None,
) -> np.ndarray:
    """Clear-sky irradiance in the reference plane of the sensor being checked.

    For a horizontal (GHI) sensor this is clear-sky GHI. For a tilted
    plane-of-array sensor the clear-sky components are transposed into the
    array plane, because a tilted surface receives substantially more than the
    horizontal in winter — at 39°N with a 20° tilt the December ratio exceeds
    1.5. Comparing a tilted sensor against a horizontal reference would flag a
    perfectly good instrument as broken every winter.
    """
    import pvlib

    # Etc/GMT signs are inverted relative to the usual convention: UTC-5 is
    # "Etc/GMT+5". A fixed-offset zone keeps solar position continuous across
    # the daylight-saving boundary.
    tz = f"Etc/GMT{-utc_offset_hours:+d}"
    location = pvlib.location.Location(latitude, longitude, tz=tz, altitude=altitude or 0.0)

    index = pd.DatetimeIndex(pd.to_datetime(timestamps)).tz_localize(tz)
    clearsky = location.get_clearsky(index, model="ineichen")

    if tilt_deg is None or azimuth_deg is None:
        return clearsky["ghi"].to_numpy()

    solar_position = location.get_solarposition(index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        model="isotropic",
    )
    return poa["poa_global"].fillna(0.0).to_numpy()


def validate_irradiance_against_clearsky(
    timestamps: pd.Series,
    irradiance_wm2: pd.Series,
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
    utc_offset_hours: int = -5,
    tilt_deg: float | None = None,
    azimuth_deg: float | None = None,
) -> tuple[float, str]:
    """Sanity-check a W/m² irradiance channel against a clear-sky model.

    This is a **check, not a calibration**: the channel is already published in
    physical units, and this confirms the selected sensor really behaves like
    irradiance at the stated coordinates, orientation and timezone. It catches a
    wrong channel, wrong coordinates, or a timezone error — each of which would
    otherwise surface as a plausible-looking but wrong model.

    Pass ``tilt_deg``/``azimuth_deg`` for a plane-of-array sensor so the
    reference is computed in the same plane.

    Returns:
        ``(clear_sky_index, note)`` for the provenance record.

    Raises:
        OEDIError: When the clear-sky index is implausible.
    """
    reference = clearsky_reference(
        timestamps,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        utc_offset_hours=utc_offset_hours,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
    )

    values = pd.to_numeric(irradiance_wm2, errors="coerce").to_numpy(dtype=float)
    daytime = (reference > 200) & np.isfinite(values)
    if daytime.sum() < 50:
        raise OEDIError(
            f"Only {int(daytime.sum())} daytime samples available; need at least 50 to "
            "validate the irradiance channel."
        )

    plane = (
        f"plane-of-array (tilt {tilt_deg:g}°, azimuth {azimuth_deg:g}°)"
        if tilt_deg is not None and azimuth_deg is not None
        else "horizontal"
    )
    csi = float(np.percentile(values[daytime] / reference[daytime], 95))
    if not PLAUSIBLE_CSI_RANGE[0] <= csi <= PLAUSIBLE_CSI_RANGE[1]:
        raise OEDIError(
            f"Irradiance channel failed its clear-sky check: 95th-percentile clear-sky "
            f"index is {csi:.2f} against a {plane} reference, outside the plausible range "
            f"{PLAUSIBLE_CSI_RANGE}. This usually means the wrong channel was selected, the "
            "site coordinates or orientation are wrong, or the timezone offset is wrong."
        )

    note = (
        f"Validated against a pvlib Ineichen clear-sky model transposed to the sensor's "
        f"{plane} reference: 95th-percentile clear-sky index {csi:.2f} (a calibrated sensor "
        f"sits near 1.0)."
    )
    logger.info("Irradiance clear-sky check passed (95th-pct CSI %.2f, %s).", csi, plane)
    return csi, note
