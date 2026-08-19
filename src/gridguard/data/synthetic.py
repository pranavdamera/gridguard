"""
Synthetic DMV fleet generator.

Produces physically-motivated telemetry for the illustrative DMV campus fleet.
This exists to demonstrate the detection pipeline against *known* ground truth,
which measured telemetry cannot provide (see :mod:`gridguard.data.faults`).

Nothing here is measured data. Institution names in the registry are
illustrative; no named institution supplied telemetry to this project. Every
frame this module produces carries ``data_mode="synthetic"``.

Physical model
--------------
1. Clear-sky global horizontal irradiance from pvlib's Ineichen model at the
   site's real coordinates and elevation.
2. Transposition to the plane of array using the site's tilt and azimuth
   (isotropic sky diffuse — adequate for a demonstration signal).
3. Cloud attenuation as a first-order autoregressive process, so cloudiness
   persists across intervals the way weather actually does rather than
   flickering independently each sample.
4. Ambient temperature from a seasonal plus diurnal cycle fitted loosely to the
   Mid-Atlantic climate, with noise.
5. AC power from plane-of-array irradiance with a temperature efficiency derate
   and an inverter AC cap.

A deliberate caveat
-------------------
This generator and the physics baseline in :mod:`gridguard.models.physics` both
rest on the same pvlib clear-sky model. The physics baseline will therefore look
near-perfect on synthetic data, because it is being asked to invert the process
that generated it. That number is not evidence of anything. The physics
baseline's only honest evaluation is on measured telemetry, and the model card
reports it only there.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from gridguard.data.faults import FaultEvent, FaultType, apply_faults, build_evaluation_scenario
from gridguard.data.provenance import DatasetProvenance
from gridguard.data.schema import validate_canonical
from gridguard.sites.registry import SYNTHETIC_DISCLAIMER, Site

logger = logging.getLogger(__name__)

#: Fixed offset for the DMV region's local standard time (EST).
UTC_OFFSET_HOURS = -5

#: Default simulated period.
#:
#: Deliberately aligned with the measured datasets' window so that the fleet
#: view can show measured and simulated sites on one timeline. Without this
#: alignment a single fleet snapshot would always report one of the two fleets
#: as "no data", since the two would never overlap in time. Simulated dates are
#: arbitrary, so it is the simulation that moves.
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2016-12-31"

#: The scripted demonstration event. Deterministic so a recorded walkthrough
#: reproduces exactly. Placed after the train/test split date so it lands in the
#: held-out evaluation period the detector is actually scored on.
DEMO_SITE_ID = "gmu_fairfax"
DEMO_DATE = "2016-11-15"
DEMO_START_HOUR = 9.0
DEMO_END_HOUR = 12.25
DEMO_SEVERITY = 0.70

_TEMP_COEFF_PER_C = 0.004  # fractional efficiency loss per degree above 25 C
_SYSTEM_DERATE = 0.85  # soiling, wiring, inverter efficiency, mismatch


def generate_site_telemetry(
    site: Site,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    freq: str = "15min",
    seed: int = 42,
    inject_scenario: bool = True,
    demo: bool = False,
) -> tuple[pd.DataFrame, DatasetProvenance]:
    """Generate canonical synthetic telemetry for one site.

    Args:
        site:            Registry entry. Must have ``data_mode == "synthetic"``.
        start, end:      Simulated period, inclusive.
        freq:            Sampling interval.
        seed:            Base RNG seed. Combined with the site id so that each
                         site gets independent weather while the whole fleet
                         remains reproducible.
        inject_scenario: Inject the full fault taxonomy for detector evaluation.
        demo:            Additionally inject the scripted demonstration event
                         (only meaningful for the demo site).

    Returns:
        ``(frame, provenance)`` where frame satisfies the canonical schema and
        additionally carries the ground-truth label columns.
    """
    if site.data_mode != "synthetic":
        raise ValueError(
            f"generate_site_telemetry is for synthetic sites; '{site.site_id}' is {site.data_mode}."
        )

    # Site-specific but deterministic: same fleet every run, different weather
    # per site rather than seven copies of one cloud sequence.
    #
    # A stable digest, not the builtin hash(): Python salts string hashing per
    # process unless PYTHONHASHSEED is pinned, so hash() here would silently
    # produce different "deterministic" data on every run.
    site_seed = seed + _stable_site_offset(site.site_id)
    rng = np.random.default_rng(site_seed)

    index = pd.date_range(start, end, freq=freq)
    ghi, poa = _clear_sky_poa(index, site)

    # --- cloud attenuation: AR(1) so cloudiness persists -------------------
    cloud = _autocorrelated_cloud_factor(len(index), rng)
    irradiance = np.clip(poa * cloud, 0, 1400)
    irradiance[ghi <= 1.0] = 0.0  # no plane-of-array signal without sun

    # --- ambient temperature ----------------------------------------------
    doy = index.day_of_year.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    seasonal = 15.0 + 11.0 * np.sin(np.radians(360 / 365 * (doy - 105)))
    diurnal = 5.0 * np.sin(np.radians(15 * (hour - 15)))
    temperature = seasonal + diurnal + rng.normal(0, 1.4, len(index))

    # --- wind --------------------------------------------------------------
    wind_speed = np.abs(rng.normal(3.2, 1.4, len(index)))

    # --- AC power ----------------------------------------------------------
    # Cell temperature rises above ambient roughly in proportion to irradiance.
    cell_temp = temperature + irradiance / 1000.0 * 28.0
    efficiency = 1.0 - _TEMP_COEFF_PER_C * np.maximum(0.0, cell_temp - 25.0)

    # Slowly-drifting performance factor: soiling accumulation and washing,
    # spectral mismatch, sensor calibration drift. Without this the simulated
    # array is a deterministic function of irradiance, which is both unrealistic
    # and quietly misleading — it makes lagged power carry no information, so
    # the leakage hazard that motivates the weather-only feature set could not
    # arise. Real arrays deviate from their weather-implied output in ways that
    # persist for days, which is exactly what makes a lag-aware model dangerous
    # as an anomaly baseline.
    performance = _performance_drift(len(index), rng)

    ac_power = site.capacity_kw * (irradiance / 1000.0) * efficiency * _SYSTEM_DERATE * performance
    # Measurement noise on the power channel itself, proportional to output.
    ac_power *= 1.0 + rng.normal(0, 0.01, len(index))
    ac_power = np.clip(ac_power, 0, site.capacity_kw)

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "site_id": site.site_id,
            "ac_power_kw": np.round(ac_power, 4),
            "irradiance_wm2": np.round(irradiance, 2),
            "temperature_c": np.round(temperature, 2),
            "wind_speed_ms": np.round(wind_speed, 2),
            "data_mode": "synthetic",
        }
    )

    events: list[FaultEvent] = []
    if inject_scenario:
        events.extend(build_evaluation_scenario(frame["timestamp"], seed=site_seed))
    if demo and site.site_id == DEMO_SITE_ID:
        events.append(_demo_event())

    if events:
        frame = apply_faults(frame, events, capacity_kw=site.capacity_kw)
    else:
        frame["is_injected_fault"] = False
        frame["injected_fault_type"] = ""
        frame["is_generation_loss"] = False
        frame["ac_power_baseline_kw"] = frame["ac_power_kw"]

    frame = validate_canonical(frame)

    provenance = DatasetProvenance(
        site_id=site.site_id,
        data_mode="synthetic",
        dataset="GridGuard DMV simulator",
        site_name=site.name,
        latitude=site.latitude,
        longitude=site.longitude,
        elevation_m=site.elevation_m,
        capacity_kw=site.capacity_kw,
        capacity_basis=site.capacity_basis or "Illustrative estimate",
        tilt_deg=site.tilt_deg,
        azimuth_deg=site.azimuth_deg,
        interval_minutes=_freq_minutes(freq),
        native_interval_minutes=_freq_minutes(freq),
        start=str(frame["timestamp"].min()),
        end=str(frame["timestamp"].max()),
        timezone_note=(
            f"Local standard time, fixed UTC{UTC_OFFSET_HOURS:+d}, no daylight-saving shift."
        ),
        row_count=len(frame),
        weather_source="Simulated (pvlib Ineichen clear-sky with autoregressive cloud attenuation)",
        irradiance_kind="plane-of-array (modelled)",
        irradiance_channel="simulated",
        source_url="",
        license_note="Generated by this repository; no external data involved.",
        processing_notes=[
            SYNTHETIC_DISCLAIMER,
            f"Deterministic: base seed {seed}, site seed {site_seed}.",
            (
                f"{len(events)} fault events injected as labelled ground truth."
                if events
                else "No faults injected."
            ),
        ],
        known_limitations=[
            "Not measured data. Do not use to characterise any real installation.",
            "Sensor noise, soiling ramps, row-to-row shading and spectral effects are not modelled.",
            "Shares a clear-sky model with the physics baseline, so physics-model accuracy "
            "on this data is near-tautological and is not reported as evidence.",
        ],
    )
    return frame, provenance


def _demo_event() -> FaultEvent:
    """The scripted demonstration underperformance window."""
    day = pd.Timestamp(DEMO_DATE)
    return FaultEvent(
        fault_type=FaultType.PARTIAL_OUTAGE,
        start=day + pd.Timedelta(hours=DEMO_START_HOUR),
        end=day + pd.Timedelta(hours=DEMO_END_HOUR),
        severity=DEMO_SEVERITY,
        description=(
            "Scripted demonstration event: a partial outage on a clear summer "
            "morning, used for the guided walkthrough. Not a real fault."
        ),
    )


def _clear_sky_poa(index: pd.DatetimeIndex, site: Site) -> tuple[np.ndarray, np.ndarray]:
    """Clear-sky GHI and plane-of-array irradiance for a site's real geometry."""
    import pvlib

    tz = f"Etc/GMT{-UTC_OFFSET_HOURS:+d}"
    location = pvlib.location.Location(
        site.latitude, site.longitude, tz=tz, altitude=site.elevation_m or 0.0
    )
    localized = index.tz_localize(tz)
    clearsky = location.get_clearsky(localized, model="ineichen")
    solar_position = location.get_solarposition(localized)

    tilt = site.tilt_deg if site.tilt_deg is not None else 25.0
    azimuth = site.azimuth_deg if site.azimuth_deg is not None else 180.0

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        model="isotropic",
    )
    return clearsky["ghi"].to_numpy(), poa["poa_global"].fillna(0.0).to_numpy()


def _autocorrelated_cloud_factor(n: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) cloud transmittance in [0, 1].

    Real cloudiness is strongly autocorrelated: a cloudy interval is very likely
    followed by another. Independent per-sample noise would produce a shimmer
    that no forecasting model could ever learn, making the task artificially
    hard and the residual distribution unrealistic.
    """
    phi = 0.985  # persistence across 15-minute steps
    noise = rng.normal(0, 1, n)
    latent = np.empty(n)
    latent[0] = noise[0]
    for i in range(1, n):
        latent[i] = phi * latent[i - 1] + np.sqrt(1 - phi**2) * noise[i]

    # Map the latent Gaussian onto a transmittance distribution skewed toward
    # clear conditions, which matches Mid-Atlantic climatology better than a
    # symmetric one.
    from scipy.stats import norm

    uniform = norm.cdf(latent)
    return np.clip(0.15 + 0.85 * uniform**0.55, 0.05, 1.0)


def _stable_site_offset(site_id: str) -> int:
    """A per-site seed offset that is identical across processes and platforms."""
    digest = hashlib.blake2b(site_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % 10_000


def _performance_drift(n: int, rng: np.random.Generator) -> np.ndarray:
    """A multiplicative performance factor that drifts over days, not intervals.

    Modelled as a strongly-persistent AR(1) around 1.0. The persistence is what
    matters: a deviation that lasts days is what makes recent output predictive
    of current output, and therefore what makes a lag-aware model capable of
    absorbing a real fault as "normal".

    Bounded well below 1.0 at the low end so ordinary drift is never mistaken
    for the injected faults, which are far deeper.

    Persistence is calibrated against the measured NIST arrays rather than
    guessed. Their daily weather-normalised performance ratio has a lag-1-day
    autocorrelation near 0.6, decaying to under 0.15 by a week — a correlation
    time of two to three days, not weeks. Getting this wrong is not cosmetic: an
    over-persistent drift makes calibration and evaluation residuals
    systematically non-exchangeable, which silently destroys the conformal
    coverage guarantee (observed dropping to ~0.35 against a 0.95 target before
    this was calibrated).
    """
    # phi**96 == 0.6 for 96 fifteen-minute steps per day, matching the measured
    # lag-1-day autocorrelation of the real arrays.
    phi = 0.6 ** (1 / 96)
    noise = rng.normal(0, 1, n)
    latent = np.empty(n)
    latent[0] = noise[0]
    for i in range(1, n):
        latent[i] = phi * latent[i - 1] + np.sqrt(1 - phi**2) * noise[i]
    return np.clip(1.0 + 0.02 * latent, 0.90, 1.06)


def _freq_minutes(freq: str) -> int:
    return int(pd.Timedelta(freq).total_seconds() // 60)
