"""
Layer 1 — environment.

What the sky and the air are actually doing, at a site, at each tick. No
instruments and no equipment: this layer would produce the same numbers if
nobody were measuring.

Physics is reused, not rebuilt. Clear-sky plane-of-array irradiance comes from
:func:`gridguard.data.synthetic.clear_sky_poa` (pvlib Ineichen plus isotropic
transposition), and cloud attenuation from
:func:`gridguard.data.synthetic.autocorrelated_cloud_factor`. Those are the same
functions the shipped generator and the physics baseline rest on, which keeps
one solar model in the project rather than two that can drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridguard.data.synthetic import autocorrelated_cloud_factor, clear_sky_poa
from gridguard.simulation.clock import LogicalClock
from gridguard.simulation.config import SimulationConfig, stream_for
from gridguard.sites.registry import Site


@dataclass(frozen=True)
class EnvironmentTruth:
    """Ground-truth weather over a whole run, for one site.

    These are the real values. Whatever the pyranometer later reports, *this*
    is what the sun was doing — which is what makes it possible to score a
    sensor fault as a sensor fault rather than as underperformance.
    """

    site_id: str
    event_time: pd.DatetimeIndex

    #: Clear-sky plane-of-array irradiance, before cloud (W/m²).
    clear_sky_poa_wm2: np.ndarray
    #: Cloud transmittance in [0, 1].
    cloud_transmittance: np.ndarray
    #: Actual plane-of-array irradiance reaching the modules (W/m²).
    poa_irradiance_wm2: np.ndarray
    #: True ambient air temperature (°C).
    ambient_temperature_c: np.ndarray
    #: True wind speed (m/s).
    wind_speed_ms: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_time": self.event_time,
                "site_id": self.site_id,
                "clear_sky_poa_wm2": self.clear_sky_poa_wm2,
                "cloud_transmittance": self.cloud_transmittance,
                "poa_irradiance_wm2": self.poa_irradiance_wm2,
                "ambient_temperature_c": self.ambient_temperature_c,
                "wind_speed_ms": self.wind_speed_ms,
            }
        )


class EnvironmentLayer:
    """Generates ground-truth weather for one site."""

    def __init__(self, site: Site, config: SimulationConfig) -> None:
        self.site = site
        self.config = config

    def run(self, clock: LogicalClock) -> EnvironmentTruth:
        event_time = clock.index()

        # clear_sky_poa works in the site's local standard time, which is what
        # keeps the solar day continuous across daylight-saving boundaries. The
        # simulated clock is UTC, so it is converted here rather than the solar
        # model being taught about time zones.
        from gridguard.data.synthetic import UTC_OFFSET_HOURS

        local = pd.DatetimeIndex(
            event_time.tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=UTC_OFFSET_HOURS)
        )
        ghi, poa = clear_sky_poa(local, self.site)

        n = len(event_time)
        cloud = autocorrelated_cloud_factor(
            n, stream_for(self.config.seed, self.site.site_id, "environment", "cloud")
        )

        irradiance = np.clip(poa * cloud, 0.0, 1400.0)
        # Plane-of-array irradiance is meaningless when the sun is down; the
        # transposition can return small positive values from diffuse terms at
        # very low sun angles.
        irradiance[ghi <= 1.0] = 0.0

        day_of_year = local.day_of_year.to_numpy()
        hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
        seasonal = 15.0 + 11.0 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 105)))
        diurnal = 5.0 * np.sin(np.radians(15.0 * (hour - 15)))
        temperature = (
            seasonal
            + diurnal
            + stream_for(self.config.seed, self.site.site_id, "environment", "temperature").normal(
                0.0, 1.4, n
            )
        )

        wind = np.abs(
            stream_for(self.config.seed, self.site.site_id, "environment", "wind").normal(
                3.2, 1.4, n
            )
        )

        return EnvironmentTruth(
            site_id=self.site.site_id,
            event_time=event_time,
            clear_sky_poa_wm2=poa,
            cloud_transmittance=cloud,
            poa_irradiance_wm2=irradiance,
            ambient_temperature_c=temperature,
            wind_speed_ms=wind,
        )
