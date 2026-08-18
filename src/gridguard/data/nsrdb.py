"""
NSRDB weather enrichment (optional).

The curated GridGuard real-data pipeline does **not** use this module. The
PVDAQ systems it ships with publish irradiance, ambient temperature and wind
measured by instruments at the array itself, which is strictly better for
modelling what a given array experienced than a satellite-derived estimate for
the surrounding grid cell — and it keeps the real-data path free of credentials.

This module exists for the case GridGuard does not currently ship: extending
the fleet to a PV system that reports generation but carries no on-site weather
instrumentation. NSRDB then supplies GHI, DNI, DHI, temperature and wind for the
system's coordinates.

Credential
----------
NSRDB requires a free NREL developer API key. Set it in the environment:

    NREL_API_KEY=your-key-here      # https://developer.nrel.gov/signup/
    NREL_API_EMAIL=you@example.com  # NSRDB additionally requires an email

Nothing in this repository ships a key, and the module raises rather than
falling back to a demonstration key that would silently rate-limit.

Endpoint
--------
NSRDB PSM v3 download API:
https://developer.nrel.gov/docs/solar/nsrdb/psm3-2-2-download/
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from gridguard.config import settings

logger = logging.getLogger(__name__)

PSM3_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv"

#: NSRDB attribute names mapped onto GridGuard's canonical weather columns.
_ATTRIBUTE_MAP = {
    "GHI": "ghi_wm2",
    "DNI": "dni_wm2",
    "DHI": "dhi_wm2",
    "Temperature": "temperature_c",
    "Wind Speed": "wind_speed_ms",
}

REQUESTED_ATTRIBUTES = "ghi,dni,dhi,air_temperature,wind_speed"


class NSRDBError(RuntimeError):
    """Raised when NSRDB cannot satisfy a request."""


class NSRDBCredentialsMissing(NSRDBError):
    """Raised when the NREL API key or email is not configured."""


class NSRDBWeatherSource:
    """Fetch historical irradiance and weather for a coordinate from NSRDB."""

    def __init__(self, api_key: str | None = None, email: str | None = None) -> None:
        self.api_key = api_key or settings.nrel_api_key
        self.email = email or settings.nrel_api_email

    def _require_credentials(self) -> None:
        if not self.api_key or self.api_key in ("", "DEMO_KEY", "your-key-here"):
            raise NSRDBCredentialsMissing(
                "NSRDB requires a real NREL API key. Set NREL_API_KEY in your environment "
                "(free at https://developer.nrel.gov/signup/). GridGuard's shipped real-data "
                "pipeline does not need this — it uses weather measured at the array."
            )
        if not self.email or "@" not in self.email:
            raise NSRDBCredentialsMissing(
                "NSRDB requires a contact email. Set NREL_API_EMAIL in your environment."
            )

    def fetch_year(
        self,
        latitude: float,
        longitude: float,
        year: int,
        *,
        interval_minutes: int = 30,
    ) -> pd.DataFrame:
        """Fetch one calendar year of weather for a coordinate.

        Returns a frame with ``timestamp`` in local standard time plus
        ``ghi_wm2``, ``dni_wm2``, ``dhi_wm2``, ``temperature_c`` and
        ``wind_speed_ms``.
        """
        self._require_credentials()

        params = {
            "api_key": self.api_key,
            "email": self.email,
            "wkt": f"POINT({longitude:.4f} {latitude:.4f})",
            "names": str(year),
            "interval": str(interval_minutes),
            "attributes": REQUESTED_ATTRIBUTES,
            "utc": "false",  # local standard time, matching PVDAQ's convention
            "leap_day": "true",
        }
        logger.info("Requesting NSRDB %s for (%.4f, %.4f) …", year, latitude, longitude)
        resp = requests.get(PSM3_URL, params=params, timeout=120)
        if resp.status_code == 403:
            raise NSRDBCredentialsMissing(
                "NSRDB rejected the credentials (HTTP 403). Check NREL_API_KEY and NREL_API_EMAIL."
            )
        if resp.status_code == 429:
            raise NSRDBError("NSRDB rate limit reached (HTTP 429). Retry later.")
        resp.raise_for_status()

        return self._parse_psm3(resp.text)

    @staticmethod
    def _parse_psm3(text: str) -> pd.DataFrame:
        """Parse a PSM3 CSV: two metadata rows, then a header, then the data."""
        raw = pd.read_csv(io.StringIO(text), skiprows=2)

        missing = [c for c in _ATTRIBUTE_MAP if c not in raw.columns]
        if missing:
            raise NSRDBError(f"NSRDB response is missing expected columns: {missing}")

        frame = raw.rename(columns=_ATTRIBUTE_MAP)
        frame["timestamp"] = pd.to_datetime(
            dict(
                year=raw["Year"],
                month=raw["Month"],
                day=raw["Day"],
                hour=raw["Hour"],
                minute=raw["Minute"],
            )
        )
        keep = ["timestamp", *_ATTRIBUTE_MAP.values()]
        return frame[keep].sort_values("timestamp").reset_index(drop=True)
