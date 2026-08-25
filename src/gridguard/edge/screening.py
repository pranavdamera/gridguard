"""
Local screening: what a disconnected node can still detect.

This exists because of decision D4 in the architecture notes. Detection proper
runs in the core, where neighbour context and the calibrated conformal bound
live. Neither is available on an isolated node — a neighbour is by definition
somewhere else, and shipping a calibration artifact to every agent would make
the model a deployment dependency of the fleet.

But one expectation needs neither: clear-sky irradiance is a function of time
and position, computable from pvlib on the node itself with no data from
anywhere. An array producing far below what clear-sky physics allows is worth
noticing even when nobody can be told about it yet.

This is not a second detector
-----------------------------
It answers a strictly weaker question — "is this grossly below what the sun
could possibly support" — and it will miss everything subtle, because a
clear-sky ceiling is an upper bound and real weather sits far below it most of
the time. Its threshold is a fixed fraction, not a calibrated bound, so it
carries no false-alarm guarantee and its output is never presented as a
detection.

What it is for is the experiment: **how much detection capability survives
isolation?** Reporting that honestly requires a screen that actually runs on an
isolated node, so it is built, and its weakness is measured rather than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from gridguard.sites.registry import Site

logger = logging.getLogger(__name__)

#: Fraction of the clear-sky ceiling below which output is called suspicious.
#:
#: 0.15 is deliberately low. A clear-sky ceiling is an *upper* bound: overcast
#: days legitimately produce 10–20% of it, so anything higher would fire
#: constantly on ordinary weather. At 0.15 the screen effectively catches gross
#: loss — a dead inverter in bright sun — and little else. That is the honest
#: reach of a check with no weather input, and the number is a threshold, not a
#: calibrated bound.
DEFAULT_SCREEN_FRACTION = 0.15

#: Ceiling below which there is nothing to screen.
MIN_CEILING_KW = 1.0


@dataclass(frozen=True)
class ScreenResult:
    """One interval's local verdict."""

    event_time: datetime
    asset_id: str
    observed_kw: float
    clear_sky_ceiling_kw: float
    ratio: float
    suspicious: bool
    reason: str

    @property
    def explanation(self) -> str:
        return (
            f"{self.asset_id} at {self.event_time.isoformat()}: {self.observed_kw:.1f} kW "
            f"against a clear-sky ceiling of {self.clear_sky_ceiling_kw:.1f} kW "
            f"({self.ratio:.0%}). {self.reason}"
        )


class ClearSkyScreen:
    """A detector-of-last-resort for a node that cannot reach the core.

    Uses the same pvlib clear-sky model as the simulator and the physics
    baseline, so an isolated node's notion of "what the sun could support" is
    the same one the core would use. One solar model in the project, not two.
    """

    def __init__(
        self,
        site: Site,
        *,
        capacity_kw: float | None = None,
        fraction: float = DEFAULT_SCREEN_FRACTION,
        system_derate: float = 0.85,
    ) -> None:
        self.site = site
        self.capacity_kw = capacity_kw if capacity_kw is not None else site.capacity_kw
        self.fraction = fraction
        self.system_derate = system_derate

    def ceiling_kw(self, event_times: pd.DatetimeIndex) -> np.ndarray:
        """Clear-sky AC ceiling for each instant, in kW.

        No temperature derate is applied. The node may have no working
        thermometer — that is one of the failures it might be screening for —
        and omitting the derate makes the ceiling slightly generous, which is
        the safe direction for an upper bound used to flag gross shortfalls.
        """
        from gridguard.data.synthetic import UTC_OFFSET_HOURS, clear_sky_poa

        local = pd.DatetimeIndex(
            pd.to_datetime(event_times, utc=True).tz_localize(None)
            + pd.Timedelta(hours=UTC_OFFSET_HOURS)
        )
        _, poa = clear_sky_poa(local, self.site)
        return self.capacity_kw * (poa / 1000.0) * self.system_derate

    def screen(
        self,
        event_times: pd.DatetimeIndex,
        observed_kw: np.ndarray,
        *,
        asset_id: str,
    ) -> list[ScreenResult]:
        """Compare observed output against the local clear-sky ceiling."""
        ceiling = self.ceiling_kw(event_times)
        results: list[ScreenResult] = []

        for position, event_time in enumerate(event_times):
            limit = float(ceiling[position])
            observed = float(observed_kw[position])

            if limit < MIN_CEILING_KW:
                # Night, or near enough. Nothing to conclude.
                continue

            ratio = observed / limit
            suspicious = ratio < self.fraction
            results.append(
                ScreenResult(
                    event_time=event_time.to_pydatetime(),
                    asset_id=asset_id,
                    observed_kw=observed,
                    clear_sky_ceiling_kw=limit,
                    ratio=ratio,
                    suspicious=suspicious,
                    reason=(
                        f"Below {self.fraction:.0%} of the clear-sky ceiling. Local screen "
                        "only — no neighbour context and no calibrated bound, so this is "
                        "a flag, not a detection."
                        if suspicious
                        else "Within the range clear-sky physics allows."
                    ),
                )
            )

        return results

    def suspicious_fraction(
        self, event_times: pd.DatetimeIndex, observed_kw: np.ndarray, *, asset_id: str
    ) -> float:
        """Share of daylight intervals the screen flags.

        The quantity the isolation experiment reports: how much a node can still
        see on its own, against how much the core sees with everything.
        """
        results = self.screen(event_times, observed_kw, asset_id=asset_id)
        if not results:
            return 0.0
        return sum(r.suspicious for r in results) / len(results)
