"""
Neighbour-aware anomaly attribution.

The question this answers
-------------------------
A single-site detector can tell you that an array produced less than expected.
It cannot tell you *why*, and the two explanations call for completely different
responses:

* **Site-specific** — this array underperformed while its neighbours, under the
  same sky, did not. Something is wrong with the equipment. Dispatch a
  technician.
* **Regional** — several nearby arrays dropped together. The weather model was
  wrong, not the hardware. Do nothing.

Distinguishing them is the difference between an alert worth acting on and a
false alarm that erodes trust in the system.

How it works
------------
For a site and time window, GridGuard compares that site's normalised residual
against its geographic neighbours' residuals over the *same* window:

* Residuals are normalised by nameplate capacity, making a 70 kW roof and a
  270 kW ground array directly comparable.
* Neighbour agreement is measured both unweighted (median, robust to one odd
  neighbour) and inverse-distance weighted (closer sites share weather more).
* The classification requires the site to be meaningfully worse than its
  neighbours, not merely negative at the same time as them.

What it does not do
-------------------
This is **not** a fault classifier. It does not diagnose inverter failure,
soiling, or shading, and it makes no claim to. It answers one narrower question
— is this deviation shared with nearby sites or not — and always reports the
evidence behind its answer so a human can disagree.

It is also only as good as fleet density. With one neighbour, or with neighbours
tens of kilometres away under a different cloud field, the verdict is
``indeterminate`` rather than a guess.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from gridguard.spatial.geo import neighbor_graph

logger = logging.getLogger(__name__)


class AnomalyScope(StrEnum):
    """Whether a deviation appears local to one site or shared regionally."""

    SITE_SPECIFIC = "site_specific"
    REGIONAL = "regional"
    INDETERMINATE = "indeterminate"


#: A site must underperform its neighbourhood by at least this much of nameplate
#: before the deviation is called site-specific. Set from the residual spread
#: typically seen between co-located arrays under identical weather.
SITE_SPECIFIC_MARGIN = 0.05

#: Normalised residual below which a neighbour counts as "also affected".
NEIGHBOR_AFFECTED_THRESHOLD = -0.04

#: Fraction of neighbours that must be affected to call a deviation regional.
REGIONAL_AGREEMENT_FRACTION = 0.5

#: Below this many neighbours, no spatial claim is made at all.
MIN_NEIGHBORS_FOR_VERDICT = 2


@dataclass
class SpatialContext:
    """Neighbourhood evidence for one site over one window."""

    site_id: str
    scope: AnomalyScope
    confidence: str  # "low" | "moderate" | "high"
    site_normalised_residual: float
    neighbor_count: int
    neighbors_considered: list[str] = field(default_factory=list)
    neighbor_residual_mean: float | None = None
    neighbor_residual_median: float | None = None
    distance_weighted_residual: float | None = None
    neighbors_affected: int = 0
    regional_anomaly_score: float = 0.0
    excess_deviation: float | None = None
    radius_km: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        for key in (
            "site_normalised_residual",
            "neighbor_residual_mean",
            "neighbor_residual_median",
            "distance_weighted_residual",
            "regional_anomaly_score",
            "excess_deviation",
        ):
            if payload[key] is not None:
                payload[key] = round(float(payload[key]), 4)
        return payload


def normalised_residual(
    frame: pd.DataFrame,
    capacity_kw: float,
) -> float:
    """Mean residual over a window as a fraction of nameplate capacity.

    Normalising by capacity is what makes sites of different sizes comparable:
    a 20 kW shortfall is catastrophic for a 70 kW roof and unremarkable for a
    500 kW field.
    """
    if frame.empty or capacity_kw <= 0:
        return 0.0
    residual = frame["ac_power_kw"] - frame["predicted_kw"]
    return float(residual.mean() / capacity_kw)


def classify_anomaly_scope(
    site_id: str,
    site_residual: float,
    neighbor_residuals: dict[str, float],
    neighbor_distances: dict[str, float],
    radius_km: float,
) -> SpatialContext:
    """Attribute a deviation to the site or to the region, with evidence.

    Args:
        site_id:            Site under investigation.
        site_residual:      Its capacity-normalised residual over the window.
        neighbor_residuals: Capacity-normalised residual per neighbour, same window.
        neighbor_distances: Distance in km per neighbour.
        radius_km:          Neighbourhood radius used.
    """
    neighbors = sorted(neighbor_residuals)
    count = len(neighbors)

    context = SpatialContext(
        site_id=site_id,
        scope=AnomalyScope.INDETERMINATE,
        confidence="low",
        site_normalised_residual=site_residual,
        neighbor_count=count,
        neighbors_considered=neighbors,
        radius_km=radius_km,
    )

    if count < MIN_NEIGHBORS_FOR_VERDICT:
        context.explanation = (
            f"Only {count} site(s) within {radius_km:.0f} km with data for this window — "
            f"too few to separate a site-specific fault from regional weather. "
            f"Deploy more sites nearby, or treat this as unattributed."
        )
        return context

    values = np.array([neighbor_residuals[n] for n in neighbors], dtype=float)
    distances = np.array([max(neighbor_distances.get(n, radius_km), 0.1) for n in neighbors])

    context.neighbor_residual_mean = float(np.mean(values))
    context.neighbor_residual_median = float(np.median(values))

    # Inverse-distance weighting: nearer sites share a cloud field more often.
    weights = 1.0 / distances
    context.distance_weighted_residual = float(np.sum(values * weights) / np.sum(weights))

    context.neighbors_affected = int(np.sum(values <= NEIGHBOR_AFFECTED_THRESHOLD))
    context.regional_anomaly_score = float(context.neighbors_affected / count)

    # How much worse is this site than its neighbourhood? The median is used as
    # the reference because it is unmoved by a single odd neighbour.
    reference = context.neighbor_residual_median
    context.excess_deviation = float(site_residual - reference)

    affected_fraction = context.regional_anomaly_score
    is_site_worse = context.excess_deviation <= -SITE_SPECIFIC_MARGIN
    is_widely_shared = affected_fraction >= REGIONAL_AGREEMENT_FRACTION

    if is_site_worse and not is_widely_shared:
        context.scope = AnomalyScope.SITE_SPECIFIC
        context.confidence = "high" if count >= 3 else "moderate"
        context.explanation = (
            f"{site_id} ran {abs(site_residual) * 100:.1f}% of nameplate below its expected "
            f"output while the median of {count} site(s) within {radius_km:.0f} km sat at "
            f"{reference * 100:+.1f}%. The {abs(context.excess_deviation) * 100:.1f} percentage-point "
            f"gap is not explained by shared weather, which points to something local to this "
            f"site. {context.neighbors_affected} of {count} neighbours were also below par."
        )
    elif is_widely_shared and site_residual <= NEIGHBOR_AFFECTED_THRESHOLD:
        context.scope = AnomalyScope.REGIONAL
        context.confidence = "high" if affected_fraction >= 0.75 and count >= 3 else "moderate"
        context.explanation = (
            f"{context.neighbors_affected} of {count} sites within {radius_km:.0f} km fell below "
            f"expected output over the same window (median {reference * 100:+.1f}% of nameplate, "
            f"this site {site_residual * 100:+.1f}%). A shared deviation of this shape is far more "
            f"consistent with the weather model being wrong than with simultaneous "
            f"equipment failures."
        )
    else:
        context.scope = AnomalyScope.INDETERMINATE
        context.confidence = "low"
        context.explanation = (
            f"Evidence is mixed: this site is at {site_residual * 100:+.1f}% of nameplate against a "
            f"neighbourhood median of {reference * 100:+.1f}%, with {context.neighbors_affected} of "
            f"{count} neighbours affected. That is neither a clean local outlier nor a clearly "
            f"shared regional event."
        )

    return context


def compute_fleet_spatial_context(
    site_frames: dict[str, pd.DataFrame],
    radius_km: float,
    *,
    sites: list | None = None,
) -> dict[str, SpatialContext]:
    """Compute spatial context for every site in a fleet snapshot.

    Args:
        site_frames: ``{site_id: frame}`` where each frame covers the *same*
            window and carries ``ac_power_kw`` and ``predicted_kw``.
        radius_km:   Neighbourhood radius.
        sites:       Site records; loaded from the registry when omitted.

    Returns:
        ``{site_id: SpatialContext}`` for the sites present in ``site_frames``.
    """
    from gridguard.sites.registry import list_sites

    all_sites = sites if sites is not None else list_sites()
    by_id = {s.site_id: s for s in all_sites}

    known = [by_id[sid] for sid in site_frames if sid in by_id]
    if not known:
        return {}

    graph = neighbor_graph(known, radius_km)

    residuals = {
        site_id: normalised_residual(frame, by_id[site_id].capacity_kw)
        for site_id, frame in site_frames.items()
        if site_id in by_id and not frame.empty
    }

    contexts: dict[str, SpatialContext] = {}
    for site_id in residuals:
        # Neighbours must share the site's data mode. A measured site's
        # residuals say nothing about a simulated one's: the simulator draws
        # each site's weather independently, so a synthetic "neighbour" carries
        # no information about the real cloud field over a measured array.
        # Mixing them would manufacture agreement or disagreement out of noise.
        mode = by_id[site_id].data_mode
        neighbors = [
            (n, d)
            for n, d in graph.get(site_id, [])
            if n in residuals and by_id[n].data_mode == mode
        ]
        neighbor_residuals = {n: residuals[n] for n, _ in neighbors}
        neighbor_distances = dict(neighbors)
        contexts[site_id] = classify_anomaly_scope(
            site_id=site_id,
            site_residual=residuals[site_id],
            neighbor_residuals=neighbor_residuals,
            neighbor_distances=neighbor_distances,
            radius_km=radius_km,
        )
    return contexts
