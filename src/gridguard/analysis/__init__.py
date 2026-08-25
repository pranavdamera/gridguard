"""Offline research diagnostics.

Analyses that belong in a report or a notebook rather than in the operational
API. Everything here is a pure function over a DataFrame: it computes numbers
and returns them. Plotting lives in ``experiments/``, so the figures can change
without the analysis changing, and so the numbers behind any published figure
can be tested.
"""

from gridguard.analysis.diagnostics import (
    clear_sky_projection,
    feature_importance,
    power_curve,
    power_curve_summary,
)

__all__ = [
    "clear_sky_projection",
    "feature_importance",
    "power_curve",
    "power_curve_summary",
]
