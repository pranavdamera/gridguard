"""Helpers shared across test modules."""

from __future__ import annotations

import pandas as pd

from gridguard.data.synthetic import generate_site_telemetry
from gridguard.sites.registry import get_site


def make_frame(
    start: str = "2022-01-01",
    end: str = "2022-06-30",
    seed: int = 42,
    site_id: str = "gmu_fairfax",
    inject_scenario: bool = False,
) -> pd.DataFrame:
    """Canonical synthetic telemetry for tests.

    Replaces the old private ``_generate_synthetic`` helper, which moved into
    :mod:`gridguard.data.synthetic` when the data layer was split by mode.
    """
    frame, _ = generate_site_telemetry(
        get_site(site_id),
        start=start,
        end=end,
        seed=seed,
        inject_scenario=inject_scenario,
    )
    return frame
