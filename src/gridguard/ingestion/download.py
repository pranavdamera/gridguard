"""
Backwards-compatible shim over :mod:`gridguard.data`.

The ingestion logic moved into the ``gridguard.data`` package, which separates
measured telemetry (:class:`~gridguard.data.sources.RealPVDataSource`) from
simulated demonstration data
(:class:`~gridguard.data.sources.SyntheticDMVSource`) and attaches a provenance
record to both.

The old ``load_raw_data`` entry point is kept because notebooks, the Streamlit
dashboard, and older scripts call it. New code should call
:func:`gridguard.data.sources.load_site_data`, which returns provenance
alongside the frame.

The retired ``_fetch_nrel_pvdaq`` helper targeted the PVDAQ v3 REST API, which
NREL has decommissioned. Real data now comes from the OEDI data lake; see
:mod:`gridguard.data.oedi`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gridguard.data.sources import load_site_data
from gridguard.sites.registry import list_sites

logger = logging.getLogger(__name__)

__all__ = ["load_raw_data"]


def load_raw_data(
    source: str | None = None,
    output_dir: Path | None = None,
    site_id: str | None = None,
    demo: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load canonical telemetry for a site. Deprecated in favour of ``load_site_data``.

    Args:
        source:     Ignored. The data mode is a property of the site in the
                    registry, not a per-call choice, so that a site can never be
                    served with the wrong kind of data by accident.
        output_dir: Cache directory.
        site_id:    Site to load. Defaults to the first synthetic site.
        demo:       Inject the scripted demonstration event.

    Returns:
        The canonical telemetry frame (without its provenance record).
    """
    if source is not None:
        logger.warning(
            "load_raw_data(source=%r) is ignored: data mode now comes from the site "
            "registry. Call gridguard.data.sources.load_site_data instead.",
            source,
        )

    if site_id is None:
        synthetic = list_sites(data_mode="synthetic")
        if not synthetic:
            raise ValueError("No synthetic sites in the registry and no site_id given.")
        site_id = synthetic[0].site_id
        logger.info("No site_id given; defaulting to '%s'.", site_id)

    frame, _ = load_site_data(site_id, cache_dir=output_dir, demo=demo, **kwargs)
    return frame
