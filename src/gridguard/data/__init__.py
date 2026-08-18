"""
Data layer: explicit, clearly-separated data modes.

GridGuard runs in two data modes and never blurs the line between them:

  ``real``       Measured photovoltaic telemetry from a public open-energy
                 dataset (NREL PVDAQ, distributed via the OEDI data lake),
                 paired with weather measured by instruments at the same site.

  ``synthetic``  A physically-motivated simulation of a DMV campus fleet, used
                 for controlled fault-injection demonstrations. Institution
                 names are illustrative; no real institution supplied telemetry.

Every frame produced by this package carries a ``data_mode`` column, and every
dataset carries a :class:`~gridguard.data.provenance.DatasetProvenance` record
describing exactly where the numbers came from.
"""

from gridguard.data.provenance import DatasetProvenance, load_provenance, save_provenance
from gridguard.data.schema import (
    CANONICAL_COLUMNS,
    DataMode,
    resample_to_interval,
    validate_canonical,
)
from gridguard.data.sources import (
    DataSource,
    RealPVDataSource,
    SyntheticDMVSource,
    get_source,
    load_site_data,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "DataMode",
    "DataSource",
    "DatasetProvenance",
    "RealPVDataSource",
    "SyntheticDMVSource",
    "get_source",
    "load_provenance",
    "load_site_data",
    "resample_to_interval",
    "save_provenance",
    "validate_canonical",
]
