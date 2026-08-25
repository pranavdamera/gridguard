"""
Canonical telemetry schema shared by every data source.

Any source — measured PVDAQ telemetry or the synthetic DMV simulator — must
emit a frame with these columns so that feature engineering, modelling, and
anomaly detection are entirely source-agnostic.

Canonical columns
-----------------
``timestamp``       datetime64[ns], **local standard time, timezone-naive**.
                    Local standard time (no daylight-saving shifts) is the
                    convention used by PVDAQ and keeps solar-position maths
                    continuous across the DST boundary. The IANA-style fixed
                    offset is recorded in the dataset provenance.
``site_id``         str, key into :mod:`gridguard.sites.registry`.
``ac_power_kw``     float, AC power in kilowatts.
``irradiance_wm2``  float, plane-of-array irradiance in W/m² where available,
                    otherwise global horizontal. Which one was used is recorded
                    in provenance (``irradiance_kind``).
``temperature_c``   float, ambient air temperature in °C.
``wind_speed_ms``   float, wind speed in m/s.
``data_mode``       str, ``"real"`` or ``"synthetic"``.

Optional columns
----------------
``is_injected_fault``  bool. Present only for synthetic data, where ground
                       truth is known by construction. Measured telemetry has
                       no trustworthy fault labels and therefore never carries
                       this column — see ``docs/methodology.md``.

Distributed columns
-------------------
Added by :func:`upgrade_to_domain_schema`, and optional throughout: every frame
committed before they existed still loads, and every model trained before they
existed still scores.

``event_time``   datetime64[ns, UTC], **timezone-aware**. The authoritative
                 instant a measurement was taken. Derived from ``timestamp``
                 and the fixed offset in the dataset's provenance.
``ingest_time``  datetime64[ns, UTC]. When the measurement reached the
                 collector. Equal to ``event_time`` for data that never
                 travelled — a batch file has no arrival lag.
``asset_id``     str, key into the asset registry.
``quality``      int, a :class:`~gridguard.domain.telemetry.QualityFlag` set.
``sequence``     int, per-asset monotonic counter, for de-duplicating replays.

Why ``timestamp`` stays
-----------------------
``event_time`` is authoritative, but ``timestamp`` is not removed. Solar
position is computed from local standard time throughout the project — that is
what keeps the diurnal cycle continuous across daylight-saving boundaries — and
every committed parquet, trained model and artifact manifest is indexed by it.
The two are exact inverses of each other given the site's fixed offset, which is
asserted in tests rather than assumed.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataMode = Literal["real", "synthetic"]

#: Columns every source must produce, in canonical order.
CANONICAL_COLUMNS: list[str] = [
    "timestamp",
    "site_id",
    "ac_power_kw",
    "irradiance_wm2",
    "temperature_c",
    "wind_speed_ms",
    "data_mode",
]

#: Columns that may additionally be present.
OPTIONAL_COLUMNS: list[str] = ["is_injected_fault"]

#: Columns the domain model adds. Optional: absence means the frame predates
#: them, not that it is invalid.
DOMAIN_COLUMNS: list[str] = [
    "event_time",
    "ingest_time",
    "asset_id",
    "quality",
    "sequence",
]

#: Numeric columns that get averaged when resampling to a coarser interval.
_MEAN_COLUMNS: list[str] = [
    "ac_power_kw",
    "irradiance_wm2",
    "temperature_c",
    "wind_speed_ms",
]

#: The interval GridGuard models at, in minutes.
DEFAULT_INTERVAL_MINUTES = 15


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the canonical schema."""


def validate_canonical(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Validate and normalise a frame against the canonical schema.

    Args:
        df:     Frame to validate.
        strict: When True (default), raise on any violation. When False, log a
                warning and continue where the problem is recoverable.

    Returns:
        The frame with canonical columns ordered first and dtypes normalised.

    Raises:
        SchemaError: If a required column is missing, or (in strict mode) if a
            column has an unusable dtype or the timestamps are not unique and
            monotonic.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Frame is missing required canonical columns: {missing}. "
            f"Present: {sorted(df.columns)}"
        )

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    if isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype):
        raise SchemaError(
            "timestamp must be timezone-naive local standard time. "
            "Convert with .dt.tz_localize(None) after shifting to local standard time, "
            "and record the offset in the dataset provenance."
        )

    for col in _MEAN_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bad_modes = set(out["data_mode"].unique()) - {"real", "synthetic"}
    if bad_modes:
        raise SchemaError(f"data_mode must be 'real' or 'synthetic'; found {sorted(bad_modes)}")

    out = out.sort_values("timestamp").reset_index(drop=True)

    dupes = int(out.duplicated(subset=["site_id", "timestamp"]).sum())
    if dupes:
        message = f"{dupes} duplicate (site_id, timestamp) rows"
        if strict:
            raise SchemaError(message)
        logger.warning("%s — keeping the first of each.", message)
        out = out.drop_duplicates(subset=["site_id", "timestamp"], keep="first").reset_index(
            drop=True
        )

    ordered = CANONICAL_COLUMNS + [c for c in out.columns if c not in CANONICAL_COLUMNS]
    return out[ordered]


def resample_to_interval(
    df: pd.DataFrame,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    *,
    max_gap_intervals: int = 2,
) -> pd.DataFrame:
    """Resample a single site's canonical frame to a fixed interval.

    Measured telemetry arrives at 1-minute resolution with occasional sensor
    dropouts. Averaging to the model interval both matches the synthetic
    generator and suppresses single-sample sensor spikes.

    Short gaps (up to ``max_gap_intervals``) are linearly interpolated, which is
    appropriate for smoothly-varying weather channels. Longer gaps are left as
    NaN so that downstream code can drop them rather than silently inventing
    data — a communication outage is a real operating condition, not a value to
    be imputed.

    Args:
        df:                Canonical frame for exactly one site.
        interval_minutes:  Target interval.
        max_gap_intervals: Maximum consecutive missing intervals to interpolate.

    Returns:
        Resampled canonical frame.
    """
    if df["site_id"].nunique() > 1:
        raise ValueError("resample_to_interval expects a single site; group by site_id first.")

    freq = f"{interval_minutes}min"
    indexed = df.set_index("timestamp").sort_index()

    agg = {col: "mean" for col in _MEAN_COLUMNS if col in indexed.columns}
    resampled = indexed.resample(freq).agg(agg)

    # Ground-truth fault labels are an OR over the window: if any sample inside
    # the interval was faulted, the interval is faulted.
    if "is_injected_fault" in indexed.columns:
        resampled["is_injected_fault"] = (
            indexed["is_injected_fault"].resample(freq).max().fillna(False).astype(bool)
        )

    resampled = resampled.interpolate(method="linear", limit=max_gap_intervals, limit_area="inside")

    resampled["site_id"] = df["site_id"].iloc[0]
    resampled["data_mode"] = df["data_mode"].iloc[0]
    resampled = resampled.reset_index()

    n_missing = int(resampled["ac_power_kw"].isna().sum())
    if n_missing:
        logger.info(
            "%s: %d/%d intervals remain missing after interpolation (gaps longer than %d intervals)",
            resampled["site_id"].iloc[0],
            n_missing,
            len(resampled),
            max_gap_intervals,
        )

    return validate_canonical(resampled, strict=False)


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be modelled: missing power or missing irradiance.

    Kept separate from :func:`resample_to_interval` so that callers can inspect
    the gap structure (for the data-quality report) before discarding anything.
    """
    before = len(df)
    out = df.dropna(subset=["ac_power_kw", "irradiance_wm2"]).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %d/%d rows with missing power or irradiance.", dropped, before)
    return out


def clip_physical_ranges(df: pd.DataFrame, *, capacity_kw: float | None = None) -> pd.DataFrame:
    """Clamp measured channels to physically meaningful ranges.

    Pyranometers read slightly negative at night (thermal offset) and power
    meters can report small negative values from inverter tare draw. Both are
    instrument artefacts rather than real generation, so they are clamped to
    zero. Values are *not* clipped at the top of the range: real clipping and
    over-irradiance events are genuine signals the model should see.
    """
    out = df.copy()
    out["irradiance_wm2"] = out["irradiance_wm2"].clip(lower=0)
    out["ac_power_kw"] = out["ac_power_kw"].clip(lower=0)
    if capacity_kw is not None:
        # Guard against decimal-point errors in source data, but leave headroom
        # for legitimate over-nameplate DC-side excursions.
        implausible = out["ac_power_kw"] > 2.0 * capacity_kw
        if implausible.any():
            logger.warning(
                "%d rows exceed 2x nameplate (%.1f kW) and were set to NaN as implausible.",
                int(implausible.sum()),
                capacity_kw,
            )
            out.loc[implausible, "ac_power_kw"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Domain schema upgrade
# ---------------------------------------------------------------------------


def upgrade_to_domain_schema(
    df: pd.DataFrame,
    *,
    utc_offset_hours: int,
    asset_id: str | None = None,
    ingest_time: pd.Series | None = None,
) -> pd.DataFrame:
    """Add the distributed-system columns to a legacy canonical frame.

    Every committed parquet file in this repository predates the domain model.
    Rather than rewriting them — which would invalidate the artifact manifests
    that record exactly what was built from what — the columns are synthesised
    on load from information the frame already carries.

    Defaults, and why each is the honest one:

    ``event_time``   ``timestamp`` shifted by the site's fixed UTC offset. This
                     is a lossless representation change, not an assumption.
    ``ingest_time``  Equal to ``event_time``. A batch file did not travel, so
                     its arrival lag is genuinely zero — reporting anything else
                     would invent a network that was not there.
    ``asset_id``     The site's primary array. Legacy rows are whole-site
                     measurements with no asset dimension; attributing them to
                     the largest array is the least misleading available choice
                     and is recorded as such.
    ``quality``      ``QualityFlag.OK``. These rows already passed
                     :func:`validate_canonical` and had unusable rows dropped.
    ``sequence``     Row order within the frame. Monotonic per asset, which is
                     all a de-duplicator needs of it.

    Idempotent: columns already present are left alone, so calling this twice
    does not overwrite real ingest times with synthesised ones.
    """
    from gridguard.domain.telemetry import QualityFlag, local_to_event_time

    out = df.copy()

    if "event_time" not in out.columns:
        out["event_time"] = local_to_event_time(out["timestamp"], utc_offset_hours)

    if "ingest_time" not in out.columns:
        out["ingest_time"] = (
            pd.to_datetime(ingest_time, utc=True) if ingest_time is not None else out["event_time"]
        )

    if "asset_id" not in out.columns:
        if asset_id is None:
            from gridguard.sites.registry import primary_asset_id

            site_ids = out["site_id"].unique()
            if len(site_ids) != 1:
                raise SchemaError(
                    "upgrade_to_domain_schema needs an explicit asset_id for a frame "
                    f"spanning {len(site_ids)} sites."
                )
            asset_id = primary_asset_id(str(site_ids[0]))
        out["asset_id"] = asset_id

    if "quality" not in out.columns:
        out["quality"] = int(QualityFlag.OK)

    if "sequence" not in out.columns:
        out["sequence"] = range(len(out))

    ordered = (
        CANONICAL_COLUMNS
        + [c for c in DOMAIN_COLUMNS if c in out.columns]
        + [c for c in out.columns if c not in CANONICAL_COLUMNS + DOMAIN_COLUMNS]
    )
    return out[ordered]


def has_domain_columns(df: pd.DataFrame) -> bool:
    """Whether a frame already carries the distributed-system columns."""
    return all(c in df.columns for c in DOMAIN_COLUMNS)
