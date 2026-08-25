"""
Telemetry: what a measurement is, once it has to travel.

The time model
--------------
A batch analytics system needs one timestamp. A distributed one needs two, and
conflating them is how "the site is down" gets confused with "we have not heard
from the site".

``event_time``   When the measurement was taken, in **UTC**, timezone-aware.
                 This is the authoritative instant. It is what orders samples,
                 what joins them to weather, and what a conformal bound is
                 indexed by.

``ingest_time``  When the measurement reached the collector, in UTC. Not a
                 property of the physical world — a property of the network.

The difference between them is *arrival lag*, and it is the quantity this whole
project is being rebuilt to reason about. It cannot be inferred after the fact,
which is why both are recorded at the point of capture rather than derived
later.

Relationship to the existing ``timestamp`` column
-------------------------------------------------
The shipped schema uses a timezone-naive local-standard-time ``timestamp``, and
every committed parquet file, every trained model and every artifact manifest
depends on it. That column is **not** removed. It stays as the modelling and
presentation column, and ``event_time`` is introduced alongside it as the
authoritative UTC instant, derived through :func:`local_to_event_time` using the
offset recorded in the dataset's provenance.

This is deliberate sequencing, not fence-sitting. Making UTC authoritative in
the domain layer now is what lets the edge and collector reason about lag when
they arrive; rewriting every feature, model and endpoint to a new time column in
the same change would break the forecasting and calibration this migration
exists to preserve. The two representations are kept consistent by round-trip
tests rather than by convention.

Solar position is computed from local standard time throughout the project
because that keeps the diurnal cycle continuous across daylight-saving
boundaries. That reasoning is unaffected: local standard time remains derivable
from ``event_time`` and the site's fixed offset at any point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntFlag

import pandas as pd

#: Canonical UTC column names introduced by the domain model.
EVENT_TIME_COLUMN = "event_time"
INGEST_TIME_COLUMN = "ingest_time"
SEQUENCE_COLUMN = "sequence"
QUALITY_COLUMN = "quality"


class QualityFlag(IntFlag):
    """What is known to be wrong with a measurement.

    A flag set, not an enum: a sample can be both stale and out of range, and
    collapsing that to a single "bad" loses the reason.

    Flags are set by whoever first observes the problem and are **never
    silently cleared** downstream. A consumer may decide a flagged sample is
    still usable; it may not decide the flag was mistaken.
    """

    OK = 0
    #: Outside the physically possible range for the channel.
    OUT_OF_RANGE = 1
    #: Identical to the previous sample for longer than the channel plausibly
    #: holds still — the signature of a frozen sensor.
    FROZEN = 2
    #: Older than the freshness budget for its channel when it was read.
    STALE = 4
    #: Reconstructed rather than measured (e.g. gap interpolation).
    INTERPOLATED = 8
    #: Expected but absent.
    MISSING = 16
    #: Arrived after a later sample from the same asset had already been seen.
    OUT_OF_ORDER = 32

    @property
    def is_measured(self) -> bool:
        """Whether the value reflects a real reading from the instrument."""
        return not (self & (QualityFlag.INTERPOLATED | QualityFlag.MISSING))


@dataclass(frozen=True)
class TelemetryRecord:
    """One measurement from one asset.

    Deliberately a single channel rather than a wide row. A wide row assumes
    every channel of a site arrives together, which is exactly the assumption
    that stops holding once sensors and links can fail independently.
    """

    asset_id: str
    site_id: str
    channel: str
    value: float | None
    unit: str

    event_time: datetime
    ingest_time: datetime | None = None
    sequence: int | None = None
    quality: QualityFlag = QualityFlag.OK
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_time", "ingest_time"):
            value = getattr(self, name)
            if value is None:
                continue
            if value.tzinfo is None:
                raise ValueError(
                    f"{name} must be timezone-aware UTC. A naive datetime here is the "
                    "bug this field exists to prevent."
                )
        if self.value is None and not (self.quality & QualityFlag.MISSING):
            raise ValueError(
                f"{self.asset_id}/{self.channel}: a record with no value must carry "
                "QualityFlag.MISSING, so absence is never mistaken for zero."
            )

    @property
    def arrival_lag_seconds(self) -> float | None:
        """How long the measurement took to reach the collector."""
        if self.ingest_time is None:
            return None
        return (self.ingest_time - self.event_time).total_seconds()


# ---------------------------------------------------------------------------
# Conversion between local standard time and UTC event time
# ---------------------------------------------------------------------------


def local_to_event_time(
    timestamps: pd.Series | pd.DatetimeIndex,
    utc_offset_hours: int,
) -> pd.Series:
    """Convert naive local-standard-time stamps to timezone-aware UTC.

    ``utc_offset_hours`` is a *fixed* offset with no daylight-saving component,
    which is what PVDAQ publishes and what the simulator generates. Passing a
    DST-observing zone here would be wrong, and there is deliberately no
    parameter for one: if a site ever needs it, that is a change to how the
    source records time, not a flag on this function.
    """
    series = pd.Series(pd.to_datetime(pd.Series(timestamps).to_numpy()))
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "local_to_event_time expects naive local standard time; got tz-aware "
            "values. Use event_time_to_local to go the other way."
        )
    return (series - pd.Timedelta(hours=utc_offset_hours)).dt.tz_localize(UTC)


def event_time_to_local(
    event_times: pd.Series | pd.DatetimeIndex,
    utc_offset_hours: int,
) -> pd.Series:
    """Convert timezone-aware UTC back to naive local standard time.

    The exact inverse of :func:`local_to_event_time`. Asserted as such in tests,
    because a shim that is not an exact round trip silently rewrites history.
    """
    series = pd.Series(pd.to_datetime(pd.Series(event_times).to_numpy(), utc=True))
    return (series.dt.tz_convert(UTC).dt.tz_localize(None)) + pd.Timedelta(hours=utc_offset_hours)
