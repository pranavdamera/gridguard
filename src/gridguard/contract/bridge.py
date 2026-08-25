"""
Simulator output → wire records.

The join between phases 2–3 and phase 5. The simulator produces a wide frame —
one row per interval, several channels side by side — because that is what the
existing feature and model code consumes. The wire carries narrow records, one
per channel, because channels fail independently once sensors and links can
break separately.

Widening and narrowing are not symmetric, and the asymmetry is the point.
Collapsing four channels into one row asserts that all four arrived; that
assertion is exactly what stops being true in a degraded fleet. So the narrow
form is what travels, and the wide form is reconstructed at the far end from
whatever actually showed up.

Nothing here sends anything. Phase 4 fixes the message shape; phase 5 gives it a
transport.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from gridguard.domain.telemetry import QualityFlag, TelemetryRecord

logger = logging.getLogger(__name__)

#: Canonical channel → unit, and which asset kind is its source. Units travel
#: with the number because upstream channel names were not a reliable guide to
#: them — see the unit-resolution notes in ``gridguard.data.oedi``.
CHANNEL_UNITS: dict[str, str] = {
    "ac_power_kw": "kW",
    "irradiance_wm2": "W/m2",
    "temperature_c": "degC",
    "wind_speed_ms": "m/s",
}


def records_from_frame(
    frame: pd.DataFrame,
    *,
    channels: tuple[str, ...] | None = None,
    default_asset_id: str | None = None,
) -> list[TelemetryRecord]:
    """Narrow a canonical frame into per-channel wire records.

    Args:
        frame: canonical output from the simulator, carrying ``event_time``,
            ``site_id`` and the channel columns.
        channels: which channels to emit. Defaults to every known channel
            present in the frame.
        default_asset_id: attribution for rows with no ``asset_id`` column.

    Returns:
        One record per (interval, channel). A NaN reading becomes an absent
        value carrying ``MISSING`` rather than a zero, which is the whole
        reason the wire schema declares ``value`` optional.
    """
    if "event_time" not in frame.columns:
        raise ValueError("records_from_frame needs an event_time column")

    selected = channels or tuple(c for c in CHANNEL_UNITS if c in frame.columns)
    if not selected:
        raise ValueError(
            f"No known channels in frame. Columns: {sorted(frame.columns)}; "
            f"known: {sorted(CHANNEL_UNITS)}"
        )

    event_times = pd.to_datetime(frame["event_time"], utc=True)
    site_ids = frame["site_id"] if "site_id" in frame.columns else None
    asset_ids = frame["asset_id"] if "asset_id" in frame.columns else None
    sequences = frame["sequence"] if "sequence" in frame.columns else None
    qualities = frame["quality"] if "quality" in frame.columns else None

    records: list[TelemetryRecord] = []
    for position in range(len(frame)):
        event_time = event_times.iloc[position].to_pydatetime()
        site_id = str(site_ids.iloc[position]) if site_ids is not None else ""
        asset_id = (
            str(asset_ids.iloc[position])
            if asset_ids is not None
            else (default_asset_id or f"{site_id}_array_1")
        )
        sequence = int(sequences.iloc[position]) if sequences is not None else position
        base_quality = (
            QualityFlag(int(qualities.iloc[position])) if qualities is not None else QualityFlag.OK
        )

        for channel in selected:
            raw = frame[channel].iloc[position]
            missing = pd.isna(raw)
            quality = base_quality | QualityFlag.MISSING if missing else base_quality
            records.append(
                TelemetryRecord(
                    asset_id=asset_id,
                    site_id=site_id,
                    channel=channel,
                    unit=CHANNEL_UNITS[channel],
                    value=None if missing else float(raw),
                    event_time=event_time,
                    sequence=sequence,
                    quality=quality,
                )
            )

    return records


def frame_from_records(records: list[TelemetryRecord]) -> pd.DataFrame:
    """Widen wire records back into a canonical-shaped frame.

    The inverse of :func:`records_from_frame`, and deliberately tolerant: an
    interval whose power arrived but whose irradiance did not produces a row
    with power set and irradiance NaN. That is the honest reconstruction —
    dropping the row would discard a real reading, and defaulting the gap to
    zero would invent one.
    """
    if not records:
        return pd.DataFrame(columns=["event_time", "site_id", "asset_id", *CHANNEL_UNITS])

    rows: dict[tuple, dict] = {}
    for record in records:
        key = (record.event_time, record.site_id, record.asset_id)
        row = rows.setdefault(
            key,
            {
                "event_time": record.event_time,
                "site_id": record.site_id,
                "asset_id": record.asset_id,
                "sequence": record.sequence,
                "quality": int(QualityFlag.OK),
            },
        )
        row[record.channel] = record.value
        # A row is as good as its worst channel; flags accumulate, never clear.
        row["quality"] = int(QualityFlag(row["quality"]) | record.quality)

    frame = pd.DataFrame(list(rows.values()))
    for channel in CHANNEL_UNITS:
        if channel not in frame.columns:
            frame[channel] = pd.NA

    ordered = ["event_time", "site_id", "asset_id", "sequence", "quality", *CHANNEL_UNITS]
    return frame[ordered].sort_values("event_time", ignore_index=True)


def batches_by_site(
    records: list[TelemetryRecord],
    *,
    sent_at: datetime,
    agent_prefix: str = "agent",
):
    """Group records into one wire batch per site.

    An agent reports for one site's assets, so this is the shape a collector
    actually receives.
    """
    from gridguard.contract.codec import encode_batch

    by_site: dict[str, list[TelemetryRecord]] = {}
    for record in records:
        by_site.setdefault(record.site_id, []).append(record)

    return {
        site_id: encode_batch(
            site_records,
            agent_id=f"{agent_prefix}_{site_id}",
            site_id=site_id,
            sent_at=sent_at,
        )
        for site_id, site_records in by_site.items()
    }
