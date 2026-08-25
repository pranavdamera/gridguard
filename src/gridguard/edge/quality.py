"""
Local quality checks.

What an agent can say about a reading using only that reading, its own recent
history, and the channel's physical limits. No neighbours, no model, no
calibration — so these keep working on a node that has been isolated for a week.

The checks flag; they never repair. A frozen pyranometer's readings are passed
on carrying ``FROZEN`` rather than being dropped or interpolated, because the
agent is not in a position to know whether the value is wrong: it knows only
that the channel has stopped moving. Deciding what to do about that needs
context the agent does not have. Flag locally, decide centrally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gridguard.domain.telemetry import QualityFlag, TelemetryRecord

logger = logging.getLogger(__name__)

#: Physically possible range per channel. Deliberately generous — the job is to
#: catch a decimal-point error or a dead ADC returning a rail value, not to
#: second-guess an unusual but possible reading.
CHANNEL_RANGES: dict[str, tuple[float, float]] = {
    "ac_power_kw": (-10.0, 100_000.0),
    "irradiance_wm2": (-20.0, 1600.0),
    "temperature_c": (-60.0, 70.0),
    "wind_speed_ms": (0.0, 120.0),
}

#: How many identical consecutive readings before a channel is called frozen.
#: Irradiance and power move continuously in daylight, so repetition is a strong
#: signal — but at night both sit at zero legitimately for hours, which is why
#: the frozen check has a darkness exemption below.
DEFAULT_FROZEN_THRESHOLD = 8

#: Below this irradiance, a flat power channel is night rather than a fault.
NIGHT_IRRADIANCE_WM2 = 20.0


@dataclass
class ChannelHistory:
    """The little state a frozen-channel check needs."""

    last_value: float | None = None
    repeat_count: int = 0


@dataclass
class QualityChecker:
    """Applies local checks to a stream of records, per channel.

    Stateful by necessity: "this channel has not moved in two hours" is not
    visible in a single reading. State is per ``(asset_id, channel)`` so one
    frozen sensor does not implicate another.
    """

    frozen_threshold: int = DEFAULT_FROZEN_THRESHOLD
    history: dict[tuple[str, str], ChannelHistory] = field(default_factory=dict)

    #: Most recent irradiance seen per site, so the darkness exemption can be
    #: applied to the power channel. Best-effort: if irradiance has not arrived
    #: yet the exemption simply does not apply.
    _site_irradiance: dict[str, float] = field(default_factory=dict)

    def check(self, record: TelemetryRecord) -> TelemetryRecord:
        """Return the record with locally-determined flags added.

        Flags are OR-ed onto whatever the record already carries. Nothing is
        ever cleared: a flag set upstream reflects something that upstream
        observed, and the agent is not in a position to overrule it.
        """
        quality = record.quality

        if record.value is None:
            return _with_quality(record, quality | QualityFlag.MISSING)

        if record.channel == "irradiance_wm2":
            self._site_irradiance[record.site_id] = record.value

        quality |= self._range_flag(record)
        quality |= self._frozen_flag(record)

        return _with_quality(record, quality)

    # -- individual checks --------------------------------------------------

    def _range_flag(self, record: TelemetryRecord) -> QualityFlag:
        bounds = CHANNEL_RANGES.get(record.channel)
        if bounds is None or record.value is None:
            return QualityFlag.OK
        low, high = bounds
        if not low <= record.value <= high:
            logger.warning(
                "%s/%s reading %.3f outside [%.1f, %.1f]",
                record.asset_id,
                record.channel,
                record.value,
                low,
                high,
            )
            return QualityFlag.OUT_OF_RANGE
        return QualityFlag.OK

    def _frozen_flag(self, record: TelemetryRecord) -> QualityFlag:
        key = (record.asset_id, record.channel)
        state = self.history.setdefault(key, ChannelHistory())

        if record.value is not None and state.last_value == record.value:
            state.repeat_count += 1
        else:
            state.last_value = record.value
            state.repeat_count = 0

        if state.repeat_count < self.frozen_threshold:
            return QualityFlag.OK

        # A PV array sits at exactly zero all night, and a power meter reporting
        # zero in the dark is not a stuck meter. Without this exemption every
        # night would produce a fleet-wide burst of FROZEN flags, which would
        # train an operator to ignore the flag entirely.
        if self._is_dark(record) and record.value == 0.0:
            return QualityFlag.OK

        return QualityFlag.FROZEN

    def _is_dark(self, record: TelemetryRecord) -> bool:
        irradiance = self._site_irradiance.get(record.site_id)
        return irradiance is not None and irradiance < NIGHT_IRRADIANCE_WM2

    def reset(self) -> None:
        self.history.clear()
        self._site_irradiance.clear()


def _with_quality(record: TelemetryRecord, quality: QualityFlag) -> TelemetryRecord:
    """A copy of ``record`` carrying ``quality``.

    ``TelemetryRecord`` is frozen, so flags produce a new record rather than
    mutating one someone else may hold.
    """
    if quality == record.quality:
        return record
    return TelemetryRecord(
        asset_id=record.asset_id,
        site_id=record.site_id,
        channel=record.channel,
        value=record.value,
        unit=record.unit,
        event_time=record.event_time,
        ingest_time=record.ingest_time,
        sequence=record.sequence,
        quality=quality,
        metadata=record.metadata,
    )
