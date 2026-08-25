"""
Layer 4 — transport.

What actually reaches the collector. In phase 2 this is a pass-through:
everything arrives, in order, with zero delay. The layer exists anyway, and that
is deliberate.

Why build a layer that currently does nothing
---------------------------------------------
Because the alternative is worse. If observations flowed straight into the
pipeline now, then every consumer written between here and phase 6 would quietly
assume that what was measured is what arrived. Introducing loss later would mean
finding and fixing all of those assumptions at once, in the change that is
already the riskiest in the migration.

With the seam in place from the start, a consumer is written against *delivered*
telemetry from day one. Delivery gaining real loss and delay later changes the
contents of the frame, not its shape, and not anyone's code.

The distinction it preserves
----------------------------
Three different things can make a value absent from the collector's view, and
they are not interchangeable:

* the plant produced nothing            → equipment, layer 2
* the instrument failed to report       → sensing, layer 3
* the reading never made it across      → transport, this layer

An architecture that cannot express the third will attribute it to one of the
first two. That misattribution is precisely the failure mode this project exists
to study, so the layer that owns it is not optional even while it is empty.

Delivery adds ``ingest_time`` and ``sequence``. With a pass-through,
``ingest_time == event_time``, which reports an arrival lag of exactly zero —
true, because nothing travelled. That is the honest value, not a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridguard.domain.telemetry import QualityFlag
from gridguard.simulation.config import SimulationConfig
from gridguard.simulation.faults import FaultSchedule, delivery_mask
from gridguard.simulation.sensing import Observation


@dataclass(frozen=True)
class Delivery:
    """What reached the collector, and when.

    ``delivered`` marks whether each record arrived at all. In a pass-through
    every entry is True; the column is present so that a consumer never has to
    infer arrival from a missing row.
    """

    site_id: str
    event_time: pd.DatetimeIndex
    ingest_time: pd.DatetimeIndex
    sequence: np.ndarray
    delivered: np.ndarray
    quality: np.ndarray

    #: Which communication fault was active per interval, empty where none.
    active_comm_fault: np.ndarray

    @property
    def arrival_lag_seconds(self) -> np.ndarray:
        return (self.ingest_time - self.event_time).total_seconds().to_numpy()

    @property
    def delivery_ratio(self) -> float:
        return float(np.mean(self.delivered)) if len(self.delivered) else 0.0


class TransportLayer:
    """Delivers observations to the collector.

    Phase 2 implements the lossless case only, and says so rather than silently
    ignoring a configuration it does not honour: a non-pass-through
    :class:`~gridguard.simulation.config.TransportConfig` raises instead of
    quietly producing lossless output that a caller would read as evidence.
    """

    def __init__(self, config: SimulationConfig, schedule: FaultSchedule | None = None) -> None:
        self.config = config
        self.schedule = schedule or FaultSchedule()

    def run(self, observation: Observation) -> Delivery:
        settings = self.config.transport
        if not settings.is_pass_through:
            raise NotImplementedError(
                "Stochastic transport degradation is not implemented until phase 5. "
                f"This configuration requests loss_rate={settings.loss_rate}, "
                f"mean_delay_seconds={settings.mean_delay_seconds}, "
                f"reorder={settings.reorder}. Refusing rather than returning lossless "
                "output that would be mistaken for a result. Scheduled comm_dropout "
                "faults are supported and do work — see simulation.faults."
            )

        n = len(observation.event_time)
        # Combine the per-channel sensor flags: a record is as good as its worst
        # channel. Flags are OR-ed, never cleared — transport can add MISSING or
        # OUT_OF_ORDER, but it may not decide a sensor flag was mistaken.
        quality = observation.irradiance_quality | observation.power_quality

        # Scheduled communication faults. Nothing about the *value* is touched:
        # the instrument read correctly and the reading was lost in transit.
        # Only whether it arrived changes.
        comm_faults = self.schedule.for_layer("transport")
        delivered = delivery_mask(comm_faults, observation.event_time)

        active_comm_fault = np.full(n, "", dtype=object)
        for fault in comm_faults:
            window = fault.mask(observation.event_time)
            active_comm_fault[window] = fault.fault_type.value

        # A record that never arrived is MISSING, and the flag is set on the
        # collector's view of it rather than on the reading, which was fine.
        quality = self.mark_missing(quality, ~delivered)

        return Delivery(
            site_id=observation.site_id,
            event_time=observation.event_time,
            # Nothing that arrived travelled through a delay, so arrival lag is
            # genuinely zero. Phase 5 makes this a real quantity.
            ingest_time=observation.event_time,
            sequence=np.arange(n, dtype=np.int64),
            delivered=delivered,
            quality=quality,
            active_comm_fault=active_comm_fault,
        )

    @staticmethod
    def mark_missing(quality: np.ndarray, missing: np.ndarray) -> np.ndarray:
        """OR the MISSING flag into a quality column.

        Used by scheduled communication faults, and by phase 5's stochastic
        loss, so both set flags through one path rather than inventing a second
        convention.
        """
        out = quality.copy()
        out[missing] |= int(QualityFlag.MISSING)
        return out
