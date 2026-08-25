"""
Faults, as changes to a layer rather than edits to a column.

What changes from the batch injector
------------------------------------
:mod:`gridguard.data.faults` injects faults by rewriting values in a finished
canonical frame. That is the right shape for scoring a detector against a fixed
dataset, and it stays exactly as it is — the shipped pipeline depends on it.

It cannot express the thing this project is now about. Rewriting
``ac_power_kw`` to ``NaN`` for a communication dropout makes a *lost message*
indistinguishable from a *dead inverter that reported honestly*, because both
end up as an absent number in one column. The information that would tell them
apart — that the plant kept generating and the reading simply never arrived —
is destroyed at the moment of injection.

So here a fault is owned by the layer whose behaviour it changes:

===================  ==========  ====================================================
Fault                Layer       What it changes
===================  ==========  ====================================================
complete_outage      equipment   Asset state → OFFLINE. Real generation stops.
partial_outage       equipment   Asset state → DERATED. Real generation drops.
persistent_derate    equipment   Asset state → DERATED, sustained.
gradual_degradation  equipment   Asset state → DERATED, ramping.
clipping             equipment   AC cap. Healthy behaviour, not a fault.
shading               equipment   Incident irradiance on one array is reduced.
sensor_dropout       sensing     A channel freezes. Truth is untouched.
comm_dropout         transport   Records never arrive. Truth and observation
                                 are both untouched.
===================  ==========  ====================================================

Three properties follow from that, and each is asserted in tests:

1. A **sensor** fault cannot change ground truth. The plant did what it did.
2. A **communication** fault cannot change ground truth *or* the observation.
   The instrument read correctly; the reading was lost in transit.
3. An **equipment** fault changes real generation, and the loss is recorded as
   the gap between potential and actual — not inferred later from a model.

Ground truth records which fault was active on which asset at which instant, so
a phase 8 attribution engine can be scored against the answer rather than
against another model's opinion.

Honesty
-------
These are simulated failure modes, not observed field incidents. Nothing derived
from them may be described as a measured failure rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from gridguard.data.faults import FAULT_DESCRIPTIONS, FaultType
from gridguard.domain.fault import FaultCategory, category_for

logger = logging.getLogger(__name__)

#: Which layer applies each fault class. The mapping is exhaustive over
#: ``FaultType`` and is checked as such in tests: a new fault class must declare
#: an owner rather than silently defaulting to one.
FAULT_LAYER: dict[FaultType, str] = {
    FaultType.COMPLETE_OUTAGE: "equipment",
    FaultType.PARTIAL_OUTAGE: "equipment",
    FaultType.PERSISTENT_DERATE: "equipment",
    FaultType.GRADUAL_DEGRADATION: "equipment",
    FaultType.CLIPPING: "equipment",
    FaultType.SHADING: "equipment",
    FaultType.SENSOR_DROPOUT: "sensing",
    FaultType.COMM_DROPOUT: "transport",
}

#: Irradiance below which there is no generation to lose.
DAYLIGHT_THRESHOLD_WM2 = 50.0


@dataclass(frozen=True)
class FaultSpec:
    """One scheduled fault, attached to an asset and a window of simulated time.

    ``asset_id`` is required and is the substantive difference from the batch
    injector's site-level events: attribution is a claim about a component, so
    the ground truth it is scored against has to name one.
    """

    fault_type: FaultType
    asset_id: str
    start: datetime
    end: datetime
    #: Fraction of output lost at peak, 0..1. For ``clipping`` it is the
    #: fraction of nameplate the inverter caps below.
    severity: float = 0.5

    #: For faults that recur within the same hours each day — shading is the
    #: only one today — the UTC hour band they apply in, as ``(lo, hi)``.
    #: ``None`` means the fault applies continuously across its window.
    #:
    #: Explicit rather than inferred from the window's start and end hours,
    #: which is how the batch injector does it. That inference silently
    #: produced a shading fault that cost nothing at all: its window began and
    #: ended at the same clock hour, so the band collapsed to a single instant,
    #: and that instant was at night.
    daily_hours: tuple[float, float] | None = None

    #: For ``clipping`` only: cap AC output at this percentile of the array's
    #: own daylight potential, instead of at ``capacity * (1 - severity)``.
    #:
    #: Nameplate is the wrong reference here. These arrays peak near 75% of DC
    #: nameplate once the system derate and temperature losses apply, and how
    #: far below that they land varies with tilt, azimuth and latitude — so a
    #: nameplate-relative cap that engages at one site sits above the peak at
    #: another. Tuned per site by hand, it silently stops engaging the moment a
    #: site's geometry changes. A percentile of the array's own output is
    #: self-calibrating and states the intent directly: the inverter caps on
    #: the brightest intervals.
    cap_percentile: float | None = None

    note: str = ""

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError(
                f"{self.fault_type}: fault windows must be timezone-aware UTC, to match "
                "simulated event_time."
            )
        if self.end < self.start:
            raise ValueError(f"{self.fault_type}: end {self.end} precedes start {self.start}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"{self.fault_type}: severity must be in [0, 1], got {self.severity}")
        if self.cap_percentile is not None and not 0.0 < self.cap_percentile < 100.0:
            raise ValueError(
                f"{self.fault_type}: cap_percentile must be in (0, 100), "
                f"got {self.cap_percentile}"
            )
        if self.daily_hours is not None:
            lo, hi = self.daily_hours
            if not (0.0 <= lo <= 24.0 and 0.0 <= hi <= 24.0):
                raise ValueError(f"{self.fault_type}: daily_hours must lie in [0, 24]")

    @property
    def category(self) -> FaultCategory:
        return category_for(self.fault_type)

    @property
    def layer(self) -> str:
        return FAULT_LAYER[self.fault_type]

    @property
    def is_generation_loss(self) -> bool:
        from gridguard.data.faults import GENERATION_LOSS_FAULTS

        return self.fault_type in GENERATION_LOSS_FAULTS

    @property
    def description(self) -> str:
        return self.note or FAULT_DESCRIPTIONS[self.fault_type]

    def mask(self, event_time: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask of the intervals this fault spans.

        ``np.asarray`` rather than ``.to_numpy()``: comparing a DatetimeIndex
        against a Timestamp already yields an ndarray, while the same
        comparison on a Series yields a Series. Accepting either keeps callers
        from having to know which they hold.
        """
        window = (event_time >= pd.Timestamp(self.start)) & (event_time <= pd.Timestamp(self.end))
        return np.asarray(window, dtype=bool)


@dataclass(frozen=True)
class FaultSchedule:
    """Every fault in a run, queryable by the layer that must apply it."""

    faults: tuple[FaultSpec, ...] = ()

    def __len__(self) -> int:
        return len(self.faults)

    def __iter__(self):
        return iter(self.faults)

    def for_layer(self, layer: str) -> tuple[FaultSpec, ...]:
        return tuple(f for f in self.faults if f.layer == layer)

    def for_asset(self, asset_id: str) -> tuple[FaultSpec, ...]:
        return tuple(f for f in self.faults if f.asset_id == asset_id)

    def for_site(self, site_id: str) -> FaultSchedule:
        """Faults whose asset belongs to a site.

        Asset ids are derived as ``<site_id>_<kind>_<ordinal>``, so the prefix
        is a reliable test — and :func:`gridguard.domain.ids.validate_id` keeps
        site ids from containing anything that would make it ambiguous.
        """
        return FaultSchedule(tuple(f for f in self.faults if f.asset_id.startswith(f"{site_id}_")))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fault_type": f.fault_type.value,
                    "category": f.category.value,
                    "layer": f.layer,
                    "asset_id": f.asset_id,
                    "start": f.start,
                    "end": f.end,
                    "severity": f.severity,
                    "is_generation_loss": f.is_generation_loss,
                    "description": f.description,
                }
                for f in self.faults
            ]
        )


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


def build_layered_scenario(
    asset_ids: dict[str, str],
    clock_index: pd.DatetimeIndex,
    *,
    seed: int = 42,
    include: tuple[FaultType, ...] | None = None,
    generating_hour_utc: float = 17.0,
) -> FaultSchedule:
    """A scenario spanning the taxonomy, attached to real assets.

    Each class is placed in its own non-overlapping window so that a detector's
    response to one is not confounded by another running at the same time. That
    matters more here than in the batch injector: the point of the layered
    system is to tell fault classes apart, which an overlapping schedule would
    make impossible to score.

    Args:
        asset_ids: which asset each fault class attaches to, keyed by
            ``"array"``, ``"pyranometer"`` and ``"link"``.
        clock_index: the run's simulated instants.
        seed: unused placeholder retained for signature stability with the
            batch scenario builder; placement here is fully deterministic.
        include: restrict to these classes. Defaults to the whole taxonomy.
        generating_hour_utc: the UTC hour that is solar noon for the fleet, used
            to anchor windows onto the generating part of the day. 17:00 UTC is
            local noon for the DMV sites (UTC-5).

    Returns:
        A schedule, empty if the run is too short to host separated windows.
    """
    del seed  # placement is deterministic; kept for signature symmetry

    classes = include or tuple(FaultType)
    if len(clock_index) < 96 * len(classes):
        logger.debug(
            "Run of %d intervals is too short to host %d separated fault windows.",
            len(clock_index),
            len(classes),
        )
        return FaultSchedule()

    array = asset_ids["array"]
    pyranometer = asset_ids.get("pyranometer", array)
    link = asset_ids.get("link", array)

    # Divide the run into one slot per class, and place each fault inside the
    # middle of its own slot so windows never touch.
    slot = len(clock_index) // len(classes)
    schedule: list[FaultSpec] = []

    defaults: dict[FaultType, tuple[str, float, int]] = {
        # (asset key, severity, duration in intervals)
        FaultType.COMPLETE_OUTAGE: ("array", 1.0, 24),
        FaultType.PARTIAL_OUTAGE: ("array", 0.45, 48),
        FaultType.PERSISTENT_DERATE: ("array", 0.20, 96 * 3),
        FaultType.GRADUAL_DEGRADATION: ("array", 0.30, 96 * 5),
        FaultType.SHADING: ("array", 0.55, 96 * 2),
        FaultType.CLIPPING: ("array", 0.25, 96),
        FaultType.SENSOR_DROPOUT: ("pyranometer", 1.0, 48),
        FaultType.COMM_DROPOUT: ("link", 1.0, 32),
    }

    for position, fault_type in enumerate(classes):
        asset_key, severity, duration = defaults[fault_type]
        asset = {"array": array, "pyranometer": pyranometer, "link": link}[asset_key]

        slot_start = position * slot
        centred = slot_start + max(1, (slot - duration) // 2)
        begin = _snap_to_generating_hour(clock_index, centred, generating_hour_utc)
        begin = min(begin, len(clock_index) - 2)
        finish = min(begin + duration, len(clock_index) - 1)

        schedule.append(
            FaultSpec(
                fault_type=fault_type,
                asset_id=asset,
                start=clock_index[begin].to_pydatetime(),
                end=clock_index[finish].to_pydatetime(),
                severity=severity,
                # Shading recurs within the same hours each day. Centred on the
                # generating part of the day, since shading that falls entirely
                # at night costs nothing and tests nothing.
                daily_hours=(
                    (generating_hour_utc - 2.0, generating_hour_utc + 2.0)
                    if fault_type is FaultType.SHADING
                    else None
                ),
                # Cap on the brightest 10% of daylight intervals. Self-calibrating
                # across sites, where a nameplate-relative cap is not: at 25% below
                # nameplate, clipping engaged at two of three sites tried and not at
                # the third, which made the class look present while testing nothing.
                cap_percentile=90.0 if fault_type is FaultType.CLIPPING else None,
            )
        )

    return FaultSchedule(tuple(schedule))


def _snap_to_generating_hour(
    clock_index: pd.DatetimeIndex, position: int, target_hour_utc: float
) -> int:
    """Move a window start forward to the next interval near solar noon.

    Without this, windows land wherever arithmetic on tick indices puts them —
    which in practice was mostly at night. A fault scheduled at night costs no
    generation and exercises no detector, so the scenario looked complete while
    testing almost nothing. Two of the eight classes were affected before this
    existed: shading cost exactly zero, and a 45%% partial outage over twelve
    hours cost 9 kWh.
    """
    hours = clock_index.hour.to_numpy() + clock_index.minute.to_numpy() / 60.0
    tail = np.flatnonzero(np.abs(hours[position:] - target_hour_utc) < 0.5)
    return position + int(tail[0]) if len(tail) else position


# ---------------------------------------------------------------------------
# Per-layer application
# ---------------------------------------------------------------------------


def equipment_multiplier(
    fault: FaultSpec,
    event_time: pd.DatetimeIndex,
    irradiance: np.ndarray,
) -> np.ndarray:
    """Multiplicative factor this equipment fault applies to real generation.

    Reproduces the batch injector's physics exactly, so a layered run and a
    batch run of the same fault agree. Returns ones where the fault is inactive.
    """
    window = fault.mask(event_time)
    factor = np.ones(len(event_time))
    if not window.any():
        return factor

    # A fault can only take generation that exists.
    daylight = irradiance > DAYLIGHT_THRESHOLD_WM2
    active = window & daylight
    ft = fault.fault_type

    if ft is FaultType.COMPLETE_OUTAGE:
        factor[active] = 0.0

    elif ft in (FaultType.PARTIAL_OUTAGE, FaultType.PERSISTENT_DERATE):
        factor[active] = 1.0 - fault.severity

    elif ft is FaultType.SHADING:
        # Confined to the same hours each day, so a multi-day shading fault
        # recurs rather than applying continuously.
        if fault.daily_hours is None:
            factor[active] = 1.0 - fault.severity
        else:
            hours = event_time.hour.to_numpy() + event_time.minute.to_numpy() / 60.0
            lo, hi = fault.daily_hours
            in_hours = (hours >= lo) & (hours <= hi) if lo <= hi else (hours >= lo) | (hours <= hi)
            factor[active & in_hours] = 1.0 - fault.severity

    elif ft is FaultType.GRADUAL_DEGRADATION:
        # Loss ramps linearly from zero at the start to `severity` at the end.
        positions = np.flatnonzero(window)
        ramp = np.linspace(0.0, fault.severity, len(positions))
        full = np.zeros(len(event_time))
        full[positions] = ramp
        factor[active] = 1.0 - full[active]

    elif ft is FaultType.CLIPPING:
        # Handled as an absolute cap by the caller, not a multiplier, because
        # it depends on nameplate rather than on current output.
        pass

    else:  # pragma: no cover - equipment layer never sees the others
        raise ValueError(f"{ft} is not an equipment fault (owned by {FAULT_LAYER[ft]})")

    return factor


def apply_sensor_fault(
    fault: FaultSpec,
    event_time: pd.DatetimeIndex,
    reading: np.ndarray,
) -> np.ndarray:
    """What the instrument reports while this fault is active.

    The truth it was reading is not passed in and cannot be modified — that is
    the point of the signature.
    """
    if fault.fault_type is not FaultType.SENSOR_DROPOUT:
        raise ValueError(
            f"{fault.fault_type} is not a sensor fault (owned by {FAULT_LAYER[fault.fault_type]})"
        )

    window = fault.mask(event_time)
    if not window.any():
        return reading

    out = reading.copy()
    positions = np.flatnonzero(window)
    first = positions[0]
    # Freeze at the last good reading before the fault. At the very start of a
    # run there is none, so the first in-window value stands in.
    frozen = out[first - 1] if first > 0 else out[first]
    out[positions] = frozen
    return out


def delivery_mask(faults: tuple[FaultSpec, ...], event_time: pd.DatetimeIndex) -> np.ndarray:
    """Which intervals reach the collector, given communication faults.

    Returns a boolean array: True where the record arrives. Nothing about the
    value is touched — the reading was correct, it simply never got there.
    """
    delivered = np.ones(len(event_time), dtype=bool)
    for fault in faults:
        if fault.fault_type is not FaultType.COMM_DROPOUT:
            raise ValueError(
                f"{fault.fault_type} is not a communication fault "
                f"(owned by {FAULT_LAYER[fault.fault_type]})"
            )
        delivered &= ~fault.mask(event_time)
    return delivered
