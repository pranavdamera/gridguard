"""
Injected fault scenarios for evaluating the detector.

Why injection at all
--------------------
Measured PV telemetry does not come with trustworthy fault labels. Nobody
annotated the NIST arrays interval-by-interval, and inferring labels from the
same residuals the detector uses would be circular. So GridGuard separates two
questions that are often conflated:

* **Forecast quality** is measured on *real* measured telemetry, where the
  target (actual generation) is genuine and no labels are needed.
* **Detection quality** is measured on *injected* faults, where ground truth is
  known by construction.

Injected faults are simulations of failure modes, not observations of real field
failures, and results derived from them are never presented as field-validated
detection rates.

Fault taxonomy
--------------
Each class models a distinct physical or operational failure with a different
signature, so that precision and recall can be reported per class rather than
as a single aggregate that hides which failures are actually hard.

``complete_outage``      Inverter trip or grid disconnect. Output falls to zero
                         during daylight. The easy case.
``partial_outage``       One string or sub-inverter offline. Output drops by a
                         fixed fraction. Harder: the profile still looks solar.
``persistent_derate``    Soiling, degradation, or a stuck curtailment setpoint.
                         A modest multiplicative loss sustained for days. The
                         case a lag-aware model would hide entirely.
``shading``              Structural or vegetation shading. A deep loss confined
                         to the same hours each day.
``gradual_degradation``  Slow ramp of loss over weeks. Hardest to catch early.
``sensor_dropout``       The irradiance sensor freezes at its last value while
                         generation continues normally. A *data-quality* fault:
                         a naive detector reports underperformance that is not
                         happening.
``comm_dropout``         Telemetry stops arriving; power becomes missing. No
                         generation loss, and nothing to flag.
``clipping``             Inverter output limited at its AC rating on high
                         irradiance. **Not a fault** — normal, healthy design
                         behaviour, included so evaluation measures whether the
                         detector correctly stays silent.

``sensor_dropout``, ``comm_dropout`` and ``clipping`` carry
``is_generation_loss=False``: they are conditions the detector must handle
without raising a false alarm, and they are scored as negatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FaultType(StrEnum):
    COMPLETE_OUTAGE = "complete_outage"
    PARTIAL_OUTAGE = "partial_outage"
    PERSISTENT_DERATE = "persistent_derate"
    SHADING = "shading"
    GRADUAL_DEGRADATION = "gradual_degradation"
    SENSOR_DROPOUT = "sensor_dropout"
    COMM_DROPOUT = "comm_dropout"
    CLIPPING = "clipping"


#: Classes that represent genuine lost generation the detector should flag.
GENERATION_LOSS_FAULTS: frozenset[FaultType] = frozenset(
    {
        FaultType.COMPLETE_OUTAGE,
        FaultType.PARTIAL_OUTAGE,
        FaultType.PERSISTENT_DERATE,
        FaultType.SHADING,
        FaultType.GRADUAL_DEGRADATION,
    }
)

#: Classes the detector should *not* flag: nothing is actually being lost.
NON_LOSS_CONDITIONS: frozenset[FaultType] = frozenset(
    {
        FaultType.SENSOR_DROPOUT,
        FaultType.COMM_DROPOUT,
        FaultType.CLIPPING,
    }
)

FAULT_DESCRIPTIONS: dict[FaultType, str] = {
    FaultType.COMPLETE_OUTAGE: "Inverter trip or grid disconnect — output falls to zero in daylight.",
    FaultType.PARTIAL_OUTAGE: "One string or sub-inverter offline — output drops by a fixed fraction.",
    FaultType.PERSISTENT_DERATE: "Soiling or a stuck curtailment setpoint — sustained multiplicative loss.",
    FaultType.SHADING: "Structural or vegetation shading — deep loss confined to the same hours daily.",
    FaultType.GRADUAL_DEGRADATION: "Slow ramp of loss over weeks — hardest to catch early.",
    FaultType.SENSOR_DROPOUT: "Irradiance sensor frozen while generation continues — data-quality fault, no real loss.",
    FaultType.COMM_DROPOUT: "Telemetry stops arriving — power missing, no generation loss.",
    FaultType.CLIPPING: "Inverter limited at its AC rating — normal healthy behaviour, not a fault.",
}


@dataclass(frozen=True)
class FaultEvent:
    """One injected fault, retained as ground truth for evaluation."""

    fault_type: FaultType
    start: pd.Timestamp
    end: pd.Timestamp
    severity: float  # fraction of output lost at peak, 0..1
    description: str = ""

    @property
    def is_generation_loss(self) -> bool:
        return self.fault_type in GENERATION_LOSS_FAULTS

    def to_dict(self) -> dict:
        return {
            "fault_type": self.fault_type.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "severity": round(float(self.severity), 4),
            "is_generation_loss": self.is_generation_loss,
            "description": self.description or FAULT_DESCRIPTIONS[self.fault_type],
        }


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

#: Label columns added by injection.
LABEL_COLUMNS = [
    "is_injected_fault",
    "injected_fault_type",
    "is_generation_loss",
    "ac_power_baseline_kw",
]


def _ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_injected_fault" not in out.columns:
        out["is_injected_fault"] = False
        out["injected_fault_type"] = ""
        out["is_generation_loss"] = False
        # Snapshot of pre-fault generation. This is what makes exact
        # lost-energy ground truth available: true loss is baseline minus
        # actual, with no need to model what "should" have happened.
        out["ac_power_baseline_kw"] = out["ac_power_kw"]
    return out


def apply_fault(df: pd.DataFrame, event: FaultEvent, *, capacity_kw: float) -> pd.DataFrame:
    """Apply one fault event to a canonical frame, returning a new frame.

    The frame must contain ``timestamp``, ``ac_power_kw`` and ``irradiance_wm2``.
    Label columns are created if absent and updated in place otherwise, so
    several events can be layered onto the same frame.
    """
    out = _ensure_label_columns(df)
    window = (out["timestamp"] >= event.start) & (out["timestamp"] <= event.end)
    if not window.any():
        logger.debug("Fault %s covers no rows (%s - %s)", event.fault_type, event.start, event.end)
        return out

    # Faults only manifest where there is generation to lose.
    daylight = out["irradiance_wm2"] > 50
    active = window & daylight

    ft = event.fault_type

    if ft is FaultType.COMPLETE_OUTAGE:
        out.loc[active, "ac_power_kw"] = 0.0

    elif ft in (FaultType.PARTIAL_OUTAGE, FaultType.PERSISTENT_DERATE):
        out.loc[active, "ac_power_kw"] *= 1.0 - event.severity

    elif ft is FaultType.SHADING:
        # Confine the loss to the hours spanned by the event window each day,
        # so a multi-day shading fault recurs rather than applying continuously.
        hours = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60
        lo = event.start.hour + event.start.minute / 60
        hi = event.end.hour + event.end.minute / 60
        in_hours = (hours >= lo) & (hours <= hi)
        shaded = window & daylight & in_hours
        out.loc[shaded, "ac_power_kw"] *= 1.0 - event.severity

    elif ft is FaultType.GRADUAL_DEGRADATION:
        # Loss ramps linearly from zero at the start to `severity` at the end.
        idx = out.index[window]
        span = max(len(idx) - 1, 1)
        ramp = pd.Series(np.linspace(0.0, event.severity, len(idx)), index=idx)
        factor = 1.0 - ramp
        rows = out.index[active]
        out.loc[rows, "ac_power_kw"] *= factor.reindex(rows).to_numpy()
        del span

    elif ft is FaultType.SENSOR_DROPOUT:
        # The pyranometer freezes at its last good reading; generation is
        # untouched. Downstream this looks like a large negative residual that
        # is entirely an artefact of the sensor.
        first = out.index[window][0]
        prev = max(first - 1, out.index[0])
        frozen = float(out.loc[prev, "irradiance_wm2"])
        out.loc[window, "irradiance_wm2"] = frozen

    elif ft is FaultType.COMM_DROPOUT:
        out.loc[window, "ac_power_kw"] = np.nan

    elif ft is FaultType.CLIPPING:
        # Healthy behaviour: the inverter caps AC output at a rating below the
        # DC array's potential on the brightest intervals.
        cap = capacity_kw * (1.0 - event.severity)
        out.loc[window, "ac_power_kw"] = out.loc[window, "ac_power_kw"].clip(upper=cap)

    else:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"Unhandled fault type: {ft}")

    out.loc[active, "is_injected_fault"] = True
    out.loc[active, "injected_fault_type"] = ft.value
    out.loc[active, "is_generation_loss"] = event.is_generation_loss

    # Comm dropout blanks power outside the daylight mask too, so label the
    # whole window rather than only the daylight part.
    if ft is FaultType.COMM_DROPOUT:
        out.loc[window, "is_injected_fault"] = True
        out.loc[window, "injected_fault_type"] = ft.value
        out.loc[window, "is_generation_loss"] = False

    return out


def apply_faults(
    df: pd.DataFrame,
    events: list[FaultEvent],
    *,
    capacity_kw: float,
) -> pd.DataFrame:
    """Apply a list of fault events in order."""
    out = _ensure_label_columns(df)
    for event in events:
        out = apply_fault(out, event, capacity_kw=capacity_kw)
    out["ac_power_kw"] = out["ac_power_kw"].clip(lower=0)
    return out


def build_evaluation_scenario(
    timestamps: pd.Series,
    *,
    seed: int = 42,
) -> list[FaultEvent]:
    """Construct a reproducible fault scenario spanning every fault class.

    Events are placed on distinct days so that per-class metrics are not
    confounded by overlapping faults. Returns an empty list when the window is
    too short to host the full taxonomy.
    """
    rng = np.random.default_rng(seed)
    days = sorted(pd.to_datetime(pd.Series(timestamps)).dt.normalize().unique())
    ordered = [
        FaultType.COMPLETE_OUTAGE,
        FaultType.PARTIAL_OUTAGE,
        FaultType.SHADING,
        FaultType.SENSOR_DROPOUT,
        FaultType.COMM_DROPOUT,
        FaultType.CLIPPING,
        FaultType.PERSISTENT_DERATE,
        FaultType.GRADUAL_DEGRADATION,
    ]
    # Multi-day classes need room at the end of the window.
    if len(days) < len(ordered) * 2 + 14:
        logger.warning(
            "Window of %d days is too short for the full fault taxonomy; skipping injection.",
            len(days),
        )
        return []

    events: list[FaultEvent] = []
    # Space single-day faults evenly through the first two-thirds of the window.
    single_day_slots = np.linspace(2, int(len(days) * 0.66), num=6, dtype=int)

    for fault_type, slot in zip(ordered[:6], single_day_slots, strict=True):
        day = pd.Timestamp(days[slot])
        if fault_type is FaultType.SHADING:
            start, end = day + pd.Timedelta(hours=14), day + pd.Timedelta(hours=17)
            severity = 0.55
        elif fault_type is FaultType.CLIPPING:
            start, end = day + pd.Timedelta(hours=10), day + pd.Timedelta(hours=15)
            severity = 0.25  # cap at 75% of nameplate
        elif fault_type is FaultType.COMM_DROPOUT:
            start, end = day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=13)
            severity = 1.0
        elif fault_type is FaultType.SENSOR_DROPOUT:
            start, end = day + pd.Timedelta(hours=11), day + pd.Timedelta(hours=15)
            severity = 1.0
        elif fault_type is FaultType.COMPLETE_OUTAGE:
            start, end = day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=16)
            severity = 1.0
        else:  # PARTIAL_OUTAGE
            start, end = day + pd.Timedelta(hours=7), day + pd.Timedelta(hours=18)
            severity = float(rng.uniform(0.3, 0.45))
        events.append(
            FaultEvent(
                fault_type=fault_type,
                start=start,
                end=end,
                severity=severity,
                description=FAULT_DESCRIPTIONS[fault_type],
            )
        )

    # Persistent derate: one week, late in the window.
    derate_start = pd.Timestamp(days[int(len(days) * 0.72)])
    events.append(
        FaultEvent(
            fault_type=FaultType.PERSISTENT_DERATE,
            start=derate_start,
            end=derate_start + pd.Timedelta(days=7),
            severity=0.22,
            description=FAULT_DESCRIPTIONS[FaultType.PERSISTENT_DERATE],
        )
    )

    # Gradual degradation: the final stretch, ramping to a meaningful loss.
    degr_start = pd.Timestamp(days[int(len(days) * 0.85)])
    events.append(
        FaultEvent(
            fault_type=FaultType.GRADUAL_DEGRADATION,
            start=degr_start,
            end=pd.Timestamp(days[-1]) + pd.Timedelta(hours=23, minutes=59),
            severity=0.30,
            description=FAULT_DESCRIPTIONS[FaultType.GRADUAL_DEGRADATION],
        )
    )

    logger.info("Built evaluation scenario with %d injected fault events.", len(events))
    return events
