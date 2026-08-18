"""
Detection evaluation against labelled fault scenarios.

The distinction this module enforces
------------------------------------
GridGuard reports two kinds of accuracy and never conflates them:

**Forecast accuracy** is measured on *measured* telemetry. The target — actual
generation — is genuine, so MAE and RMSE there describe real predictive skill.
That lives in :mod:`gridguard.models.evaluate`.

**Detection accuracy** is measured here, on *injected* faults, because measured
PV telemetry carries no trustworthy fault labels. Nobody annotated the NIST
arrays interval by interval, and deriving labels from the same residuals the
detector consumes would be circular.

So every number this module produces describes performance against **simulated
failures**. It is evidence that the detection logic works as designed. It is not
a field-validated detection rate, and GridGuard does not present it as one.

Metrics
-------
*Interval level* — precision, recall, F1 over 15-minute intervals. Answers
"what fraction of faulted intervals were caught?"

*Event level* — an injected fault counts as detected when the detector flags any
interval inside it. Closer to how an operator experiences the system: catching a
four-hour outage twenty minutes late is a success, not 80 separate failures.

*Detection latency* — time from fault onset to first flag. The metric an O&M
team actually cares about, since lost energy accrues until someone is notified.

*False alarms per day* — flagged intervals grouped into contiguous events, over
healthy periods only. Reported per day because that is the unit in which alert
fatigue is felt.

*Lost-energy error* — estimated versus true lost energy. True loss is exact
here: the generator retains pre-fault generation in ``ac_power_baseline_kw``.

Non-loss conditions
-------------------
``clipping``, ``sensor_dropout`` and ``comm_dropout`` are injected as
*negatives*. They perturb the data without destroying generation, and a detector
that flags them is producing false alarms. Scoring them separately is what stops
the headline recall number from hiding a detector that simply alarms constantly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from gridguard.anomaly.detect import INTERVAL_HOURS
from gridguard.data.faults import FaultEvent, FaultType

logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Interval-level and event-level detection performance."""

    # interval level
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")

    # event level
    events_total: int = 0
    events_detected: int = 0
    event_recall: float = float("nan")

    # operational
    false_alarm_events: int = 0
    healthy_days: float = 0.0
    false_alarms_per_day: float = float("nan")
    median_detection_latency_minutes: float | None = None

    # energy accounting
    true_lost_kwh: float = 0.0
    estimated_lost_kwh: float = 0.0
    lost_energy_error_pct: float | None = None

    per_fault_type: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, float) and np.isfinite(value):
                payload[key] = round(value, 4)
        return payload

    def summary(self) -> str:
        return (
            f"interval P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} | "
            f"events {self.events_detected}/{self.events_total} "
            f"({self.event_recall:.0%}) | "
            f"false alarms {self.false_alarms_per_day:.2f}/day | "
            f"latency {self.median_detection_latency_minutes}min"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_detection(
    df: pd.DataFrame,
    events: list[FaultEvent] | None = None,
) -> DetectionMetrics:
    """Score detector output against injected ground truth.

    Args:
        df: Detector output. Must carry ``timestamp``, ``is_anomaly``,
            ``lost_energy_kwh``, and the injection labels
            ``is_generation_loss`` / ``injected_fault_type``. When
            ``ac_power_baseline_kw`` is present, true lost energy is computed
            exactly from it.
        events: The injected events, used for event-level recall and latency.
            Interval-level metrics are still produced without them.

    Returns:
        Populated :class:`DetectionMetrics`.
    """
    required = {"timestamp", "is_anomaly"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Detector output is missing columns: {sorted(missing)}")

    metrics = DetectionMetrics()

    if "is_generation_loss" not in df.columns:
        metrics.notes.append(
            "No ground-truth labels present; detection metrics cannot be computed."
        )
        return metrics

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    predicted = frame["is_anomaly"].fillna(False).to_numpy(dtype=bool)
    actual = frame["is_generation_loss"].fillna(False).to_numpy(dtype=bool)

    # Only daylight intervals are assessable: at night there is no generation
    # to lose, so counting those as easy true negatives would inflate every
    # rate. Detection has already restricted flags to daylight.
    assessable = frame.get("irradiance_wm2", pd.Series(np.ones(len(frame)) * 1000)).to_numpy() > 50
    predicted, actual = predicted & assessable, actual & assessable

    metrics.true_positives = int(np.sum(predicted & actual))
    metrics.false_positives = int(np.sum(predicted & ~actual))
    metrics.false_negatives = int(np.sum(~predicted & actual))
    metrics.true_negatives = int(np.sum(~predicted & ~actual & assessable))

    metrics.precision = _safe_divide(
        metrics.true_positives, metrics.true_positives + metrics.false_positives
    )
    metrics.recall = _safe_divide(
        metrics.true_positives, metrics.true_positives + metrics.false_negatives
    )
    if np.isfinite(metrics.precision) and np.isfinite(metrics.recall):
        denominator = metrics.precision + metrics.recall
        metrics.f1 = _safe_divide(2 * metrics.precision * metrics.recall, denominator)

    _score_false_alarms(frame, predicted, actual, metrics)
    _score_energy(frame, metrics)

    if events:
        _score_events(frame, events, metrics)

    if "injected_fault_type" in frame.columns:
        metrics.per_fault_type = _score_per_fault_type(frame, predicted, actual, events)

    metrics.notes.append(
        "Computed against injected fault scenarios, not observed field failures. "
        "These figures describe how the detector behaves on simulated faults."
    )
    logger.info("Detection evaluation: %s", metrics.summary())
    return metrics


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end index pairs for each contiguous True run."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _score_false_alarms(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    actual: np.ndarray,
    metrics: DetectionMetrics,
) -> None:
    """Group false-positive intervals into events and normalise per healthy day."""
    false_positive_mask = predicted & ~actual
    metrics.false_alarm_events = len(_contiguous_runs(false_positive_mask))

    healthy = ~actual
    healthy_days = frame.loc[healthy, "timestamp"].dt.normalize().nunique()
    metrics.healthy_days = float(healthy_days)
    metrics.false_alarms_per_day = _safe_divide(metrics.false_alarm_events, healthy_days)


def _score_energy(frame: pd.DataFrame, metrics: DetectionMetrics) -> None:
    """Compare estimated lost energy against exact ground truth."""
    if "lost_energy_kwh" in frame.columns:
        metrics.estimated_lost_kwh = float(frame["lost_energy_kwh"].sum())

    if "ac_power_baseline_kw" in frame.columns:
        shortfall = (frame["ac_power_baseline_kw"] - frame["ac_power_kw"]).clip(lower=0)
        # Only count loss during genuine generation-loss faults; a comm dropout
        # blanks power without destroying any energy.
        if "is_generation_loss" in frame.columns:
            shortfall = shortfall.where(frame["is_generation_loss"].fillna(False), 0.0)
        metrics.true_lost_kwh = float(shortfall.sum() * INTERVAL_HOURS)

    if metrics.true_lost_kwh > 0:
        metrics.lost_energy_error_pct = float(
            100 * (metrics.estimated_lost_kwh - metrics.true_lost_kwh) / metrics.true_lost_kwh
        )


def _score_events(
    frame: pd.DataFrame,
    events: list[FaultEvent],
    metrics: DetectionMetrics,
) -> None:
    """Event-level recall and detection latency."""
    loss_events = [e for e in events if e.is_generation_loss]
    metrics.events_total = len(loss_events)

    latencies: list[float] = []
    detected = 0

    for event in loss_events:
        window = (frame["timestamp"] >= event.start) & (frame["timestamp"] <= event.end)
        inside = (
            frame[window & (frame["irradiance_wm2"] > 50)]
            if "irradiance_wm2" in frame
            else frame[window]
        )
        if inside.empty:
            continue
        flags = inside["is_anomaly"].fillna(False).to_numpy(dtype=bool)
        if flags.any():
            detected += 1
            first_flag_time = inside.loc[inside["is_anomaly"].fillna(False), "timestamp"].iloc[0]
            # Latency is measured from the first assessable interval in the
            # window, not the nominal start: a fault beginning at 03:00 cannot
            # be seen until the sun is up, and charging that to the detector
            # would be misleading.
            onset = inside["timestamp"].iloc[0]
            latencies.append((first_flag_time - onset).total_seconds() / 60)

    metrics.events_detected = detected
    metrics.event_recall = _safe_divide(detected, metrics.events_total)
    if latencies:
        metrics.median_detection_latency_minutes = float(np.median(latencies))


def _score_per_fault_type(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    actual: np.ndarray,
    events: list[FaultEvent] | None,
) -> dict[str, dict]:
    """Break performance down by fault class.

    Loss faults are scored on recall; non-loss conditions are scored on false
    alarm rate, since flagging them at all is the error.
    """
    result: dict[str, dict] = {}
    fault_types = frame["injected_fault_type"].fillna("")

    for fault_type in sorted(t for t in fault_types.unique() if t):
        mask = (fault_types == fault_type).to_numpy()
        try:
            is_loss = FaultType(fault_type) not in (
                FaultType.SENSOR_DROPOUT,
                FaultType.COMM_DROPOUT,
                FaultType.CLIPPING,
            )
        except ValueError:
            is_loss = True

        intervals = int(mask.sum())
        flagged = int(np.sum(predicted & mask))

        entry: dict[str, object] = {
            "intervals": intervals,
            "flagged_intervals": flagged,
            "is_generation_loss": is_loss,
        }
        if is_loss:
            entry["recall"] = round(_safe_divide(flagged, intervals), 4)
            entry["interpretation"] = "Fraction of faulted intervals detected (higher is better)."
        else:
            entry["false_alarm_rate"] = round(_safe_divide(flagged, intervals), 4)
            entry["interpretation"] = (
                "Fraction of intervals wrongly flagged; this condition is not a "
                "generation loss, so lower is better."
            )
        result[fault_type] = entry

    del actual, events  # kept in the signature for symmetry with callers
    return result


def detection_report_frame(metrics: DetectionMetrics) -> pd.DataFrame:
    """Per-fault-class metrics as a tidy frame, for CSV export."""
    rows = []
    for fault_type, entry in metrics.per_fault_type.items():
        rows.append(
            {
                "fault_type": fault_type,
                "is_generation_loss": entry.get("is_generation_loss"),
                "intervals": entry.get("intervals"),
                "flagged_intervals": entry.get("flagged_intervals"),
                "recall": entry.get("recall"),
                "false_alarm_rate": entry.get("false_alarm_rate"),
            }
        )
    return pd.DataFrame(rows)
