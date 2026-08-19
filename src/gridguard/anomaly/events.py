"""
Anomaly event grouping: collapse consecutive flagged 15-minute intervals into events.

An "event" is a contiguous run of is_anomaly=True intervals. Grouping is valuable
because operators care about "there was a 2-hour outage" not "312 individual flags".

Algorithm:
  1. Sort by timestamp (assumes 15-min regularity, handles gaps by treating any
     break in consecutive indices as a new event).
  2. Assign an event_id using cumsum on the boundary mask (changes in is_anomaly
     or timestamp gaps > 20 min).
  3. Aggregate per event_id.

Severity
--------
Severity is the **fraction of expected generation lost** during the event, not
an absolute kWh figure. An absolute threshold is meaningless across a fleet
spanning 60 kW to 500 kW: 10 kWh lost is a rounding error for a 500 kW field and
a total outage for a 60 kW roof.

  high    >= 50% of expected generation lost
  medium  >= 20%
  low     below that

A small absolute floor still applies so that a fractionally-large but
energetically-trivial blip at dawn is not escalated.

Fleet-level correlation of events across sites is handled separately, in
:mod:`gridguard.spatial.context`.
"""

from __future__ import annotations

import pandas as pd

INTERVAL_MINUTES = 15
GAP_THRESHOLD_MINUTES = 20  # gaps larger than this break event continuity

#: Consecutive flagged intervals required before a run becomes an event.
#:
#: This matters more than it looks. The conformal detector is *designed* to flag
#: alpha (5%) of healthy intervals — that is what the coverage guarantee means.
#: Promoting every isolated flag to an event therefore yields roughly one
#: "event" per twenty healthy daylight intervals, burying real faults under
#: hundreds of single-interval blips.
#:
#: Requiring persistence converts a per-interval false-positive rate into a far
#: lower per-event one: for roughly independent intervals, three in a row occurs
#: at about alpha^3. Physically it is also the right filter — a passing cloud
#: edge produces one bad interval, whereas an inverter fault, soiling, or
#: shading persists. Genuine faults comfortably exceed 45 minutes; the cost is
#: that a true fault shorter than that is not reported as an event, though its
#: intervals remain flagged in the anomaly feed.
MIN_EVENT_INTERVALS = 3

#: Fraction of expected generation lost that escalates an event.
HIGH_SHORTFALL_FRACTION = 0.50
MEDIUM_SHORTFALL_FRACTION = 0.20

#: Events losing less than this are capped at "low" regardless of fraction,
#: so a near-total shortfall of almost no energy is not called critical.
MIN_MATERIAL_LOSS_KWH = 0.5


def group_anomaly_events(
    anomaly_df: pd.DataFrame,
    min_intervals: int = MIN_EVENT_INTERVALS,
) -> pd.DataFrame:
    """Group consecutive anomaly intervals into events.

    Args:
        anomaly_df:    Output of detect_anomalies() — must have columns
                       timestamp, is_anomaly, ac_power_kw, predicted_kw,
                       residual_sigma, lost_energy_kwh.
        min_intervals: Consecutive flagged intervals required to report an
                       event. See :data:`MIN_EVENT_INTERVALS` for why this
                       defaults above 1.

    Returns:
        DataFrame with one row per event, sorted by start_time descending.
        Empty DataFrame (correct schema) if there are no qualifying events.
    """
    anom = anomaly_df[anomaly_df["is_anomaly"]].copy()
    anom = anom.sort_values("timestamp").reset_index(drop=True)

    if anom.empty:
        return _empty_events_df()

    # Detect event boundaries: gap > threshold OR start of series
    ts_diff = anom["timestamp"].diff().dt.total_seconds().div(60)  # minutes
    boundary = (ts_diff > GAP_THRESHOLD_MINUTES) | ts_diff.isna()
    anom["event_id"] = boundary.cumsum().astype(int)

    # Drop runs too short to be distinguishable from the detector's designed
    # false-positive rate.
    if min_intervals > 1:
        run_lengths = anom.groupby("event_id")["event_id"].transform("size")
        anom = anom[run_lengths >= min_intervals]
        if anom.empty:
            return _empty_events_df()

    aggregations = {
        "start_time": ("timestamp", "min"),
        "end_time": ("timestamp", "max"),
        "interval_count": ("timestamp", "count"),
        "total_lost_kwh": ("lost_energy_kwh", "sum"),
        "max_residual_sigma": ("residual_sigma", lambda s: s.abs().max()),
        "mean_actual_kw": ("ac_power_kw", "mean"),
        "mean_predicted_kw": ("predicted_kw", "mean"),
    }
    if "expected_lower_kw" in anom.columns:
        aggregations["mean_expected_lower_kw"] = ("expected_lower_kw", "mean")

    events = anom.groupby("event_id").agg(**aggregations).reset_index()

    # Expected energy over the event, used to express severity as a fraction.
    events["expected_kwh"] = (
        events["mean_predicted_kw"] * events["interval_count"] * INTERVAL_MINUTES / 60
    )
    events["shortfall_fraction"] = (
        events["total_lost_kwh"] / events["expected_kwh"].where(events["expected_kwh"] > 0)
    ).fillna(0.0)

    # Duration = from start of first interval to end of last interval
    events["duration_minutes"] = (
        (
            (events["end_time"] - events["start_time"]).dt.total_seconds() / 60
            + INTERVAL_MINUTES  # include the last interval itself
        )
        .round(0)
        .astype(int)
    )

    events["severity"] = [
        _classify_severity(lost, fraction)
        for lost, fraction in zip(
            events["total_lost_kwh"], events["shortfall_fraction"], strict=True
        )
    ]

    # Add site_id if present in source
    if "site_id" in anomaly_df.columns:
        site_map = anom.groupby("event_id")["site_id"].first()
        events["site_id"] = events["event_id"].map(site_map)

    events = events.sort_values("start_time", ascending=False).reset_index(drop=True)
    # Re-assign event_ids as sequential integers (most-recent = 1)
    events["event_id"] = range(1, len(events) + 1)

    # Plain-English explanation for each event
    events["explanation"] = events.apply(explain_event_text, axis=1)

    cols = [
        "event_id",
        "start_time",
        "end_time",
        "duration_minutes",
        "interval_count",
        "total_lost_kwh",
        "expected_kwh",
        "shortfall_fraction",
        "max_residual_sigma",
        "mean_actual_kw",
        "mean_predicted_kw",
        "severity",
        "explanation",
    ]
    if "mean_expected_lower_kw" in events.columns:
        cols.insert(cols.index("mean_predicted_kw") + 1, "mean_expected_lower_kw")
    if "site_id" in events.columns:
        cols.append("site_id")

    return events[cols]


def _classify_severity(lost_kwh: float, shortfall_fraction: float) -> str:
    """Severity from the fraction of expected generation lost.

    Capacity-independent by construction: a fraction is comparable across a
    60 kW roof and a 500 kW field, where an absolute kWh threshold is not.
    """
    if lost_kwh < MIN_MATERIAL_LOSS_KWH:
        return "low"
    if shortfall_fraction >= HIGH_SHORTFALL_FRACTION:
        return "high"
    if shortfall_fraction >= MEDIUM_SHORTFALL_FRACTION:
        return "medium"
    return "low"


def explain_event_text(event: pd.Series | dict) -> str:
    """Return a plain-English summary of an anomaly event.

    Example output:
        "During this 3h 15min event on 2023-06-15, the model expected 45.2 kW
        average output but actual generation was 13.5 kW — 70% below forecast.
        Estimated lost energy: 23.6 kWh. Severity: high."
    """
    duration = int(event["duration_minutes"])
    lost = float(event["total_lost_kwh"])
    severity = str(event["severity"])
    predicted = float(event["mean_predicted_kw"])
    actual = float(event["mean_actual_kw"])

    # Format duration
    if duration >= 60:
        h, m = divmod(duration, 60)
        dur_str = f"{h}h {m}min" if m else f"{h}h"
    else:
        dur_str = f"{duration}min"

    # Date string from start_time
    try:
        date_str = f" on {pd.Timestamp(event['start_time']).strftime('%Y-%m-%d')}"
    except Exception:
        date_str = ""

    pct_drop = round(100 * (1 - actual / predicted)) if predicted > 0 else 0

    bound_clause = ""
    lower = event.get("mean_expected_lower_kw") if hasattr(event, "get") else None
    if lower is not None and pd.notna(lower):
        bound_clause = (
            f" The calibrated lower bound of healthy output was {float(lower):.1f} kW, so actual "
            f"generation fell outside the expected range rather than merely below the point "
            f"forecast."
        )

    return (
        f"During this {dur_str} event{date_str}, the model expected {predicted:.1f} kW "
        f"average output but actual generation was {actual:.1f} kW — {pct_drop}% below forecast."
        f"{bound_clause} Estimated lost energy: {lost:.1f} kWh. Severity: {severity}."
    )


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "start_time",
            "end_time",
            "duration_minutes",
            "interval_count",
            "total_lost_kwh",
            "expected_kwh",
            "shortfall_fraction",
            "max_residual_sigma",
            "mean_actual_kw",
            "mean_predicted_kw",
            "mean_expected_lower_kw",
            "severity",
            "explanation",
        ]
    )
