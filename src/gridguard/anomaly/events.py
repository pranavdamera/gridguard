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

Severity tiers (based on total_lost_kwh):
  low    — < 1 kWh
  medium — 1–10 kWh
  high   — > 10 kWh

These thresholds are illustrative. Tune them based on system capacity.

TODO: Add spatial grouping across sites for fleet-level event correlation.
TODO: Parameterise severity thresholds by site capacity_kw.
"""

from __future__ import annotations

import pandas as pd

INTERVAL_MINUTES = 15
GAP_THRESHOLD_MINUTES = 20  # gaps larger than this break event continuity


def group_anomaly_events(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive anomaly intervals into events.

    Args:
        anomaly_df: Output of detect_anomalies() — must have columns
                    timestamp, is_anomaly, ac_power_kw, predicted_kw,
                    residual_sigma, lost_energy_kwh.

    Returns:
        DataFrame with one row per event, sorted by start_time descending.
        Empty DataFrame (correct schema) if there are no anomalies.
    """
    anom = anomaly_df[anomaly_df["is_anomaly"]].copy()
    anom = anom.sort_values("timestamp").reset_index(drop=True)

    if anom.empty:
        return _empty_events_df()

    # Detect event boundaries: gap > threshold OR start of series
    ts_diff = anom["timestamp"].diff().dt.total_seconds().div(60)  # minutes
    boundary = (ts_diff > GAP_THRESHOLD_MINUTES) | ts_diff.isna()
    anom["event_id"] = boundary.cumsum().astype(int)

    events = (
        anom.groupby("event_id")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            interval_count=("timestamp", "count"),
            total_lost_kwh=("lost_energy_kwh", "sum"),
            max_residual_sigma=("residual_sigma", lambda s: s.abs().max()),
            mean_actual_kw=("ac_power_kw", "mean"),
            mean_predicted_kw=("predicted_kw", "mean"),
        )
        .reset_index()
    )

    # Duration = from start of first interval to end of last interval
    events["duration_minutes"] = (
        (
            (events["end_time"] - events["start_time"]).dt.total_seconds() / 60
            + INTERVAL_MINUTES  # include the last interval itself
        )
        .round(0)
        .astype(int)
    )

    events["severity"] = events["total_lost_kwh"].apply(_classify_severity)

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
        "max_residual_sigma",
        "mean_actual_kw",
        "mean_predicted_kw",
        "severity",
        "explanation",
    ]
    if "site_id" in events.columns:
        cols.append("site_id")

    return events[cols]


def _classify_severity(lost_kwh: float) -> str:
    if lost_kwh < 1.0:
        return "low"
    elif lost_kwh < 10.0:
        return "medium"
    return "high"


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

    return (
        f"During this {dur_str} event{date_str}, the model expected {predicted:.1f} kW "
        f"average output but actual generation was {actual:.1f} kW — {pct_drop}% below forecast. "
        f"Estimated lost energy: {lost:.1f} kWh. Severity: {severity}."
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
            "max_residual_sigma",
            "mean_actual_kw",
            "mean_predicted_kw",
            "severity",
            "explanation",
        ]
    )
