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
import numpy as np

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
        (events["end_time"] - events["start_time"]).dt.total_seconds() / 60
        + INTERVAL_MINUTES  # include the last interval itself
    ).round(0).astype(int)

    events["severity"] = events["total_lost_kwh"].apply(_classify_severity)

    # Add site_id if present in source
    if "site_id" in anomaly_df.columns:
        site_map = anom.groupby("event_id")["site_id"].first()
        events["site_id"] = events["event_id"].map(site_map)

    events = events.sort_values("start_time", ascending=False).reset_index(drop=True)
    # Re-assign event_ids as sequential integers (most-recent = 1)
    events["event_id"] = range(1, len(events) + 1)

    cols = ["event_id", "start_time", "end_time", "duration_minutes",
            "interval_count", "total_lost_kwh", "max_residual_sigma",
            "mean_actual_kw", "mean_predicted_kw", "severity"]
    if "site_id" in events.columns:
        cols.append("site_id")

    return events[cols]


def _classify_severity(lost_kwh: float) -> str:
    if lost_kwh < 1.0:
        return "low"
    elif lost_kwh < 10.0:
        return "medium"
    return "high"


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id", "start_time", "end_time", "duration_minutes",
            "interval_count", "total_lost_kwh", "max_residual_sigma",
            "mean_actual_kw", "mean_predicted_kw", "severity",
        ]
    )
