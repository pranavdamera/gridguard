"""
Arrival-lag and delivery metrics.

The first quantities in this project that describe the *network* rather than the
plant. They exist because "we cannot see the site" and "the site is dark" are
different conditions with different responses, and telling them apart needs
numbers about delivery, not about generation.

All of these read the store. None of them infers anything: arrival lag is
``ingest_time - event_time``, both of which were recorded rather than derived,
and completeness is measured against the intervals that should exist rather than
against the ones that do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from gridguard.collector.store import TelemetryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LagSummary:
    """How long telemetry took to arrive."""

    n: int
    median_seconds: float
    p95_seconds: float
    max_seconds: float

    #: Lag below which delivery counts as live, in seconds.
    #:
    #: Two intervals, not one. A measurement covering 12:00–12:15 does not exist
    #: until 12:15, so its lag can never be less than one interval no matter how
    #: fast the network is — a one-interval threshold would classify even
    #: instantaneous delivery as late. The second interval is the actual budget
    #: for buffering, publishing and being written down.
    live_threshold_seconds: float = 1800.0

    @property
    def is_live(self) -> bool:
        """Whether delivery looks live rather than replayed."""
        return self.median_seconds < self.live_threshold_seconds

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "median_seconds": round(self.median_seconds, 3),
            "p95_seconds": round(self.p95_seconds, 3),
            "max_seconds": round(self.max_seconds, 3),
            "live_threshold_seconds": self.live_threshold_seconds,
            "is_live": self.is_live,
        }


def arrival_lag(
    store: TelemetryStore,
    *,
    site_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> LagSummary | None:
    """Distribution of arrival lag over the selected records."""
    records = store.read(site_id=site_id, start=start, end=end)
    if not records:
        return None

    lags = np.array([r.arrival_lag_seconds for r in records], dtype=float)
    return LagSummary(
        n=len(lags),
        median_seconds=float(np.median(lags)),
        p95_seconds=float(np.percentile(lags, 95)),
        max_seconds=float(lags.max()),
    )


@dataclass(frozen=True)
class CompletenessSummary:
    """How much of what should have arrived did.

    Measured against the intervals the schedule says should exist, not against
    the rows present — counting only what arrived would make a totally silent
    site look perfectly complete.
    """

    expected: int
    received: int
    missing: int
    completeness: float
    largest_gap_intervals: int

    def as_dict(self) -> dict:
        return {
            "expected": self.expected,
            "received": self.received,
            "missing": self.missing,
            "completeness": round(self.completeness, 4),
            "largest_gap_intervals": self.largest_gap_intervals,
        }


def completeness(
    store: TelemetryStore,
    *,
    asset_id: str,
    channel: str,
    start: datetime,
    end: datetime,
    interval_minutes: int = 15,
) -> CompletenessSummary:
    """What fraction of an expected series actually arrived."""
    step = timedelta(minutes=interval_minutes)
    expected = int((end - start) / step) + 1

    records = store.read(asset_id=asset_id, channel=channel, start=start, end=end)
    received = len(records)

    largest_gap = 0
    if records:
        times = sorted(r.event_time for r in records)
        # A gap before the first record and after the last one both count: a
        # site that went silent at noon and never came back has a gap, even
        # though nothing follows it to bound one.
        boundaries = [start - step, *times, end + step]
        for earlier, later in zip(boundaries, boundaries[1:], strict=False):
            gap = int((later - earlier) / step) - 1
            largest_gap = max(largest_gap, gap)
    else:
        largest_gap = expected

    return CompletenessSummary(
        expected=expected,
        received=received,
        missing=max(0, expected - received),
        completeness=received / expected if expected else 0.0,
        largest_gap_intervals=largest_gap,
    )


@dataclass(frozen=True)
class AgentHealth:
    """What is known about one agent, from delivery alone."""

    agent_id: str
    site_id: str
    last_seen: datetime
    silent_for_seconds: float
    batches_received: int
    records_received: int
    records_duplicate: int
    buffered_remaining: int

    @property
    def is_silent(self) -> bool:
        """Nothing heard for more than four intervals."""
        return self.silent_for_seconds > 3600.0

    @property
    def is_behind(self) -> bool:
        """Reporting, but buffering rather than delivering live."""
        return self.buffered_remaining > 0

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "site_id": self.site_id,
            "last_seen": self.last_seen.isoformat(),
            "silent_for_seconds": round(self.silent_for_seconds, 1),
            "batches_received": self.batches_received,
            "records_received": self.records_received,
            "records_duplicate": self.records_duplicate,
            "buffered_remaining": self.buffered_remaining,
            "is_silent": self.is_silent,
            "is_behind": self.is_behind,
        }


def agent_health(store: TelemetryStore, agent_id: str, *, now: datetime) -> AgentHealth | None:
    """An agent's delivery health, or None if it has never been heard from."""
    state = store.agent_state(agent_id)
    if state is None:
        return None

    return AgentHealth(
        agent_id=state["agent_id"],
        site_id=state["site_id"],
        last_seen=state["last_seen"],
        silent_for_seconds=(now - state["last_seen"]).total_seconds(),
        batches_received=state["batches_received"],
        records_received=state["records_received"],
        records_duplicate=state["records_duplicate"],
        buffered_remaining=state["buffered_remaining"],
    )
