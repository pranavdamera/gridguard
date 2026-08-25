"""
The logical clock.

Simulated time is generated, never read. Nothing in the simulation calls
``datetime.now()``, ``time.time()`` or ``pd.Timestamp.now()`` — the clock is the
sole authority on what time it is inside a run, and it produces the same
sequence of instants every time it is constructed with the same configuration.

Why this matters more than it looks
-----------------------------------
Wall-clock time enters a simulator in small ways and destroys reproducibility in
large ones: a timestamp captured at generation, a seed derived from the current
second, a timeout that fires differently under load. Once any of those exist, a
"deterministic" run is only deterministic on the machine that produced it. This
project has already lost a determinism guarantee once to a subtler version of
the same problem (a per-process hash salt), so the clock is a hard boundary
rather than a convention.

Speed is not time
-----------------
A demo wants to *watch* a day unfold; an experiment wants the same day in
milliseconds. Both must produce byte-identical output. So the clock separates
two things that are easy to conflate:

``event_time``   Simulated time. Advances by a fixed interval per tick. This is
                 the only time that reaches any value the simulation computes.
``speedup``      How fast the walk through simulated time is paced against the
                 wall clock, for a human watching. Affects *when* ticks are
                 handed out, never *what* they contain.

``speedup=None`` (the default) is unpaced: ticks are yielded as fast as the
consumer takes them, which is what experiments, tests and artifact builds want.
Any finite speedup sleeps between ticks for a demo. The equality of results
across speeds is asserted in tests, not assumed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

#: Simulated interval GridGuard models at, matching the canonical schema.
DEFAULT_INTERVAL_MINUTES = 15


@dataclass(frozen=True, slots=True)
class Tick:
    """One instant of simulated time."""

    #: Zero-based position in the run. The canonical way to index a tick — it is
    #: an integer, so it never suffers from timestamp comparison surprises.
    index: int

    #: Simulated instant, timezone-aware UTC.
    event_time: datetime

    #: Simulated seconds since the run started.
    elapsed_seconds: float

    @property
    def timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.event_time)


@dataclass(frozen=True)
class LogicalClock:
    """A reproducible sequence of simulated instants.

    Args:
        start: First instant. Must be timezone-aware; a naive datetime here is
            the class of bug ``event_time`` exists to prevent.
        periods: Number of ticks to emit.
        interval_minutes: Simulated minutes between ticks.
        speedup: Wall-clock pacing factor for demos. ``None`` (default) runs
            unpaced. ``60.0`` plays one simulated hour per wall-clock minute.
            Never affects the values a tick carries.
    """

    start: datetime
    periods: int
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES
    speedup: float | None = None

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            raise ValueError(
                "LogicalClock.start must be timezone-aware UTC. Simulated time is "
                "generated, not read from the host, so there is no local zone to assume."
            )
        if self.periods < 1:
            raise ValueError(f"periods must be >= 1, got {self.periods}")
        if self.interval_minutes < 1:
            raise ValueError(f"interval_minutes must be >= 1, got {self.interval_minutes}")
        if self.speedup is not None and self.speedup <= 0:
            raise ValueError(f"speedup must be positive or None, got {self.speedup}")

    @classmethod
    def from_range(
        cls,
        start: str | datetime,
        end: str | datetime,
        *,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        speedup: float | None = None,
    ) -> LogicalClock:
        """Build a clock spanning ``[start, end]`` inclusive.

        Naive inputs are read as UTC and stated to be so, rather than silently
        picking up the host's zone.
        """
        first = pd.Timestamp(start)
        last = pd.Timestamp(end)
        first = first.tz_localize(UTC) if first.tzinfo is None else first.tz_convert(UTC)
        last = last.tz_localize(UTC) if last.tzinfo is None else last.tz_convert(UTC)
        if last < first:
            raise ValueError(f"end {last} precedes start {first}")

        step = pd.Timedelta(minutes=interval_minutes)
        periods = int((last - first) / step) + 1
        return cls(
            start=first.to_pydatetime(),
            periods=periods,
            interval_minutes=interval_minutes,
            speedup=speedup,
        )

    # -- properties ---------------------------------------------------------

    @property
    def step(self) -> timedelta:
        return timedelta(minutes=self.interval_minutes)

    @property
    def end(self) -> datetime:
        return self.start + self.step * (self.periods - 1)

    @property
    def simulated_duration(self) -> timedelta:
        return self.step * (self.periods - 1)

    def at(self, index: int) -> Tick:
        """The tick at ``index``, without iterating to it."""
        if not 0 <= index < self.periods:
            raise IndexError(f"tick {index} outside [0, {self.periods})")
        return Tick(
            index=index,
            event_time=self.start + self.step * index,
            elapsed_seconds=self.step.total_seconds() * index,
        )

    def index(self) -> pd.DatetimeIndex:
        """Every instant as a pandas index, for vectorised layers.

        The layers compute over whole runs rather than tick by tick — that is
        what makes them fast enough to build artifacts with — so this is the
        form they actually consume. :meth:`__iter__` exists for the streaming
        consumers that arrive in later phases.
        """
        return pd.date_range(
            self.start, periods=self.periods, freq=f"{self.interval_minutes}min", tz=UTC
        )

    # -- iteration ----------------------------------------------------------

    def __len__(self) -> int:
        return self.periods

    def __iter__(self) -> Iterator[Tick]:
        """Yield every tick, pacing against the wall clock only if asked to.

        Pacing measures from a monotonic reference rather than accumulating
        sleeps, so a slow consumer does not compound drift into the schedule.
        The values yielded are identical either way.
        """
        if self.speedup is None:
            for i in range(self.periods):
                yield self.at(i)
            return

        wall_per_tick = self.step.total_seconds() / self.speedup
        origin = time.monotonic()
        for i in range(self.periods):
            due = origin + wall_per_tick * i
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            yield self.at(i)

    def replace(self, **changes) -> LogicalClock:
        """A copy with fields overridden — chiefly to change ``speedup``."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)
