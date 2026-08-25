"""
The telemetry store.

Where delivered records land, and the one place in the system that has to be
correct about *arriving twice*. An agent recovering from an outage resends
records the collector may already hold; the store's job is to make that a no-op
rather than a double count.

Idempotency
-----------
Every write is an upsert keyed on ``(asset_id, channel, event_time)`` — the
identity of a measurement, independent of how many times it was sent or in what
order it arrived. Applying the same batch twice therefore leaves the table in
exactly the state one application would, which is the gate this phase has to
pass.

``sequence`` is stored but is deliberately **not** part of the key. It
identifies a record within an agent's stream, and an agent replaced or rebuilt
mid-deployment starts a new stream; the measurement it reports is still the same
measurement. Keying on the physical identity rather than the transport's
bookkeeping is what makes the store robust to the agent being redeployed.

One dialect, two backends
-------------------------
The SQL here runs unmodified on both SQLite and PostgreSQL, and the same test
suite runs against both — SQLite always, Postgres when one is reachable. That is
not a compromise between them. The alternative, a Postgres-only store, would be
verifiable only where a database container can run, which is neither this
development environment nor a fast CI job; the logic would then be exercised
nowhere and the first real test would be in the demo.

Two choices make the shared dialect honest rather than lowest-common-denominator:

* ``INSERT ... ON CONFLICT (...) DO UPDATE SET ... excluded.x`` is supported
  identically by SQLite 3.24+ and Postgres 9.5+. It is not emulated on either.
* Times are stored as **integer microseconds since the Unix epoch, UTC**.
  Postgres ``TIMESTAMPTZ`` would be marginally nicer to query by hand, but the
  two backends differ in how they parse and return it, and this project has
  already lost time to timestamp ambiguity. An integer is exact, sortable,
  indexable, and cannot carry an implicit zone.

Postgres remains the deployed choice (architecture decision D2): the collector
and the API are separate containers, and SQLite across a shared volume loses
writes. SQLite here is the test backend and a single-process fallback, not the
recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from gridguard.domain.telemetry import QualityFlag, TelemetryRecord

logger = logging.getLogger(__name__)

#: Microseconds per second, for the epoch conversion.
_MICROS = 1_000_000


def to_epoch_micros(value: datetime) -> int:
    """UTC datetime → integer microseconds since the epoch."""
    if value.tzinfo is None:
        raise ValueError(
            "Refusing to store a naive datetime. Every time in this system is UTC by "
            "contract; a naive value means an offset was lost upstream."
        )
    return int(value.astimezone(UTC).timestamp() * _MICROS)


def from_epoch_micros(value: int) -> datetime:
    """Integer microseconds since the epoch → timezone-aware UTC datetime."""
    return datetime.fromtimestamp(value / _MICROS, tz=UTC)


SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    asset_id     TEXT   NOT NULL,
    channel      TEXT   NOT NULL,
    event_time   BIGINT NOT NULL,
    site_id      TEXT   NOT NULL,
    unit         TEXT   NOT NULL,
    value        DOUBLE PRECISION,
    ingest_time  BIGINT NOT NULL,
    sequence     BIGINT,
    quality      INTEGER NOT NULL DEFAULT 0,
    agent_id     TEXT,
    is_replay    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (asset_id, channel, event_time)
);

CREATE INDEX IF NOT EXISTS telemetry_site_time ON telemetry (site_id, event_time);
CREATE INDEX IF NOT EXISTS telemetry_event_time ON telemetry (event_time);

CREATE TABLE IF NOT EXISTS agent_state (
    agent_id           TEXT   NOT NULL,
    site_id            TEXT   NOT NULL,
    last_seen          BIGINT NOT NULL,
    last_event_time    BIGINT,
    batches_received   BIGINT NOT NULL DEFAULT 0,
    records_received   BIGINT NOT NULL DEFAULT 0,
    records_duplicate  BIGINT NOT NULL DEFAULT 0,
    buffered_remaining BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id)
);
"""

#: The upsert. Runs unmodified on SQLite and Postgres.
#:
#: ``ingest_time`` is deliberately *not* overwritten on conflict: the first
#: arrival is when the collector actually learned the value, and a replay
#: arriving later must not rewrite history to make the original delivery look
#: slower than it was. Everything else is refreshed, because a re-send may carry
#: corrected quality flags.
_UPSERT = """
INSERT INTO telemetry (
    asset_id, channel, event_time, site_id, unit, value,
    ingest_time, sequence, quality, agent_id, is_replay
) VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
ON CONFLICT (asset_id, channel, event_time) DO UPDATE SET
    value    = excluded.value,
    unit     = excluded.unit,
    sequence = excluded.sequence,
    quality  = excluded.quality,
    agent_id = excluded.agent_id
"""


@dataclass(frozen=True)
class StoredRecord:
    """One measurement as the store holds it."""

    asset_id: str
    channel: str
    site_id: str
    unit: str
    value: float | None
    event_time: datetime
    ingest_time: datetime
    sequence: int | None
    quality: QualityFlag
    agent_id: str | None
    is_replay: bool

    @property
    def arrival_lag_seconds(self) -> float:
        """How long this measurement took to reach the collector."""
        return (self.ingest_time - self.event_time).total_seconds()


@dataclass(frozen=True)
class WriteResult:
    """What a write actually did.

    ``inserted`` and ``updated`` are counted separately because a replay that
    reports as an insert would mean the idempotency key is wrong — and that is a
    silent correctness failure, not a performance detail.
    """

    inserted: int = 0
    updated: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated


class Connection(Protocol):
    """The little of DB-API the store needs, shared by sqlite3 and psycopg."""

    def cursor(self): ...
    def commit(self): ...
    def close(self): ...


class TelemetryStore:
    """Idempotent storage for delivered telemetry.

    Args:
        connection: an open DB-API connection.
        paramstyle: ``"?"`` for SQLite, ``"%s"`` for psycopg. The only thing
            that differs between the two backends, and it is a placeholder
            character rather than a difference in behaviour.
    """

    def __init__(self, connection: Connection, *, paramstyle: str = "?") -> None:
        self._connection = connection
        self._p = paramstyle
        self._create_schema()

    # -- lifecycle ----------------------------------------------------------

    def _sql(self, template: str) -> str:
        return template.format(p=self._p)

    def _create_schema(self) -> None:
        cursor = self._connection.cursor()
        for statement in SCHEMA.split(";"):
            if statement.strip():
                cursor.execute(statement)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ------------------------------------------------------------

    def write(
        self,
        records: list[TelemetryRecord],
        *,
        ingest_time: datetime | None = None,
        agent_id: str | None = None,
        is_replay: bool = False,
    ) -> WriteResult:
        """Upsert records. Applying the same batch twice is a no-op.

        ``ingest_time`` is stamped by the collector, not taken from the record:
        an edge node's clock is not trusted to say when someone else received
        something. A record that already carries one keeps it, which is what
        lets a test replay a historical batch with its original arrival times.
        """
        if not records:
            return WriteResult()

        stamped = ingest_time or datetime.now(UTC)
        cursor = self._connection.cursor()

        existing = self._existing_keys(cursor, records)
        inserted = updated = 0

        for record in records:
            key = (record.asset_id, record.channel, to_epoch_micros(record.event_time))
            if key in existing:
                updated += 1
            else:
                inserted += 1
                existing.add(key)

            cursor.execute(
                self._sql(_UPSERT),
                (
                    record.asset_id,
                    record.channel,
                    key[2],
                    record.site_id,
                    record.unit,
                    record.value,
                    to_epoch_micros(record.ingest_time or stamped),
                    record.sequence,
                    int(record.quality),
                    agent_id,
                    1 if is_replay else 0,
                ),
            )

        self._connection.commit()
        return WriteResult(inserted=inserted, updated=updated)

    def _existing_keys(self, cursor, records: list[TelemetryRecord]) -> set[tuple]:
        """Which of these measurements the store already holds.

        Looked up before writing so insert and update can be reported
        separately. The upsert itself would happily do the work without this,
        but then a replay and a first delivery would be indistinguishable, and
        that distinction is the whole gate for this phase.
        """
        found: set[tuple] = set()
        for record in records:
            cursor.execute(
                self._sql(
                    "SELECT 1 FROM telemetry WHERE asset_id = {p} AND channel = {p} "
                    "AND event_time = {p}"
                ),
                (record.asset_id, record.channel, to_epoch_micros(record.event_time)),
            )
            if cursor.fetchone() is not None:
                found.add((record.asset_id, record.channel, to_epoch_micros(record.event_time)))
        return found

    # -- reading ------------------------------------------------------------

    def read(
        self,
        *,
        site_id: str | None = None,
        asset_id: str | None = None,
        channel: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[StoredRecord]:
        """Records matching the filters, ordered by event time."""
        clauses: list[str] = []
        params: list = []

        for column, value in (
            ("site_id", site_id),
            ("asset_id", asset_id),
            ("channel", channel),
        ):
            if value is not None:
                clauses.append(f"{column} = {self._p}")
                params.append(value)

        if start is not None:
            clauses.append(f"event_time >= {self._p}")
            params.append(to_epoch_micros(start))
        if end is not None:
            clauses.append(f"event_time <= {self._p}")
            params.append(to_epoch_micros(end))

        sql = (
            "SELECT asset_id, channel, site_id, unit, value, event_time, ingest_time, "
            "sequence, quality, agent_id, is_replay FROM telemetry"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY event_time, asset_id, channel"
        if limit is not None:
            sql += f" LIMIT {self._p}"
            params.append(limit)

        cursor = self._connection.cursor()
        cursor.execute(sql, tuple(params))

        return [
            StoredRecord(
                asset_id=row[0],
                channel=row[1],
                site_id=row[2],
                unit=row[3],
                value=row[4],
                event_time=from_epoch_micros(row[5]),
                ingest_time=from_epoch_micros(row[6]),
                sequence=row[7],
                quality=QualityFlag(row[8] or 0),
                agent_id=row[9],
                is_replay=bool(row[10]),
            )
            for row in cursor.fetchall()
        ]

    def count(self) -> int:
        cursor = self._connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM telemetry")
        return int(cursor.fetchone()[0])

    def fingerprint(self) -> str:
        """A digest of the whole table's contents.

        The gate for this phase compares table state before and after a replay,
        and comparing a digest says "identical" or "not" without a diff of
        thousands of rows. ``ingest_time`` is excluded: a replay legitimately
        arrives later, and the point of the check is that the *measurements* are
        unchanged, not that time stood still.
        """
        from hashlib import blake2b

        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT asset_id, channel, event_time, site_id, unit, value, sequence, quality "
            "FROM telemetry ORDER BY asset_id, channel, event_time"
        )
        digest = blake2b(digest_size=16)
        for row in cursor.fetchall():
            digest.update(repr(tuple(row)).encode())
        return digest.hexdigest()

    # -- agent state --------------------------------------------------------

    def record_agent_batch(
        self,
        *,
        agent_id: str,
        site_id: str,
        seen_at: datetime,
        last_event_time: datetime | None,
        records: int,
        duplicates: int,
        buffered_remaining: int,
    ) -> None:
        """Update what is known about an agent from a batch it sent."""
        cursor = self._connection.cursor()
        cursor.execute(
            self._sql("""
                INSERT INTO agent_state (
                    agent_id, site_id, last_seen, last_event_time,
                    batches_received, records_received, records_duplicate,
                    buffered_remaining
                ) VALUES ({p}, {p}, {p}, {p}, 1, {p}, {p}, {p})
                ON CONFLICT (agent_id) DO UPDATE SET
                    site_id            = excluded.site_id,
                    last_seen          = excluded.last_seen,
                    last_event_time    = excluded.last_event_time,
                    batches_received   = agent_state.batches_received + 1,
                    records_received   = agent_state.records_received + excluded.records_received,
                    records_duplicate  = agent_state.records_duplicate
                                         + excluded.records_duplicate,
                    buffered_remaining = excluded.buffered_remaining
                """),
            (
                agent_id,
                site_id,
                to_epoch_micros(seen_at),
                to_epoch_micros(last_event_time) if last_event_time else None,
                records,
                duplicates,
                buffered_remaining,
            ),
        )
        self._connection.commit()

    def agent_state(self, agent_id: str) -> dict | None:
        cursor = self._connection.cursor()
        cursor.execute(
            self._sql(
                "SELECT agent_id, site_id, last_seen, last_event_time, batches_received, "
                "records_received, records_duplicate, buffered_remaining "
                "FROM agent_state WHERE agent_id = {p}"
            ),
            (agent_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "agent_id": row[0],
            "site_id": row[1],
            "last_seen": from_epoch_micros(row[2]),
            "last_event_time": from_epoch_micros(row[3]) if row[3] is not None else None,
            "batches_received": row[4],
            "records_received": row[5],
            "records_duplicate": row[6],
            "buffered_remaining": row[7],
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def sqlite_store(path: str = ":memory:") -> TelemetryStore:
    """A store backed by SQLite. The test backend, and a single-process option."""
    import sqlite3

    connection = sqlite3.connect(path)
    return TelemetryStore(connection, paramstyle="?")


def postgres_store(dsn: str) -> TelemetryStore:
    """A store backed by PostgreSQL. The deployed backend (decision D2)."""
    import psycopg

    connection = psycopg.connect(dsn)
    return TelemetryStore(connection, paramstyle="%s")
