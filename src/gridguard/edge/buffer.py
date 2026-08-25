"""
The edge agent's store-and-forward buffer.

A node that cannot reach the collector must not lose what it measured. The
buffer is what makes that true across a process restart as well as a network
outage, which is why it is on disk rather than in memory: an agent killed
mid-outage and restarted has to come back holding everything it had.

SQLite, and why that is not the objection raised earlier
--------------------------------------------------------
The architecture notes reject SQLite for the *collector's* store, because the
collector and the API are separate containers and SQLite across a shared volume
loses writes. None of that applies here. An edge buffer has exactly one writer,
in one process, on local disk — the case SQLite in WAL mode is built for. It is
also in the standard library, so the agent gains durable buffering without
gaining a dependency.

What is stored
--------------
The **serialized wire message**, not a decomposition of it. Buffering therefore
cannot corrupt a record: what comes out is byte-identical to what went in, and
an agent can buffer a message carrying fields its own build does not understand.
``asset_id`` and ``sequence`` are stored alongside as indexed columns purely so
the buffer can order and deduplicate without parsing anything.

Sequence numbers
----------------
Assigned here, once, at ingest — and persisted. Two consequences that the tests
pin down:

* A record keeps its sequence across a restart, so a replay after a crash does
  not renumber anything the collector may already hold.
* Sequences never restart from zero for a stream that has reported before, so
  the idempotency key stays usable for the life of the deployment rather than
  only for the life of the process.

The key is ``(asset_id, channel, sequence)``, not ``(asset_id, sequence)``.
That distinction was found by an end-to-end test rather than by reasoning: one
asset emits several channels per interval, so an asset-scoped key silently
rejected three of every four records as duplicates. A sequence numbers a
*stream*, and the stream is the channel — which is also the thing that fails
independently, so the two agree.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT    NOT NULL,
    channel     TEXT    NOT NULL,
    sequence    INTEGER NOT NULL,
    event_time  TEXT    NOT NULL,
    payload     BLOB    NOT NULL,
    acked       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (asset_id, channel, sequence)
);

CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (acked, id);

CREATE TABLE IF NOT EXISTS asset_sequence (
    asset_id      TEXT    NOT NULL,
    channel       TEXT    NOT NULL,
    next_sequence INTEGER NOT NULL,
    PRIMARY KEY (asset_id, channel)
);
"""


@dataclass(frozen=True)
class BufferedRecord:
    """One record as the buffer holds it."""

    id: int
    asset_id: str
    channel: str
    sequence: int
    event_time: str
    payload: bytes


class OutboxBuffer:
    """Durable store-and-forward queue for one agent.

    ``synchronous=FULL`` is deliberate and costs write throughput. The buffer
    exists precisely for the case where the process dies unexpectedly; a
    faster setting would let the operating system hold recent writes in a cache
    that a crash discards, which would lose exactly the records the buffer was
    added to protect.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> OutboxBuffer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sequences ----------------------------------------------------------

    def next_sequence(self, asset_id: str, channel: str) -> int:
        """Reserve the next sequence number for an asset.

        Persisted before it is handed out, so a crash between reserving and
        appending burns a number rather than reusing one. A gap in a sequence is
        harmless — the collector deduplicates on the key and never assumes
        contiguity — whereas a reused number would silently overwrite a
        different reading.
        """
        row = self._connection.execute(
            "SELECT next_sequence FROM asset_sequence WHERE asset_id = ? AND channel = ?",
            (asset_id, channel),
        ).fetchone()
        current = int(row["next_sequence"]) if row else 0

        self._connection.execute(
            "INSERT INTO asset_sequence (asset_id, channel, next_sequence) VALUES (?, ?, ?) "
            "ON CONFLICT(asset_id, channel) DO UPDATE SET next_sequence = excluded.next_sequence",
            (asset_id, channel, current + 1),
        )
        return current

    def peek_sequence(self, asset_id: str, channel: str) -> int:
        """The next sequence a stream would be given, without reserving it."""
        row = self._connection.execute(
            "SELECT next_sequence FROM asset_sequence WHERE asset_id = ? AND channel = ?",
            (asset_id, channel),
        ).fetchone()
        return int(row["next_sequence"]) if row else 0

    # -- writing ------------------------------------------------------------

    def append(
        self, *, asset_id: str, channel: str, sequence: int, event_time: str, payload: bytes
    ) -> bool:
        """Add a record to the outbox.

        Returns False when ``(asset_id, channel, sequence)`` is already present,
        which makes re-ingesting the same record harmless. The agent can
        therefore be restarted mid-batch without needing to know how far it had
        got.
        """
        try:
            self._connection.execute(
                "INSERT INTO outbox (asset_id, channel, sequence, event_time, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (asset_id, channel, sequence, event_time, payload),
            )
            return True
        except sqlite3.IntegrityError:
            logger.debug("Record %s/%s/%d already buffered; ignoring.", asset_id, channel, sequence)
            return False

    # -- reading ------------------------------------------------------------

    def pending(self, limit: int | None = None) -> list[BufferedRecord]:
        """Unacknowledged records, oldest first."""
        sql = (
            "SELECT id, asset_id, channel, sequence, event_time, payload "
            "FROM outbox WHERE acked = 0 ORDER BY id"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [
            BufferedRecord(
                id=int(r["id"]),
                asset_id=str(r["asset_id"]),
                channel=str(r["channel"]),
                sequence=int(r["sequence"]),
                event_time=str(r["event_time"]),
                payload=bytes(r["payload"]),
            )
            for r in self._connection.execute(sql, params)
        ]

    def pending_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM outbox WHERE acked = 0"
        ).fetchone()
        return int(row["n"])

    def total_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS n FROM outbox").fetchone()
        return int(row["n"])

    # -- acknowledging ------------------------------------------------------

    def acknowledge(self, ids: list[int]) -> int:
        """Mark records delivered.

        Called only after the transport has confirmed receipt. Acknowledging
        optimistically — before confirmation — is the mistake that turns a
        transport failure into permanent data loss, so the agent does the
        publish and the acknowledgement as two separate steps.
        """
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        cursor = self._connection.execute(
            f"UPDATE outbox SET acked = 1 WHERE id IN ({placeholders}) AND acked = 0", ids
        )
        return int(cursor.rowcount)

    def last_delivered_sequence(self, asset_id: str, channel: str) -> int | None:
        row = self._connection.execute(
            "SELECT MAX(sequence) AS s FROM outbox "
            "WHERE asset_id = ? AND channel = ? AND acked = 1",
            (asset_id, channel),
        ).fetchone()
        return int(row["s"]) if row and row["s"] is not None else None

    # -- housekeeping -------------------------------------------------------

    def prune_acknowledged(self, keep_last: int = 0) -> int:
        """Drop delivered records, optionally retaining the most recent few.

        An edge node has finite disk. Delivered records are the safe thing to
        drop; pending ones never are, so this deliberately cannot touch them.
        """
        cursor = self._connection.execute(
            "DELETE FROM outbox WHERE acked = 1 AND id NOT IN "
            "(SELECT id FROM outbox WHERE acked = 1 ORDER BY id DESC LIMIT ?)",
            (keep_last,),
        )
        return int(cursor.rowcount)
