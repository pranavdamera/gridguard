"""
How an agent publishes, and how that is made to fail on purpose.

Phase 5 defines the seam and ships an in-process implementation. The MQTT
client arrives in phase 6 alongside the broker and the collector.

That split is not a shortcut. The properties this phase has to establish — no
sample lost, none duplicated, a buffer that survives a restart — are properties
of the *agent's* behaviour when publishing fails. Testing them needs a transport
that fails exactly when told to, at a chosen record, deterministically. A real
broker makes those cases hard to reach and impossible to reproduce; a fake makes
them one line each. The real client then has to satisfy the same interface.

``Transport`` is deliberately small. Everything an agent needs from a network is
"try to send these bytes, tell me truthfully whether they arrived" — and the
truthfulness is the whole contract. A transport that reports success on a send
that did not land converts a recoverable outage into permanent data loss,
because the agent will acknowledge and prune on the strength of that answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class TransportError(RuntimeError):
    """Publishing failed. The caller must keep the records buffered."""


@runtime_checkable
class Transport(Protocol):
    """What an agent needs from a network."""

    @property
    def connected(self) -> bool:
        """Whether the transport currently believes it can deliver."""
        ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def publish(self, topic: str, payload: bytes) -> None:
        """Deliver ``payload``, or raise :class:`TransportError`.

        Must not return normally unless the payload was accepted for delivery.
        """
        ...


@dataclass
class InMemoryTransport:
    """A transport that can be broken on demand.

    Not a mock in the usual sense: it really delivers, in order, and really
    refuses when told to. Delivered payloads are retained so a test can assert
    on exactly what crossed the wire — including that a replayed record is
    byte-identical to its first attempt.
    """

    #: Every payload accepted for delivery, in order, as ``(topic, payload)``.
    delivered: list[tuple[str, bytes]] = field(default_factory=list)

    #: When False, every publish raises. Stands in for a broker being down, a
    #: link being cut, or a node being isolated.
    online: bool = True

    #: Fail the next N publishes, then recover. For testing partial outages
    #: without having to toggle ``online`` around a call.
    fail_next: int = 0

    #: Raise after this many *successful* publishes. Cuts the link mid-batch,
    #: which is where off-by-one acknowledgement bugs live.
    fail_after: int | None = None

    _successes: int = 0

    @property
    def connected(self) -> bool:
        return self.online

    def connect(self) -> None:
        self.online = True

    def disconnect(self) -> None:
        self.online = False

    def publish(self, topic: str, payload: bytes) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise TransportError("scheduled failure (fail_next)")
        if not self.online:
            raise TransportError("transport is offline")
        if self.fail_after is not None and self._successes >= self.fail_after:
            raise TransportError(f"scheduled failure after {self.fail_after} successes")

        self.delivered.append((topic, payload))
        self._successes += 1

    # -- inspection ---------------------------------------------------------

    def payloads(self) -> list[bytes]:
        return [payload for _, payload in self.delivered]

    def reset_failures(self) -> None:
        self.fail_next = 0
        self.fail_after = None
        self._successes = 0
        self.online = True


def telemetry_topic(site_id: str, agent_id: str) -> str:
    """Topic a batch is published on.

    Hierarchical and site-first, which is the shape MQTT subscriptions want: a
    collector takes ``gridguard/v1/telemetry/+/+`` for everything, or narrows to
    one site without the broker having to filter payloads it cannot read.
    """
    return f"gridguard/v1/telemetry/{site_id}/{agent_id}"


def status_topic(site_id: str, agent_id: str) -> str:
    """Topic an agent's liveness is published on.

    Separate from telemetry because the two fail independently, and because
    this is the topic an MQTT last-will message is registered against: the
    broker publishes it when the agent's connection drops without a clean
    disconnect, which is a correct "this node is gone" signal that costs no
    polling and no code on either side.
    """
    return f"gridguard/v1/status/{site_id}/{agent_id}"
