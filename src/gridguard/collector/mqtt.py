"""
MQTT: the publisher an agent uses and the subscriber the collector runs.

Why MQTT (architecture decision D1)
-----------------------------------
The workload is many intermittently-connected publishers sending small frequent
messages to one consumer, where the disconnections are the subject of study.
MQTT fits that natively and supplies one thing gRPC streaming would require
hand-rolling: a **last will and testament**. The broker publishes a message the
agent registered in advance whenever its connection drops without a clean
disconnect — a correct "this node is gone" signal that costs no polling and no
code on either side, for exactly the event this project exists to detect.

Kafka was rejected for the opposite reason: it is engineered to make delivery
loss invisible, which is the phenomenon under study, and its operational weight
would exceed the system it was carrying.

QoS 1, not 2
------------
At-least-once, with duplicates handled by the store's idempotent upsert.
Exactly-once (QoS 2) costs a four-way handshake per message to solve a problem
already solved one layer up, and its guarantee would still not survive an agent
restarting mid-flight. Duplicate-tolerance is a property the system needs
regardless, so it is built there and the transport is allowed to be simple.

Verification status
-------------------
**This module has not been run against a broker.** No Docker daemon is available
in the environment where it was written, so Mosquitto could not be started. It
conforms to the same :class:`~gridguard.edge.transport.Transport` interface the
agent was tested against, and the interface conformance *is* checked — but
"conforms to the interface" and "works against a real broker" are different
claims, and only the first is currently supported. The integration test that
would establish the second is written and skips without a broker.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from gridguard.edge.transport import TransportError

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1883

#: At-least-once. See the module docstring for why not QoS 2.
DEFAULT_QOS = 1

#: Everything, from every site and agent.
TELEMETRY_WILDCARD = "gridguard/v1/telemetry/+/+"
STATUS_WILDCARD = "gridguard/v1/status/+/+"


def _require_paho():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ImportError(
            'MQTT support needs paho-mqtt. Install with: pip install -e ".[collector]"'
        ) from exc
    return mqtt


@dataclass
class MqttTransport:
    """Publishes agent batches over MQTT.

    Satisfies :class:`~gridguard.edge.transport.Transport`, so an agent tested
    against the in-process fake can be handed this one unchanged.
    """

    site_id: str
    agent_id: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    qos: int = DEFAULT_QOS
    keepalive: int = 60

    _client: object | None = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Connect, registering a last will before doing so.

        The will is registered *before* connecting, which is the only time the
        broker accepts it. An agent that connects first and registers later has
        a window in which its death is silent — and that window is exactly when
        an unstable node is most likely to die.
        """
        mqtt = _require_paho()
        from gridguard.contract.codec import serialize, to_timestamp
        from gridguard.contract.v1 import telemetry_pb2 as pb
        from gridguard.edge.transport import status_topic

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=self.agent_id, clean_session=False
        )

        from datetime import UTC, datetime

        will = pb.NodeStatus(
            agent_id=self.agent_id,
            site_id=self.site_id,
            state=pb.NODE_STATE_OFFLINE_UNEXPECTED,
            detail="connection lost without a clean disconnect (broker last will)",
        )
        will.observed_at.CopyFrom(to_timestamp(datetime.now(UTC)))
        client.will_set(
            status_topic(self.site_id, self.agent_id),
            serialize(will),
            qos=self.qos,
            retain=True,
        )

        try:
            client.connect(self.host, self.port, self.keepalive)
        except OSError as exc:
            raise TransportError(
                f"could not reach broker at {self.host}:{self.port}: {exc}"
            ) from exc

        client.loop_start()
        self._client = client
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect cleanly, which suppresses the last will."""
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False

    def publish(self, topic: str, payload: bytes) -> None:
        """Publish, and wait for the broker to confirm.

        Blocking on the acknowledgement is the point. The agent acknowledges its
        buffer on the strength of this call returning, so returning before the
        broker has the message would turn a recoverable outage into data loss.
        """
        if self._client is None or not self._connected:
            raise TransportError("not connected")

        info = self._client.publish(topic, payload, qos=self.qos)
        try:
            info.wait_for_publish(timeout=30)
        except (RuntimeError, ValueError) as exc:
            raise TransportError(f"publish to {topic} failed: {exc}") from exc

        if not info.is_published():
            raise TransportError(f"publish to {topic} was not confirmed by the broker")


@dataclass
class MqttSubscriber:
    """Feeds arriving payloads to a handler.

    The handler is a plain callable taking bytes, so the collector stays
    transport-agnostic and the same code path serves a live subscription, a
    replayed capture file, and a test.
    """

    on_payload: Callable[[bytes], None]
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    qos: int = DEFAULT_QOS
    topics: tuple[str, ...] = (TELEMETRY_WILDCARD,)
    client_id: str = "gridguard-collector"

    _client: object | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        mqtt = _require_paho()

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id, clean_session=False
        )

        def _on_connect(client_, userdata, flags, reason_code, properties=None):
            if reason_code != 0:
                logger.error("Broker refused the collector's connection: %s", reason_code)
                return
            for topic in self.topics:
                client_.subscribe(topic, qos=self.qos)
                logger.info("Collector subscribed to %s", topic)

        def _on_message(client_, userdata, message):
            try:
                self.on_payload(message.payload)
            except Exception:  # noqa: BLE001 - one bad message must not kill the loop
                logger.exception("Handler raised on a message from %s", message.topic)

        client.on_connect = _on_connect
        client.on_message = _on_message

        try:
            client.connect(self.host, self.port, 60)
        except OSError as exc:
            raise TransportError(
                f"collector could not reach broker at {self.host}:{self.port}: {exc}"
            ) from exc

        client.loop_start()
        self._client = client

    def stop(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None
