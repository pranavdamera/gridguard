#!/usr/bin/env python3
"""
Run the collector: subscribe to the broker and write what arrives.

    python scripts/run_collector.py
    python scripts/run_collector.py --mqtt-host localhost --dsn postgresql://...

Reads configuration from the environment so the compose service needs no
arguments:

    GRIDGUARD_MQTT_HOST      broker hostname (default: localhost)
    GRIDGUARD_MQTT_PORT      broker port (default: 1883)
    GRIDGUARD_STORE_DSN      Postgres DSN. Falls back to a local SQLite file,
                             which is fine for one process and wrong for two —
                             see decision D2 in the architecture notes.

Requires the collector extra: pip install -e ".[collector]"
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from datetime import UTC, datetime

from gridguard.collector import Collector, TelemetryStore, sqlite_store
from gridguard.collector.mqtt import MqttSubscriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-6s %(message)s")
logger = logging.getLogger("collector")


def open_store(dsn: str | None) -> TelemetryStore:
    if dsn:
        from gridguard.collector import postgres_store

        logger.info("Store: PostgreSQL")
        return postgres_store(dsn)

    path = os.environ.get("GRIDGUARD_STORE_PATH", "data/collector.sqlite")
    logger.warning(
        "Store: SQLite at %s. Fine for a single process; the deployed collector "
        "and API are separate containers, where SQLite over a shared volume loses "
        "writes. Set GRIDGUARD_STORE_DSN for Postgres.",
        path,
    )
    return sqlite_store(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mqtt-host", default=os.environ.get("GRIDGUARD_MQTT_HOST", "localhost"))
    parser.add_argument(
        "--mqtt-port", type=int, default=int(os.environ.get("GRIDGUARD_MQTT_PORT", "1883"))
    )
    parser.add_argument("--dsn", default=os.environ.get("GRIDGUARD_STORE_DSN"))
    parser.add_argument(
        "--report-seconds",
        type=float,
        default=30.0,
        help="How often to log a delivery summary.",
    )
    args = parser.parse_args()

    store = open_store(args.dsn)
    collector = Collector(store)

    def handle(payload: bytes) -> None:
        collector.handle_payload(payload, received_at=datetime.now(UTC))

    subscriber = MqttSubscriber(on_payload=handle, host=args.mqtt_host, port=args.mqtt_port)
    subscriber.start()
    logger.info("Collector listening on %s:%d", args.mqtt_host, args.mqtt_port)

    stopping = threading.Event()

    def _stop(signum, frame):
        logger.info("Signal %s — shutting down.", signum)
        stopping.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not stopping.wait(args.report_seconds):
        logger.info(
            "batches=%(batches)d records=%(records)d inserted=%(inserted)d "
            "duplicates=%(duplicates)d replays=%(replays)d decode_failures=%(decode_failures)d",
            collector.stats.as_dict(),
        )

    subscriber.stop()
    store.close()
    logger.info("Final: %s", collector.stats.as_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
