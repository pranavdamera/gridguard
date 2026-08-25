#!/usr/bin/env python3
"""
Run the layered site simulator.

    python experiments/run_simulation.py --days 7
    python experiments/run_simulation.py --site nist_roof --site gmu_fairfax --seed 3
    python experiments/run_simulation.py --days 1 --speedup 2000 --replay  # paced demo
    python experiments/run_simulation.py --days 30 --out artifacts/reports/sim

Writes three views of the run, kept separate on purpose:

    truth.parquet      what physically happened, per asset
    observed.parquet   what the instruments reported
    canonical.parquet  what reached the collector, in the shipped schema
    run.json           config, seed, fingerprint, and throughput

``--speedup`` paces a *replay* of the finished run against the wall clock, for
watching a demo unfold. The layers are vectorised, so computing the run is
already fast — there is nothing to pace during computation, and slowing it would
only make experiments slower. Pacing therefore applies to ``--replay``, after
the values exist, which is what makes "pacing cannot change results" true by
construction rather than by discipline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from hashlib import blake2b
from pathlib import Path

import pandas as pd

from gridguard.config import settings
from gridguard.simulation import (
    DEFAULT_SEED,
    DEFAULT_START,
    FleetSimulator,
    SimulationConfig,
)
from gridguard.sites.registry import list_sites

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-6s %(message)s")
logger = logging.getLogger("simulation")


def _digest(frame: pd.DataFrame) -> str:
    """A stable digest of a frame's contents, for eyeballing reproducibility."""
    return blake2b(frame.to_csv(index=False).encode(), digest_size=8).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site id; repeatable. Default: every synthetic site in the registry.",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--days", type=int, default=7, help="Simulated days from --start.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument(
        "--speedup",
        type=float,
        default=None,
        help="Wall-clock pacing for --replay (e.g. 2000 = ~2000x real time). "
        "Omit to replay unpaced. Never changes the values produced.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="After computing, replay the run tick by tick at --speedup. "
        "For demos; experiments do not need it.",
    )
    parser.add_argument("--out", type=Path, default=Path(settings.report_dir) / "simulation")
    parser.add_argument("--no-write", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    site_ids = (
        tuple(args.sites)
        if args.sites
        else tuple(s.site_id for s in list_sites(data_mode="synthetic"))
    )
    end = (
        pd.Timestamp(args.start)
        + pd.Timedelta(days=args.days)
        - pd.Timedelta(minutes=args.interval_minutes)
    ).strftime("%Y-%m-%d %H:%M")

    config = SimulationConfig(
        site_ids=site_ids,
        start=args.start,
        end=end,
        interval_minutes=args.interval_minutes,
        seed=args.seed,
        speedup=args.speedup,
    )

    started = time.monotonic()
    result = FleetSimulator(config).run()
    truth = result.truth()
    observed = result.observed()
    canonical = result.canonical()
    wall_seconds = time.monotonic() - started

    ticks = len(result.clock)
    simulated_days = result.clock.simulated_duration.total_seconds() / 86400
    acceleration = (simulated_days * 86400 / wall_seconds) if wall_seconds > 0 else float("inf")

    summary = {
        "fingerprint": result.fingerprint,
        "seed": config.seed,
        "sites": list(site_ids),
        "start": config.start,
        "end": config.end,
        "interval_minutes": config.interval_minutes,
        "ticks": ticks,
        "simulated_days": round(simulated_days, 3),
        "speedup_requested": args.speedup,
        "wall_seconds": round(wall_seconds, 3),
        "acceleration_vs_real_time": round(acceleration, 1),
        "rows": {"truth": len(truth), "observed": len(observed), "canonical": len(canonical)},
        "digests": {
            "truth": _digest(truth),
            "observed": _digest(observed),
            "canonical": _digest(canonical),
        },
    }

    logger.info(
        "%d site(s) x %d ticks = %d canonical rows in %.2fs (%.0fx real time)",
        len(site_ids),
        ticks,
        len(canonical),
        wall_seconds,
        acceleration,
    )
    logger.info(
        "config fingerprint %s | canonical digest %s",
        result.fingerprint,
        summary["digests"]["canonical"],
    )

    print("\nObservation error — how far the instruments sit from the truth:")
    print(result.observation_error().to_string(index=False))

    if args.replay:
        logger.info(
            "Replaying %d ticks at speedup=%s ...",
            ticks,
            args.speedup if args.speedup else "unpaced",
        )
        replay_started = time.monotonic()
        replayed_rows = 0
        for tick, chunk in result.replay():
            replayed_rows += len(chunk)
            if tick.index % max(1, ticks // 8) == 0:
                total_kw = float(chunk["ac_power_kw"].sum()) if len(chunk) else 0.0
                logger.info(
                    "  t=%-5d %s  fleet %7.1f kW",
                    tick.index,
                    tick.event_time.strftime("%Y-%m-%d %H:%M"),
                    total_kw,
                )
        replay_seconds = time.monotonic() - replay_started
        observed_rate = (
            result.clock.simulated_duration.total_seconds() / replay_seconds
            if replay_seconds > 0
            else float("inf")
        )
        logger.info(
            "Replayed %d rows in %.2fs wall (%.0fx real time)",
            replayed_rows,
            replay_seconds,
            observed_rate,
        )
        summary["replay"] = {
            "wall_seconds": round(replay_seconds, 3),
            "observed_rate_vs_real_time": round(observed_rate, 1),
            "rows": replayed_rows,
        }
        # The values were fixed before pacing began; this proves it.
        assert _digest(result.canonical()) == summary["digests"]["canonical"]

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        truth.to_parquet(args.out / "truth.parquet", index=False)
        observed.to_parquet(args.out / "observed.parquet", index=False)
        canonical.to_parquet(args.out / "canonical.parquet", index=False)
        (args.out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("Wrote truth, observed, canonical and run.json -> %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
