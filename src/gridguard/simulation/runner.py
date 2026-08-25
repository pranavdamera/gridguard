"""
The simulator.

Wires the four layers for each site and returns both what happened and what was
observed, kept apart.

    environment  →  equipment  →  sensing  →  transport
    (truth)         (truth)       (observed)  (delivered)

The boundary between layers 2 and 3 is the one that matters. Above it, values
are what physically occurred. Below it, they are claims. A later experiment
asking "did the detector respond to a real outage or to a broken sensor?" is
answerable only because the two are never merged into one frame.

Output shape
------------
:class:`SimulationResult` carries three views of the same run:

``truth``      Per interval and asset: real weather, real generation, real
               equipment state. Never fed to a model — this is the answer key.
``observed``   What the instruments reported. What a model would see if the
               network were perfect.
``delivered``  What actually reached the collector, with ``ingest_time`` and
               ``sequence``. What a model actually sees.

``observed`` and ``delivered`` are identical in phase 2, because transport is a
pass-through. Keeping them as separate views now means the day they diverge is a
change in data, not in anyone's code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from gridguard.data.schema import validate_canonical
from gridguard.domain.asset import AssetKind
from gridguard.domain.telemetry import event_time_to_local
from gridguard.simulation.clock import LogicalClock
from gridguard.simulation.config import SimulationConfig
from gridguard.simulation.environment import EnvironmentLayer, EnvironmentTruth
from gridguard.simulation.equipment import EquipmentLayer, EquipmentTruth
from gridguard.simulation.faults import FaultSchedule, build_layered_scenario
from gridguard.simulation.sensing import Observation, SensorLayer
from gridguard.simulation.transport import Delivery, TransportLayer
from gridguard.sites.registry import assets_for_site, get_site

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteSimulation:
    """Every layer's output for one site, kept separate."""

    site_id: str
    environment: EnvironmentTruth
    equipment: EquipmentTruth
    observation: Observation
    delivery: Delivery
    schedule: FaultSchedule = field(default_factory=FaultSchedule)

    def truth_frame(self) -> pd.DataFrame:
        """Ground truth: weather, per-asset generation, and every active fault.

        Carries all three fault columns — equipment, sensor and communication —
        because the whole point of the layering is that they are different
        things. An attribution engine is scored against this frame; nothing that
        reads it is allowed to be fed to a detector.
        """
        weather = self.environment.to_frame()
        generation = self.equipment.to_frame()
        frame = generation.merge(weather, on=["event_time", "site_id"], how="left")

        per_interval = pd.DataFrame(
            {
                "event_time": self.observation.event_time,
                "active_sensor_fault": self.observation.active_sensor_fault,
                "active_comm_fault": self.delivery.active_comm_fault,
                "delivered": self.delivery.delivered,
            }
        )
        return frame.merge(per_interval, on="event_time", how="left")

    def observed_frame(self) -> pd.DataFrame:
        return self.observation.to_frame()

    def delivered_frame(self, *, utc_offset_hours: int) -> pd.DataFrame:
        """Observations as they reached the collector, in canonical schema.

        This is the only view that is shaped like the telemetry the rest of the
        project consumes, so it carries the legacy ``timestamp`` column
        alongside ``event_time`` — everything downstream still models in local
        standard time.
        """
        observed = self.observation.to_frame()
        observed = observed.assign(
            ingest_time=self.delivery.ingest_time,
            sequence=self.delivery.sequence,
            quality=self.delivery.quality,
            asset_id=self.observation.power_asset_id,
            data_mode="synthetic",
        )
        observed = observed[self.delivery.delivered].reset_index(drop=True)
        observed["timestamp"] = event_time_to_local(observed["event_time"], utc_offset_hours)
        return observed


@dataclass(frozen=True)
class SimulationResult:
    """The outcome of one run of the whole fleet."""

    config: SimulationConfig
    clock: LogicalClock
    sites: dict[str, SiteSimulation] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint()

    def truth(self) -> pd.DataFrame:
        return pd.concat([s.truth_frame() for s in self.sites.values()], ignore_index=True)

    def observed(self) -> pd.DataFrame:
        return pd.concat([s.observed_frame() for s in self.sites.values()], ignore_index=True)

    def delivered(self, *, utc_offset_hours: int | None = None) -> pd.DataFrame:
        if utc_offset_hours is None:
            from gridguard.data.synthetic import UTC_OFFSET_HOURS

            utc_offset_hours = UTC_OFFSET_HOURS
        return pd.concat(
            [s.delivered_frame(utc_offset_hours=utc_offset_hours) for s in self.sites.values()],
            ignore_index=True,
        )

    def canonical(self, *, utc_offset_hours: int | None = None) -> pd.DataFrame:
        """Delivered telemetry, validated against the canonical schema.

        The bridge to everything that already exists: features, models,
        calibration and detection all consume this shape unchanged.
        """
        frame = self.delivered(utc_offset_hours=utc_offset_hours)
        canonical_columns = [
            "timestamp",
            "site_id",
            "ac_power_kw",
            "irradiance_wm2",
            "temperature_c",
            "wind_speed_ms",
            "data_mode",
            "event_time",
            "ingest_time",
            "asset_id",
            "quality",
            "sequence",
        ]
        return validate_canonical(frame[canonical_columns], strict=False)

    def replay(self, *, utc_offset_hours: int | None = None):
        """Yield the run one tick at a time, paced by the clock's ``speedup``.

        Computation and pacing are deliberately separated. The layers are
        vectorised over the whole run, so generating a month of a fleet takes
        about a second — there is nothing to pace *during* computation, and
        slowing it down would only make experiments slower for no gain.

        A demo, though, wants to watch a day unfold. So it replays a
        finished result: the values are already fixed, and ``speedup`` controls
        only how fast they are handed out. That keeps the guarantee that pacing
        cannot change results true by construction rather than by discipline —
        the numbers exist before the pacing starts.

        Yields ``(Tick, rows)`` where ``rows`` is the canonical frame for that
        instant across every simulated site.
        """
        frame = self.canonical(utc_offset_hours=utc_offset_hours)
        by_time = {time: group for time, group in frame.groupby("event_time", sort=True)}
        for tick in self.clock:
            yield tick, by_time.get(pd.Timestamp(tick.event_time), frame.iloc[0:0])

    def schedule(self) -> FaultSchedule:
        """Every fault injected across the fleet."""
        return FaultSchedule(tuple(f for s in self.sites.values() for f in s.schedule))

    def fault_summary(self) -> pd.DataFrame:
        """Per fault: what was injected, and what it actually cost.

        ``lost_kwh`` is exact rather than modelled — it is the recorded gap
        between potential and actual generation over the fault window, which
        exists only because the equipment layer keeps both.
        """
        rows = []
        for site_id, sim in self.sites.items():
            for fault in sim.schedule:
                window = fault.mask(sim.equipment.event_time)
                interval_hours = self.config.interval_minutes / 60.0
                lost = float(
                    (sim.equipment.site_potential_kw - sim.equipment.site_actual_kw)[window].sum()
                    * interval_hours
                )
                undelivered = int((~sim.delivery.delivered)[window].sum())
                rows.append(
                    {
                        "site_id": site_id,
                        "asset_id": fault.asset_id,
                        "fault_type": fault.fault_type.value,
                        "category": fault.category.value,
                        "layer": fault.layer,
                        "start": fault.start,
                        "end": fault.end,
                        "intervals": int(window.sum()),
                        "severity": fault.severity,
                        "is_generation_loss": fault.is_generation_loss,
                        "lost_kwh": round(lost, 2),
                        "undelivered_intervals": undelivered,
                    }
                )
        return pd.DataFrame(rows)

    def observation_error(self) -> pd.DataFrame:
        """How far each observed channel sits from the truth it reports on.

        Only computable because the layers are kept apart, and the reason they
        are: this is the baseline an attribution experiment measures a sensor
        fault against.
        """
        rows = []
        for site_id, sim in self.sites.items():
            rows.append(
                {
                    "site_id": site_id,
                    "irradiance_bias_wm2": float(
                        (
                            sim.observation.observed_irradiance_wm2
                            - sim.environment.poa_irradiance_wm2
                        ).mean()
                    ),
                    "irradiance_mae_wm2": float(
                        abs(
                            sim.observation.observed_irradiance_wm2
                            - sim.environment.poa_irradiance_wm2
                        ).mean()
                    ),
                    "power_bias_kw": float(
                        (sim.observation.observed_ac_power_kw - sim.equipment.site_actual_kw).mean()
                    ),
                    "power_mae_kw": float(
                        abs(
                            sim.observation.observed_ac_power_kw - sim.equipment.site_actual_kw
                        ).mean()
                    ),
                    "delivery_ratio": sim.delivery.delivery_ratio,
                }
            )
        return pd.DataFrame(rows)


class FleetSimulator:
    """Runs the layered simulation for a fleet of sites."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def run(self) -> SimulationResult:
        clock = self.config.clock()
        logger.info(
            "Simulating %d site(s) over %d ticks (%s -> %s), seed=%d, fingerprint=%s",
            len(self.config.site_ids),
            len(clock),
            clock.start.isoformat(),
            clock.end.isoformat(),
            self.config.seed,
            self.config.fingerprint(),
        )

        sites = {}
        for site_id in self.config.site_ids:
            sites[site_id] = self.run_site(site_id, clock)
        result = SimulationResult(config=self.config, clock=clock, sites=sites)
        if self.config.inject_faults:
            logger.info("Injected %d fault(s) across the fleet.", len(result.schedule()))
        return result

    def _schedule_for(self, site_id: str, clock: LogicalClock) -> FaultSchedule:
        """The fault schedule for one site, attached to its real assets."""
        if not self.config.inject_faults:
            return FaultSchedule()

        tree = assets_for_site(site_id)
        pyranometers = tree.of_kind(AssetKind.PYRANOMETER)
        return build_layered_scenario(
            {
                "array": tree.primary_array.asset_id,
                "pyranometer": (
                    pyranometers[0].asset_id if pyranometers else tree.primary_array.asset_id
                ),
                # The communication fault belongs to the link carrying the
                # site's telemetry. There is no link asset yet — the edge agent
                # in phase 5 introduces one — so it attaches to the reporting
                # asset, which is what a collector would blame today.
                "link": tree.primary_array.asset_id,
            },
            clock.index(),
        )

    def run_site(self, site_id: str, clock: LogicalClock) -> SiteSimulation:
        site = get_site(site_id)
        tree = assets_for_site(site_id)
        schedule = self._schedule_for(site_id, clock)

        environment = EnvironmentLayer(site, self.config).run(clock)
        equipment = EquipmentLayer(tree, self.config, schedule).run(environment)
        observation = SensorLayer(tree, self.config, schedule).run(environment, equipment)
        delivery = TransportLayer(self.config, schedule).run(observation)

        return SiteSimulation(
            site_id=site_id,
            environment=environment,
            equipment=equipment,
            observation=observation,
            delivery=delivery,
            schedule=schedule,
        )


def simulate(
    site_ids: list[str] | tuple[str, ...],
    *,
    start: str | None = None,
    end: str | None = None,
    seed: int | None = None,
    **overrides,
) -> SimulationResult:
    """Convenience entry point for a default-configured run."""
    from gridguard.simulation.config import DEFAULT_END, DEFAULT_SEED, DEFAULT_START

    config = SimulationConfig(
        site_ids=tuple(site_ids),
        start=start or DEFAULT_START,
        end=end or DEFAULT_END,
        seed=DEFAULT_SEED if seed is None else seed,
        **overrides,
    )
    return FleetSimulator(config).run()
