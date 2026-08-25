"""
The layered site simulator.

Four layers, each answering a different question, and deliberately not merged:

===============  =========================================================
``environment``  What the sky and the air did. Ground truth.
``equipment``    What the plant produced, per asset. Ground truth.
``sensing``      What the instruments reported. First layer that can lie.
``transport``    What reached the collector. Pass-through until phase 5.
===============  =========================================================

The separation is the point. Three different things make a value absent or wrong
at the collector — the plant produced nothing, the instrument failed, or the
reading never arrived — and an architecture that cannot express all three will
attribute one to another. That misattribution is the failure mode this project
exists to study.

Reproducibility
---------------
A run is a pure function of ``(SimulationConfig, seed)``. No wall-clock, no
environment, no process-dependent hashing. Simulated time comes from
:class:`~gridguard.simulation.clock.LogicalClock`, which generates instants
rather than reading them, and every layer draws from its own named random
stream so that enabling a fault in one layer provably cannot perturb another.

``speedup`` paces the walk through simulated time for a demo, and never changes
what a tick contains — asserted in tests, not assumed.

    from gridguard.simulation import simulate

    result = simulate(["nist_roof"], start="2016-06-01", end="2016-06-07")
    result.truth()      # what happened
    result.observed()   # what the instruments said
    result.canonical()  # what the existing pipeline consumes
"""

from gridguard.simulation.clock import DEFAULT_INTERVAL_MINUTES, LogicalClock, Tick
from gridguard.simulation.config import (
    DEFAULT_END,
    DEFAULT_SEED,
    DEFAULT_START,
    EquipmentConfig,
    SensorConfig,
    SimulationConfig,
    TransportConfig,
    stream_for,
)
from gridguard.simulation.environment import EnvironmentLayer, EnvironmentTruth
from gridguard.simulation.equipment import (
    AssetGeneration,
    EquipmentLayer,
    EquipmentState,
    EquipmentTruth,
)
from gridguard.simulation.runner import (
    FleetSimulator,
    SimulationResult,
    SiteSimulation,
    simulate,
)
from gridguard.simulation.sensing import Observation, SensorLayer
from gridguard.simulation.transport import Delivery, TransportLayer

__all__ = [
    "DEFAULT_END",
    "DEFAULT_INTERVAL_MINUTES",
    "DEFAULT_SEED",
    "DEFAULT_START",
    "AssetGeneration",
    "Delivery",
    "EnvironmentLayer",
    "EnvironmentTruth",
    "EquipmentConfig",
    "EquipmentLayer",
    "EquipmentState",
    "EquipmentTruth",
    "FleetSimulator",
    "LogicalClock",
    "Observation",
    "SensorConfig",
    "SensorLayer",
    "SimulationConfig",
    "SimulationResult",
    "SiteSimulation",
    "Tick",
    "TransportConfig",
    "TransportLayer",
    "simulate",
    "stream_for",
]
