"""
Simulation configuration and seed derivation.

A run is a pure function of ``(SimulationConfig, seed)``. Nothing else — no
environment variables, no wall-clock, no filesystem state, no process-dependent
hashing. That is the contract the whole simulator is built to keep, and it is
what makes an experiment in phase 11 reproducible from a line in a report.

Independent random streams per layer
------------------------------------
The single most important design decision here is that each layer of each asset
draws from its **own** stream, derived from the run seed by name.

The tempting alternative — one ``default_rng(seed)`` threaded through
everything — silently couples the layers. Enable a sensor fault and the sensor
layer draws extra numbers, which shifts every subsequent draw, which changes the
weather. Ground truth would then depend on what the *instruments* were doing,
which is exactly backwards, and it would make the fault-attribution experiments
this simulator exists to support meaningless: you could never tell whether a
detector responded to the injected sensor fault or to the different weather that
came with it.

``numpy.random.SeedSequence`` spawn-by-name gives every (layer, asset, purpose)
its own independent stream from one root seed, so enabling a fault in one layer
provably cannot perturb another. That property is asserted in tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import blake2b

import numpy as np

from gridguard.simulation.clock import DEFAULT_INTERVAL_MINUTES, LogicalClock

#: Default simulated window. Aligned with the measured datasets so a single
#: fleet view can show simulated and measured sites on one timeline — see
#: :mod:`gridguard.data.synthetic`.
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2016-12-31"

#: Default run seed. Any integer works; this one is what the shipped fleet uses.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SensorConfig:
    """How faithfully instruments report the truth.

    Defaults describe a well-maintained installation: small calibration bias,
    small noise, no faults. Degradation is opt-in so that a baseline run has an
    observation layer that is nearly, but never exactly, transparent.
    """

    #: Multiplicative calibration error on the irradiance channel. A real
    #: pyranometer is typically within a couple of percent of true.
    irradiance_calibration_error: float = 0.02

    #: Additive noise on each channel, as a fraction of the reading.
    irradiance_noise: float = 0.01
    power_noise: float = 0.01

    #: Absolute noise on ambient temperature, in degrees C.
    temperature_noise_c: float = 0.3

    #: Resolution the meter quantises power to, in kW. Zero disables
    #: quantisation. Real meters report to a fixed number of digits, and that
    #: rounding is a genuine source of small residuals.
    power_quantum_kw: float = 0.0


@dataclass(frozen=True)
class EquipmentConfig:
    """How the plant itself behaves.

    These reproduce the shipped generator's physical model rather than replacing
    it — the same temperature coefficient, system derate, and drift process. The
    values live here so a scenario can vary them, not so they can diverge.
    """

    #: Fractional efficiency loss per degree of cell temperature above 25 C.
    temperature_coefficient_per_c: float = 0.004

    #: Soiling, wiring, mismatch and inverter efficiency, combined.
    system_derate: float = 0.85

    #: Whether to apply the slow performance drift. On by default: without it
    #: the array is a deterministic function of irradiance, which makes lagged
    #: power carry no information and quietly removes the leakage hazard the
    #: weather-only feature set exists to avoid.
    performance_drift: bool = True


@dataclass(frozen=True)
class TransportConfig:
    """How telemetry reaches the collector.

    Phase 2 ships a pass-through: everything arrives, in order, instantly. The
    knobs are declared now so the interface does not change when the edge agent
    and broker arrive, and so a scenario written today keeps meaning the same
    thing later.
    """

    #: Fraction of records dropped in transit. 0.0 = lossless.
    loss_rate: float = 0.0

    #: Mean delivery delay in seconds. 0.0 = instantaneous.
    mean_delay_seconds: float = 0.0

    #: Whether records may arrive out of order.
    reorder: bool = False

    @property
    def is_pass_through(self) -> bool:
        """Whether this configuration delivers everything, instantly, in order."""
        return self.loss_rate == 0.0 and self.mean_delay_seconds == 0.0 and not self.reorder


@dataclass(frozen=True)
class SimulationConfig:
    """Everything needed to reproduce a run, apart from the seed."""

    site_ids: tuple[str, ...]
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES
    seed: int = DEFAULT_SEED

    #: Wall-clock pacing for demos. Never affects the values produced.
    speedup: float | None = None

    equipment: EquipmentConfig = field(default_factory=EquipmentConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)

    #: Inject the full fault taxonomy, attached to each site's real assets.
    #: Off by default: a baseline run is a healthy fleet, and faults are an
    #: explicit choice rather than something a caller gets by accident.
    inject_faults: bool = False

    def __post_init__(self) -> None:
        if not self.site_ids:
            raise ValueError("SimulationConfig needs at least one site")
        if len(set(self.site_ids)) != len(self.site_ids):
            raise ValueError(f"duplicate site ids: {self.site_ids}")

    def clock(self) -> LogicalClock:
        """The logical clock this configuration describes."""
        return LogicalClock.from_range(
            self.start,
            self.end,
            interval_minutes=self.interval_minutes,
            speedup=self.speedup,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """A stable digest of everything that affects the output.

        ``speedup`` is excluded, because pacing must not change results — two
        runs that differ only in speed are the same run, and this is what says
        so. Recorded alongside experiment outputs so a figure can be traced to
        the exact configuration that produced it.
        """
        payload = self.to_dict()
        payload.pop("speedup", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return blake2b(encoded, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def _name_to_int(*parts: str) -> int:
    """A stable integer from string parts, for SeedSequence spawn keys.

    ``blake2b`` rather than ``hash()``: Python salts string hashing per process,
    which has already cost this project a determinism guarantee once.
    """
    digest = blake2b("\x00".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def stream_for(seed: int, *name: str) -> np.random.Generator:
    """An independent generator for one named purpose.

    ``stream_for(42, "nist_roof", "environment", "cloud")`` always returns the
    same stream, and it is statistically independent of every other name derived
    from the same seed. Adding a new named stream never perturbs an existing
    one, which is what lets a layer gain a random effect without changing any
    other layer's output.
    """
    if not name:
        raise ValueError("stream_for needs at least one name part")
    return np.random.default_rng(
        np.random.SeedSequence(entropy=seed, spawn_key=(_name_to_int(*name),))
    )
