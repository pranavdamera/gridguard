"""
Layer 2 — equipment.

What the plant actually produces, given the weather. Per asset, because an
inverter can trip while its sibling keeps running, and that difference is the
whole basis of attribution.

Still ground truth: this is generation as it physically occurred, before any
instrument has looked at it. A meter reading that disagrees with these numbers
is a sensor fault, and keeping the two apart is what makes that statement
decidable rather than a matter of interpretation.

Electrical model
----------------
Deliberately the same one the shipped generator already uses — irradiance
scaling, a linear cell-temperature derate, the combined system derate, an AC
cap at the inverter rating, and the slow AR(1) performance drift. No diode
models, no per-string mismatch, no clipping curves. The project's research
question is about detection and attribution under degraded telemetry, not about
PV device physics, and a more elaborate electrical model would add parameters
nobody can validate against the three real arrays available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from gridguard.data.synthetic import performance_drift
from gridguard.domain.asset import Asset, AssetKind, AssetTree
from gridguard.simulation.config import SimulationConfig, stream_for
from gridguard.simulation.environment import EnvironmentTruth

#: Cell temperature rise above ambient at full irradiance, in °C. The simple
#: linear proxy the shipped generator uses, kept identical.
_CELL_TEMP_RISE_AT_1000_WM2 = 28.0


class EquipmentState(StrEnum):
    """What an asset is doing, as ground truth.

    Recorded per interval so a later experiment can ask "was this asset actually
    offline?" without inferring it from the power it produced — which is exactly
    the inference under test.
    """

    HEALTHY = "healthy"
    DERATED = "derated"
    OFFLINE = "offline"


@dataclass(frozen=True)
class AssetGeneration:
    """Ground-truth generation for one generating asset over a run."""

    asset_id: str
    site_id: str
    capacity_kw: float

    #: What the asset would have produced with no equipment impairment.
    potential_ac_power_kw: np.ndarray
    #: What it actually produced.
    actual_ac_power_kw: np.ndarray
    #: Ground-truth state per interval.
    state: np.ndarray
    #: Cell temperature (°C), retained because it drives the derate and is
    #: useful evidence when attributing a thermal fault.
    cell_temperature_c: np.ndarray

    @property
    def lost_kw(self) -> np.ndarray:
        """Generation lost to equipment impairment, per interval."""
        return np.maximum(0.0, self.potential_ac_power_kw - self.actual_ac_power_kw)


@dataclass(frozen=True)
class EquipmentTruth:
    """Ground-truth generation for a whole site."""

    site_id: str
    event_time: pd.DatetimeIndex
    assets: tuple[AssetGeneration, ...]

    @property
    def site_actual_kw(self) -> np.ndarray:
        """Total real output of the site."""
        return np.sum([a.actual_ac_power_kw for a in self.assets], axis=0)

    @property
    def site_potential_kw(self) -> np.ndarray:
        """What the site would have produced with no equipment impairment."""
        return np.sum([a.potential_ac_power_kw for a in self.assets], axis=0)

    def to_frame(self) -> pd.DataFrame:
        """Long form: one row per (interval, asset)."""
        frames = []
        for asset in self.assets:
            frames.append(
                pd.DataFrame(
                    {
                        "event_time": self.event_time,
                        "site_id": self.site_id,
                        "asset_id": asset.asset_id,
                        "potential_ac_power_kw": asset.potential_ac_power_kw,
                        "actual_ac_power_kw": asset.actual_ac_power_kw,
                        "lost_ac_power_kw": asset.lost_kw,
                        "cell_temperature_c": asset.cell_temperature_c,
                        "equipment_state": asset.state,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


class EquipmentLayer:
    """Turns weather into per-asset ground-truth generation."""

    def __init__(self, tree: AssetTree, config: SimulationConfig) -> None:
        self.tree = tree
        self.config = config

    def run(self, environment: EnvironmentTruth) -> EquipmentTruth:
        arrays = self.tree.of_kind(AssetKind.ARRAY)
        if not arrays:
            raise ValueError(f"{self.tree.site_id} has no generating assets to simulate")

        generation = tuple(self._run_array(array, environment) for array in arrays)
        return EquipmentTruth(
            site_id=self.tree.site_id,
            event_time=environment.event_time,
            assets=generation,
        )

    def _run_array(self, array: Asset, environment: EnvironmentTruth) -> AssetGeneration:
        settings = self.config.equipment
        n = len(environment.event_time)
        capacity = array.capacity_kw or 0.0

        irradiance = environment.poa_irradiance_wm2
        cell_temperature = (
            environment.ambient_temperature_c + irradiance / 1000.0 * _CELL_TEMP_RISE_AT_1000_WM2
        )
        efficiency = 1.0 - settings.temperature_coefficient_per_c * np.maximum(
            0.0, cell_temperature - 25.0
        )

        if settings.performance_drift:
            drift = performance_drift(
                n, stream_for(self.config.seed, array.asset_id, "equipment", "drift")
            )
        else:
            drift = np.ones(n)

        potential = capacity * (irradiance / 1000.0) * efficiency * settings.system_derate * drift
        potential = np.clip(potential, 0.0, capacity)

        # Phase 2 injects no equipment faults: every asset is healthy, and
        # actual equals potential. The fault taxonomy is applied by
        # gridguard.data.faults, which phase 3 moves into this layer so that a
        # fault becomes a change of equipment *state* rather than an edit to a
        # column of numbers.
        actual = potential.copy()
        state = np.full(n, EquipmentState.HEALTHY.value, dtype=object)

        return AssetGeneration(
            asset_id=array.asset_id,
            site_id=array.site_id,
            capacity_kw=capacity,
            potential_ac_power_kw=potential,
            actual_ac_power_kw=actual,
            state=state,
            cell_temperature_c=cell_temperature,
        )
