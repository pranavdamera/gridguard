"""
Layer 3 — sensing.

The first layer whose output is not the truth. Everything above this line is
what happened; everything below is what somebody *claims* happened.

Instruments are assets (see :mod:`gridguard.domain.asset`), so an observation is
attributable to a specific pyranometer or meter rather than being an ambient
property of a site. That is what makes a frozen sensor a fault with an owner.

Three things separate an observation from the truth even when nothing is wrong:

* **Calibration error.** A pyranometer reads a percent or two off, consistently.
  Constant per instrument, so it looks like a bias rather than noise.
* **Noise.** Per-reading scatter.
* **Quantisation.** A meter reports to a fixed resolution.

None of these is a fault. They are why observed values never exactly equal true
ones, and why a detector trained on observations has an irreducible residual.
Faults proper — frozen channels, dropouts, drift — arrive in phase 3 and attach
to this layer, because a corrupted reading changes what is *observed* without
changing what the plant did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridguard.domain.asset import AssetKind, AssetTree
from gridguard.domain.telemetry import QualityFlag
from gridguard.simulation.config import SimulationConfig, stream_for
from gridguard.simulation.environment import EnvironmentTruth
from gridguard.simulation.equipment import EquipmentTruth


@dataclass(frozen=True)
class Observation:
    """What the instruments at one site reported, per interval.

    Deliberately parallel in shape to :class:`EnvironmentTruth` and
    :class:`EquipmentTruth`, so the two can be compared element-wise. The gap
    between them is the observation error, and it is a quantity the attribution
    experiments measure directly rather than estimate.
    """

    site_id: str
    event_time: pd.DatetimeIndex

    observed_irradiance_wm2: np.ndarray
    observed_temperature_c: np.ndarray
    observed_wind_speed_ms: np.ndarray
    observed_ac_power_kw: np.ndarray

    #: Which instrument produced each channel, for attribution.
    irradiance_asset_id: str
    power_asset_id: str
    weather_asset_id: str

    #: Per-interval quality flags, one column per channel.
    irradiance_quality: np.ndarray
    power_quality: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_time": self.event_time,
                "site_id": self.site_id,
                "irradiance_wm2": self.observed_irradiance_wm2,
                "temperature_c": self.observed_temperature_c,
                "wind_speed_ms": self.observed_wind_speed_ms,
                "ac_power_kw": self.observed_ac_power_kw,
                "irradiance_asset_id": self.irradiance_asset_id,
                "power_asset_id": self.power_asset_id,
                "irradiance_quality": self.irradiance_quality,
                "power_quality": self.power_quality,
            }
        )


class SensorLayer:
    """Turns ground truth into what the instruments report about it."""

    def __init__(self, tree: AssetTree, config: SimulationConfig) -> None:
        self.tree = tree
        self.config = config

    def _instrument(self, kind: AssetKind, fallback: str) -> str:
        assets = self.tree.of_kind(kind)
        return assets[0].asset_id if assets else fallback

    def run(self, environment: EnvironmentTruth, equipment: EquipmentTruth) -> Observation:
        settings = self.config.sensors
        n = len(environment.event_time)
        site_id = self.tree.site_id

        pyranometer = self._instrument(AssetKind.PYRANOMETER, f"{site_id}_pyranometer_1")
        weather_station = self._instrument(
            AssetKind.WEATHER_STATION, f"{site_id}_weather_station_1"
        )
        meters = self.tree.of_kind(AssetKind.METER) or self.tree.of_kind(AssetKind.INVERTER)
        power_asset = meters[0].asset_id if meters else self.tree.primary_array.asset_id

        # --- irradiance ---------------------------------------------------
        # Calibration error is drawn once per instrument, not per reading: a
        # miscalibrated pyranometer is consistently wrong, which is what makes
        # it a bias the model absorbs rather than noise it averages out.
        calibration_stream = stream_for(self.config.seed, pyranometer, "sensor", "calibration")
        calibration = 1.0 + calibration_stream.normal(0.0, settings.irradiance_calibration_error)

        irradiance_noise = stream_for(self.config.seed, pyranometer, "sensor", "noise").normal(
            0.0, settings.irradiance_noise, n
        )
        observed_irradiance = np.clip(
            environment.poa_irradiance_wm2 * calibration * (1.0 + irradiance_noise), 0.0, None
        )

        # --- ambient conditions -------------------------------------------
        observed_temperature = environment.ambient_temperature_c + stream_for(
            self.config.seed, weather_station, "sensor", "temperature"
        ).normal(0.0, settings.temperature_noise_c, n)
        observed_wind = np.clip(
            environment.wind_speed_ms
            + stream_for(self.config.seed, weather_station, "sensor", "wind").normal(0.0, 0.2, n),
            0.0,
            None,
        )

        # --- power ----------------------------------------------------------
        true_power = equipment.site_actual_kw
        power_noise = stream_for(self.config.seed, power_asset, "sensor", "power").normal(
            0.0, settings.power_noise, n
        )
        observed_power = np.clip(true_power * (1.0 + power_noise), 0.0, None)
        if settings.power_quantum_kw > 0:
            observed_power = (
                np.round(observed_power / settings.power_quantum_kw) * settings.power_quantum_kw
            )

        # Phase 2 injects no sensor faults, so every reading is a real reading.
        # The columns exist now so that phase 3 sets flags rather than adding a
        # schema, and so a consumer written today already handles them.
        ok = np.full(n, int(QualityFlag.OK), dtype=np.int64)

        return Observation(
            site_id=site_id,
            event_time=environment.event_time,
            observed_irradiance_wm2=observed_irradiance,
            observed_temperature_c=observed_temperature,
            observed_wind_speed_ms=observed_wind,
            observed_ac_power_kw=observed_power,
            irradiance_asset_id=pyranometer,
            power_asset_id=power_asset,
            weather_asset_id=weather_station,
            irradiance_quality=ok,
            power_quality=ok.copy(),
        )
