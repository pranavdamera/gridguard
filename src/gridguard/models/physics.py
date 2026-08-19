"""
Physics-based expected-generation baseline (pvlib).

Everything else in the model zoo learns the mapping from weather to power from
data. This module computes it from first principles instead, using the array's
published geometry and a standard PV performance chain:

    plane-of-array irradiance
        -> cell temperature      (Sandia thermal model, from ambient + wind)
        -> DC power              (PVWatts, with a module temperature coefficient)
        -> AC power              (inverter efficiency and AC clipping)

Why bother when XGBoost scores better
-------------------------------------
Three reasons that matter for an anomaly detector:

1. **It cannot learn a fault as normal.** A statistical baseline trained on
   history absorbs whatever degradation is present in that history. The physics
   chain has no memory: it always answers "what *should* this array produce in
   this weather?"
2. **It transfers to a site with no history.** A newly commissioned array has
   no training data, but its tilt, azimuth and nameplate are known on day one.
3. **It is auditable.** Every term corresponds to a documented physical effect
   with a stated assumption, so a disagreement can be traced to a specific one.

The honest caveat is that GridGuard's synthetic fleet is itself generated from a
pvlib clear-sky model, so this baseline is close to tautological there. Its only
meaningful evaluation is on measured telemetry, and the model card reports it
only on real sites.

Stated assumptions
------------------
Published PVDAQ metadata gives coordinates, tilt, azimuth, module model and
count, and DC nameplate. It does not give module temperature coefficients,
thermal mounting parameters, or inverter efficiency curves. Those come from
pvlib's standard parameter sets and are exposed on :class:`PhysicsAssumptions`
so that every unmeasured quantity is visible rather than buried.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from gridguard.sites.registry import Site

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhysicsAssumptions:
    """Physical parameters not published with the system metadata.

    Defaults describe a generic fixed-tilt crystalline-silicon array on an open
    rack. They are deliberately conservative and generic: this is a baseline,
    not a bankable energy model.
    """

    #: Module power temperature coefficient, fraction per degree C. Typical
    #: crystalline silicon is -0.003 to -0.005; -0.0040 is mid-range.
    gamma_pdc: float = -0.0040

    #: Sandia open-rack glass/glass thermal model coefficients.
    temp_model_a: float = -3.56
    temp_model_b: float = -0.075
    temp_model_deltaT: float = 3.0

    #: Ratio of inverter AC rating to array DC nameplate. Arrays are commonly
    #: oversized relative to the inverter, which is what causes clipping.
    dc_ac_ratio: float = 1.15

    #: Nominal inverter efficiency at rated power.
    inverter_efficiency: float = 0.96

    #: Fixed system losses not otherwise modelled: soiling, wiring, mismatch,
    #: connections, light-induced degradation, availability. PVWatts' default
    #: total is 14%; the calibrated derate learned in `fit` replaces this when
    #: training data is available.
    system_losses: float = 0.14

    def describe(self) -> dict[str, float]:
        return asdict(self)


class PhysicsBaseline(BaseEstimator, RegressorMixin):
    """Expected AC generation from array geometry and measured weather.

    Implements the scikit-learn estimator API so it drops into the same
    comparison harness as the learned models.

    ``fit`` calibrates exactly **one** scalar — an overall derate absorbing the
    difference between the generic loss assumptions and this array's actual
    performance. Everything else is fixed physics. One free parameter keeps the
    comparison against multi-hundred-parameter tree ensembles fair without
    turning the baseline into a fitted model in disguise.

    Requires ``irradiance_wm2``, ``temperature_c`` and ``wind_speed_ms`` in the
    feature frame, which the standard weather-only feature set provides.
    """

    def __init__(
        self,
        site: Site,
        assumptions: PhysicsAssumptions | None = None,
        calibrate: bool = True,
    ) -> None:
        self.site = site
        self.assumptions = assumptions or PhysicsAssumptions()
        self.calibrate = calibrate
        self.derate_: float = 1.0
        self.fitted_: bool = False

    # -- scikit-learn API ---------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PhysicsBaseline:
        """Calibrate the single scalar derate by least squares on daylight rows."""
        if not self.calibrate:
            self.fitted_ = True
            return self

        raw = self._uncalibrated_ac_power(X)
        actual = np.asarray(y, dtype=float)
        daylight = (raw > 0.01) & np.isfinite(actual)

        if daylight.sum() < 100:
            logger.warning(
                "Only %d daylight samples available to calibrate the physics baseline; "
                "leaving the derate at 1.0.",
                int(daylight.sum()),
            )
            self.derate_ = 1.0
        else:
            # Least-squares scalar: minimises ||k*raw - actual||, i.e. the
            # projection of actual onto raw.
            denominator = float(np.sum(raw[daylight] ** 2))
            self.derate_ = (
                float(np.sum(raw[daylight] * actual[daylight]) / denominator)
                if denominator > 0
                else 1.0
            )
            # Guard against a pathological fit on bad data.
            self.derate_ = float(np.clip(self.derate_, 0.3, 1.8))
            logger.info(
                "Physics baseline calibrated for %s: derate=%.3f (from %d daylight samples).",
                self.site.site_id,
                self.derate_,
                int(daylight.sum()),
            )

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self._uncalibrated_ac_power(X) * self.derate_, 0, None)

    # -- physics ------------------------------------------------------------

    def _uncalibrated_ac_power(self, X: pd.DataFrame) -> np.ndarray:
        """Run the PV performance chain, before the calibrated derate."""
        import pvlib

        missing = [
            c for c in ("irradiance_wm2", "temperature_c", "wind_speed_ms") if c not in X.columns
        ]
        if missing:
            raise ValueError(f"PhysicsBaseline needs columns {missing} in X.")

        poa = pd.to_numeric(X["irradiance_wm2"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        temp_air = pd.to_numeric(X["temperature_c"], errors="coerce").to_numpy(dtype=float)
        wind = pd.to_numeric(X["wind_speed_ms"], errors="coerce").to_numpy(dtype=float)

        # Weather channels can be missing even when power is not; fall back to
        # mild, explicitly-stated defaults rather than propagating NaN into the
        # prediction (which would silently drop rows downstream).
        temp_air = np.where(np.isfinite(temp_air), temp_air, 20.0)
        wind = np.where(np.isfinite(wind), wind, 1.0)

        a = self.assumptions
        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=poa,
            temp_air=temp_air,
            wind_speed=wind,
            a=a.temp_model_a,
            b=a.temp_model_b,
            deltaT=a.temp_model_deltaT,
        )

        pdc0_kw = self.site.capacity_kw
        # Passed positionally: pvlib renamed the first parameter from
        # `g_poa_effective` to `effective_irradiance` in 0.13, and positional
        # arguments work across both.
        dc_power = pvlib.pvsystem.pvwatts_dc(poa, cell_temp, pdc0_kw, a.gamma_pdc)
        dc_power = np.clip(dc_power, 0, None) * (1.0 - a.system_losses)

        # Inverter: constant efficiency with a hard AC clip at its rating.
        ac_rating_kw = pdc0_kw / a.dc_ac_ratio
        ac_power = np.minimum(dc_power * a.inverter_efficiency, ac_rating_kw)
        return np.clip(ac_power, 0, None)

    # -- introspection ------------------------------------------------------

    def explain(self) -> dict:
        """Report the assumptions and calibration, for the methodology page."""
        return {
            "site_id": self.site.site_id,
            "capacity_kw": self.site.capacity_kw,
            "capacity_basis": self.site.capacity_basis,
            "tilt_deg": self.site.tilt_deg,
            "azimuth_deg": self.site.azimuth_deg,
            "calibrated_derate": round(self.derate_, 4),
            "free_parameters": 1 if self.calibrate else 0,
            "assumptions": self.assumptions.describe(),
            "assumption_note": (
                "Tilt, azimuth and DC nameplate come from the published system metadata. "
                "Temperature coefficient, thermal model coefficients, inverter efficiency "
                "and DC/AC ratio are generic crystalline-silicon defaults, not "
                "manufacturer data for this specific array."
            ),
        }


class PhysicsResidualModel(BaseEstimator, RegressorMixin):
    """Hybrid: physics prediction plus a learned correction of its residual.

    The physics chain captures the deterministic part — sun geometry, thermal
    derating, inverter clipping. A gradient-boosted model then learns what the
    physics systematically misses at this particular site: soiling seasonality,
    row-to-row shading at low sun angles, spectral effects, sensor bias.

    This ordering matters for anomaly detection. The learned component only ever
    sees the *residual* of a physical model, so it has far less capacity to
    quietly absorb a real fault as "normal" than a model fitted directly to
    power. The physics term anchors the prediction.
    """

    def __init__(
        self,
        site: Site,
        assumptions: PhysicsAssumptions | None = None,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        self.site = site
        self.assumptions = assumptions
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> PhysicsResidualModel:
        from xgboost import XGBRegressor

        self.physics_ = PhysicsBaseline(self.site, self.assumptions).fit(X, y)
        physics_prediction = self.physics_.predict(X)
        residual = np.asarray(y, dtype=float) - physics_prediction

        self.correction_ = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )
        self.correction_.fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        physics_prediction = self.physics_.predict(X)
        correction = self.correction_.predict(X)
        return np.clip(physics_prediction + correction, 0, None)

    def explain(self) -> dict:
        detail = self.physics_.explain()
        detail["hybrid"] = (
            "Gradient-boosted correction fitted to the residual of the physics chain, "
            "not to power directly."
        )
        return detail
