# Research Angle — GridGuard

This document maps GridGuard's technical design to active research problems in
geospatial AI, energy systems, and machine learning. Intended audience: professors,
research collaborators, and PhD program applications.

---

## Problem framing

GridGuard addresses a concrete operational problem: **distributed solar asset visibility**.

As distributed energy resources (DERs) proliferate — rooftop PV, campus microgrids,
community solar — grid operators and facility managers lose visibility into real-time
generation. A 250 kW campus PV system that underperforms by 30% for a week may go
unnoticed because:

1. No automated comparison against expected output exists.
2. Residual-based methods require careful calibration to avoid both false positives
   (flagging a cloudy afternoon as a fault) and false negatives (missing slow degradation).
3. Geospatial fleet visibility is limited — operators see a table of numbers, not a map
   of health signals.

GridGuard builds toward a solution at the intersection of:
- **Time series anomaly detection** (ML / statistics)
- **Geospatial data integration** (site registry, irradiance, grid topology)
- **Physics-informed feature engineering** (sun geometry, temperature efficiency)
- **Uncertainty quantification** (conformal prediction for coverage-guaranteed alerts)

---

## Research threads

### 1. Conformal prediction for energy systems

**Problem:** Standard residual thresholding provides no formal coverage guarantee.
A threshold of 2σ flags ~2.3% of intervals regardless of actual fault rate, which
may be too permissive or too strict depending on system and season.

**Research direction:** Apply split conformal prediction (Tibshirani et al., 2019)
to compute a lower prediction bound with a provable miscoverage rate. Key challenges:

- Exchangeability assumption is violated for time series (solar output is autocorrelated).
- Covariate shift between seasons means calibration-set residuals may not represent
  future test conditions.
- Per-hour conditioning (to handle irradiance-magnitude variation) reduces calibration
  set size and may weaken the guarantee.

GridGuard has the foundation in `anomaly/conformal.py`. The research extension would
involve covariate-shift-robust conformal methods (Tibshirani et al., 2019; Barber et al., 2023).

### 2. Leakage-aware feature design for fault detection

**Problem:** Lag features (`ac_power_lag1`) are strong predictors for short-horizon
forecasting but introduce leakage in anomaly detection — a persistently degraded system
produces low lag values, causing the model to predict low output as "normal."

**Research direction:** Formalise the distinction between *expected-generation models*
(weather-only) and *short-horizon forecasting models* (lag-aware). Evaluate whether
a two-model architecture (separate anomaly baseline and operational forecast) outperforms
a single lag-aware model on event-level precision/recall metrics.

This connects to broader literature on **concept drift** and **out-of-distribution
detection** in time series.

### 3. Geospatial irradiance as a feature

**Problem:** The current synthetic generator uses a per-site latitude to compute
sun geometry, but ignores:

- Terrain shading (horizon masking from buildings/hills)
- Cloud shadow dynamics (moving shadows can cause sub-minute ramp events)
- Aerosol optical depth (varies by season and fire/pollution events)

**Research direction:** Integrate NREL NSRDB irradiance data (1 km resolution,
half-hourly) to replace the simplified sun geometry. Evaluate whether spatially
interpolated NSRDB features improve anomaly precision (fewer false positives on
cloudy days where irradiance drops legitimately).

This is a **geospatial ML** problem: how do you aggregate satellite-derived irradiance
to a rooftop-scale PV system, and does the aggregation error matter for fault detection?

### 4. Fleet-level anomaly correlation

**Problem:** Faults at a single site (inverter trip, soiling) appear as local anomalies.
Grid-level events (voltage sag, frequency event) or regional weather (storm front)
appear as correlated anomalies across multiple sites.

**Research direction:** Model inter-site residual correlation using a spatial covariance
function. Sites that are spatially close and share similar irradiance regimes should
have correlated residuals on cloudy days. A fleet-level detector that accounts for this
correlation would reduce false positives when a storm affects multiple sites simultaneously.

This connects to **spatiotemporal modeling**, **multi-task learning**, and potentially
**graph neural networks** over a site connectivity graph.

### 5. PJM grid context as a feature

**Problem:** The marginal value of a kWh of solar generation depends on the LMP
(locational marginal price) at the nearest PJM node. A fault that occurs during a
peak-price period has 5–10× higher economic impact than one during off-peak.

**Research direction:** Integrate PJM Data Miner LMP data as a feature in loss
estimation. Rather than reporting "23 kWh lost," report "23 kWh × $85/MWh = $2 lost
revenue." This makes fault severity economically interpretable and directly useful to
energy managers.

---

## Connections to active research

| GridGuard component | Research area | Example references |
|---|---|---|
| Conformal lower bound | Distribution-free uncertainty quantification | Angelopoulos & Bates (2022); Tibshirani et al. (2019) |
| Weather-only vs lag-aware | Causal feature selection for anomaly detection | Peters et al. (2017) *Elements of Causal Inference* |
| NSRDB irradiance features | Geospatial ML for renewable energy | Perez et al. (2013); NREL Solar Forecasting 2 |
| Fleet-level correlation | Spatiotemporal anomaly detection | Gupta et al. (2014); Laptev et al. (2015) |
| PJM LMP integration | Energy economics + ML | Weron (2014) electricity price forecasting |
| Temporal train/test split | Time series model evaluation | Bergmeir & Benítez (2012); Hyndman & Athanasopoulos (2021) |

---

## Concrete next steps for research collaboration

1. **Conformal intervals paper:** Extend conformal_bound to a full split-conformal
   anomaly detector with formal coverage proof. Compare against sigma thresholding on
   the injected fault dataset (known labels → precision/recall measurable).

2. **NSRDB ingestion + evaluation:** Connect `ingestion/download.py` to the NSRDB API,
   retrain models on real irradiance, and measure whether the feature improves MAPE
   on held-out test months (especially winter and storm days).

3. **Multi-site fleet study:** Deploy all 7 DMV sites with synthetic data, compute
   cross-site residual correlations, and test whether a spatially-aware detector reduces
   false positives during correlated cloudy periods.

4. **Economic loss estimation:** Integrate daily PJM LMP (free download from
   dataminer2.pjm.com) and compute `lost_revenue = lost_kWh × lmp_dollar_per_mwh`
   for each flagged event. Evaluate whether this changes the priority ranking of events
   vs. lost_kWh alone.

---

## Honest limitations for research context

- All data is currently synthetic. Claims about detection performance reflect
  simulated faults, not real inverter failures.
- The site registry uses approximate coordinates and capacities from public records.
- The conformal bound is implemented but not yet integrated into the production detection path.
- Single-site model only — no cross-site transfer learning.
- No streaming: batch pipeline only. Real-time fault detection would need a streaming layer.
