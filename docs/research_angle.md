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

## What is already built (and what it showed)

Three of the directions above are now implemented, and two of them produced
results worth reporting.

**Split conformal detection is the production default**, with Mondrian
calibration per irradiance bucket, replacing sigma thresholding (retained as a
benchmark). Measured coverage on held-out healthy intervals is 0.962–0.991 on the
three real arrays against a 0.95 target. Two findings emerged that are not
obvious from the literature alone: calibrating on in-sample residuals silently
breaks the guarantee (bounds too tight), and coverage degrades measurably as
calibration data moves further from the evaluated period in time — the
exchangeability assumption failing under performance drift, visibly rather than
theoretically. Two simulated sites still sit below target, which is that
limitation on display.

**Real measured data is integrated** — three NIST arrays via the OEDI data lake,
paired with weather measured at the array itself, so no reanalysis or satellite
product is needed and the real-data path requires no credentials at all.

**Spatial attribution is implemented** across a ten-site fleet: capacity-normalised
residuals, haversine neighbourhoods, and a site-specific-versus-regional verdict
that reports its evidence and declines to guess below two neighbours. On real
data it correctly localised a genuine NIST ground-array outage as site-specific,
because the two neighbouring arrays 400 m away were producing normally.

---

## Concrete next steps for research collaboration

1. **Conformal under distribution shift.** The drift-induced coverage degradation
   above is measurable and reproducible in this repository, which makes it a
   ready-made testbed for weighted or adaptive conformal methods (Tibshirani et
   al., 2019; Barber et al., 2023; Gibbs & Candès, 2021). The question is
   concrete: can adaptive conformal restore the guarantee on the two sites where
   drift currently breaks it, without widening the bound so far that recall
   collapses?

2. **Gradual degradation detection.** Currently the weakest result by a wide
   margin — ~35% interval recall against ~100% for complete outages — and also
   the most economically consequential, since slow decline is what actually
   erodes lifetime yield. A residual-trend or change-point formulation over weeks
   rather than intervals is the obvious framing.

3. **Sensor-fault discrimination.** GridGuard reliably false-alarms on a frozen
   irradiance sensor (up to 0.59 false-alarm rate on that injected class): the
   input is wrong, so the expectation is wrong, and the array looks broken when
   the instrument is. Distinguishing input faults from output faults — perhaps by
   cross-checking irradiance against neighbours or a clear-sky model — is a
   well-posed problem with an immediately useful answer.

4. **Multi-year, multi-climate validation.** One year at one site cluster is the
   single biggest limitation on every claim in this repository. PVDAQ has many
   more systems; the ingestion layer already generalises to them.

5. **Event-level validation against maintenance records.** Every detection number
   here is against *injected* faults. Nothing has been confirmed against a real
   O&M log. Any dataset pairing telemetry with maintenance tickets would convert
   this from a methodology demonstration into evidence.

6. **Economic loss estimation.** Daily PJM LMP (dataminer2.pjm.com) would allow
   `lost_revenue = lost_kWh × price`, and the interesting question is whether it
   reorders event priority relative to lost kWh alone. Deliberately not
   implemented yet: GridGuard will not fabricate prices, and this needs real
   market data to be worth anything.

---

## Honest limitations for research context

- **Detection performance is measured against injected faults**, not observed
  field failures. No alert has been confirmed against a maintenance record.
- **Interval-level recall on real data is ~0.5**, against 5/5 event-level recall.
  GridGuard notices problems reliably and characterises their extent poorly.
- **One year, one site cluster, one climate.** No cross-year or cross-climate
  generalisation is demonstrated.
- **Conformal coverage degrades under drift** — observed on two sites, not
  hypothetical.
- **Sensor dropout produces confident false alarms.**
- The simulated fleet's site registry uses approximate coordinates and capacities
  from public records, and its institution names are illustrative only.
- **No streaming.** Batch evaluation only; live detection needs a streaming layer.
- Spatial attribution is not a fault classifier and does not identify components.
