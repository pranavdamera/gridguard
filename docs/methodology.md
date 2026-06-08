# GridGuard Methodology

## What this system does

GridGuard detects underperformance events in campus-scale photovoltaic (PV) systems by:

1. Generating or ingesting 15-minute solar telemetry (irradiance, temperature, AC power)
2. Training an expected-generation model using weather and time features only
3. Comparing actual vs expected generation to compute a per-interval residual
4. Flagging intervals where the residual exceeds a calibrated threshold
5. Grouping consecutive flagged intervals into human-readable events
6. Explaining each event using SHAP feature contributions
7. Serving all of this via a FastAPI backend and Next.js operator dashboard

---

## Data: what is real vs synthetic

**All generation data in the demo is synthetic deterministic telemetry. It is not real sensor data.**

The synthetic model (`src/gridguard/ingestion/download.py`) uses:

- **Sun geometry**: simplified declination + hour-angle model parameterised by latitude (38.8° N, Northern Virginia)
- **Clear-sky irradiance**: 1000 × cos(solar zenith angle)
- **Cloud attenuation**: log-normal multiplicative factor, clipped to [0, 1.5]
- **Temperature**: seasonal mean (5°C winter, 28°C summer) + diurnal swing ± 5°C + Gaussian noise calibrated to Northern Virginia climate
- **Panel efficiency**: −0.4%/°C degradation above 25°C (standard crystalline silicon temperature coefficient)
- **Random fault days**: ~5% of days have 40–80% output reduction (soiling, shading, inverter trips)
- **Scripted demo event**: June 15, 2023 09:00–12:15 — 70% AC power reduction. Deterministic, injected for demo purposes.

**Planned real-data adapters**: NREL PVDAQ API, NSRDB satellite irradiance, CSV upload from campus SCADA/metering, PJM interval data.

---

## The most important modeling decision: two feature modes

GridGuard enforces a strict separation between feature sets for two different modeling tasks.

### Weather-only features (for anomaly detection / expected-generation model)

```
hour, day_of_year, month, hour_sin, hour_cos, doy_sin, doy_cos,
irradiance_wm2, irradiance_sq, temperature_c, wind_speed_ms, irradiance_roll1h
```

**Used to answer**: "What should a healthy system produce given today's weather?"

**No lag features.** This is intentional and critical.

**Why no lags?** If a system has been degraded for hours, its recent AC power readings (lags) are low. A model trained with lag features will learn that "low lag = low expected output" and predict low power as normal — hiding persistent degradation entirely. Weather-only features force the model to always ask what a healthy system should produce, making persistent underperformance detectable.

### Lag-aware features (for operational forecasting)

```
[all weather-only features] + ac_power_lag1 + ac_power_lag4
```

**Used to answer**: "What will this system produce in the next 15 minutes?"

**Includes lag features** because recent output is a strong predictor of near-future output for short-horizon operational planning (e.g., cloud ramp prediction). **Not used for anomaly detection baselines.**

---

## Anomaly detection: residual thresholding with training-set calibration

1. Train the expected-generation model on the **training split only** (temporal split, never random)
2. Compute residuals (actual − predicted) on the training split
3. Fit per-hour-of-day (mean, std) from training residuals → save to `artifacts/models/residual_stats.json`
4. At inference: `residual_σ = (residual − mean_train_hour) / std_train_hour`
5. Flag as anomaly if `residual_σ < −threshold` during daylight (irradiance > 50 W/m²)
6. Default threshold: 2.0 σ (configurable via `ANOMALY_THRESHOLD_SIGMA`)

**Per-hour calibration** prevents false positives at peak irradiance (where absolute residuals are naturally larger) and misses at dawn/dusk.

**Next step**: Replace sigma thresholding with conformal prediction intervals for proper coverage guarantees. Implementation stub in `src/gridguard/anomaly/conformal.py`.

---

## Temporal train/test split

All models use a temporal split (default: 2023-01-01). Training data is 2022; test data is 2023. Random splits are never used — they leak future information into training and inflate apparent accuracy on time series.

---

## Event grouping

Consecutive anomalous intervals (gaps ≤ 20 minutes) are collapsed into a single event with:
- Start/end time
- Duration (minutes)
- Total estimated lost energy (kWh)
- Maximum residual sigma
- Severity tier (low < 1 kWh, medium 1–10 kWh, high > 10 kWh)
- Plain-English explanation

---

## Models compared

| Model | Type | Feature mode |
|---|---|---|
| Persistence | Carry last observation | N/A |
| Ridge | Linear, L2-regularized | Lag-aware (forecasting) |
| Random Forest | Ensemble decision trees | Lag-aware (forecasting) |
| XGBoost | Gradient-boosted trees | Lag-aware (forecasting) |

The expected-generation model used for **anomaly detection** is trained separately using weather-only features.

---

## Explainability

SHAP (SHapley Additive exPlanations) values are computed per anomaly interval using the best available model. For tree models (XGBoost, Random Forest) this uses TreeExplainer — exact Shapley values, not kernel approximations. The dashboard shows the top contributors sorted by absolute SHAP value, with direction (positive = increased predicted output, negative = decreased).

If SHAP is unavailable at runtime, the dashboard falls back to model feature importances, clearly labeled.

---

## What this is not

- Not a real-time system. All data is static demo telemetry regenerated on demand.
- Not connected to live sensor data (no SCADA, no live inverter feeds).
- Not production-grade (no auth, rate limiting, or database persistence in this MVP).
- Not claiming research-validated results on real installations.
