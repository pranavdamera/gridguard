# Modeling Design — GridGuard

This document explains the feature set design, leakage risks, model choices,
evaluation methodology, and anomaly detection approach.

---

## Feature sets

GridGuard defines two named feature modes in `src/gridguard/features/engineer.py`.

### `weather_only` — Expected-generation baseline

**Purpose:** Learn what a healthy system *should* produce given external conditions.
Used as the baseline for anomaly detection.

| Feature | Rationale |
|---|---|
| `irradiance_wm2` | Primary driver of solar output |
| `irradiance_sq` | Captures non-linear panel response at high/low irradiance |
| `irradiance_roll1h` | 1-hour rolling mean smooths cloud transients |
| `temperature_c` | Panel efficiency drops ~0.4%/°C above 25°C |
| `wind_speed_ms` | Convective cooling effect |
| `hour`, `hour_sin`, `hour_cos` | Time of day (cyclic encoding prevents 23→0 jump) |
| `day_of_year`, `doy_sin`, `doy_cos` | Seasonal solar geometry |
| `month` | Secondary seasonality |

**What is intentionally excluded:** `ac_power_lag1`, `ac_power_lag4`.

**Why excluding lags matters for anomaly detection:**

If a system has been degraded for several days (soiling, partial shading, inverter trip),
its recent output is already low. Including lag features trains the model to predict
"low output" as normal for that system. The expected-generation signal is lost.

The `weather_only` mode forces the model to answer:
> "Given this irradiance and temperature, what should a healthy 250 kW system produce?"

independently of what the system has been doing lately.

### `lag_aware` — Short-horizon operational forecast

**Purpose:** Predict the next 15-minute interval for operational purposes (e.g., dispatch
coordination, curtailment decisions).

Adds `ac_power_lag1` (previous 15-minute interval) and `ac_power_lag4` (1 hour ago)
to the weather_only feature set. These are strong predictors for cloud-ramp events and
intra-hour variability, but their use in anomaly detection baselines masks persistent faults.

**Default behavior:** The current training pipeline uses `lag_aware` for all models,
including the one used in anomaly detection. This is documented as a known limitation.
The `weather_only` infrastructure is in place; training a separate anomaly baseline
model is the next concrete step.

---

## Leakage risk

Temporal data has two forms of leakage:

**1. Train/test contamination.** Always use `train_test_split_by_date()` — never
`sklearn.model_selection.train_test_split`. Random splits allow the model to see
future data during training, inflating R² dramatically.

**2. Residual calibration contamination.** The anomaly detector calibrates its sigma
threshold using per-hour residual statistics. These *must* come from the training split
only (frozen after training). If you re-calibrate on the test period, a systematic
underperformance in that period will inflate sigma, making the threshold too permissive.

The `compute_and_save_residual_stats()` function enforces this: it is called from
`run_training()` on the training DataFrame only, and the result is saved to
`artifacts/models/residual_stats.json`. `detect_anomalies(freeze=True)` loads this file
and never recomputes from the evaluation data.

---

## Baselines

| Model | Key hyperparameters | Notes |
|---|---|---|
| Persistence | — | Predicts next = previous; strong baseline for solar (autocorrelated) |
| Ridge regression | α=1.0 | Linear, fast; useful for understanding feature contributions |
| RandomForest | 100 trees, default sklearn | Handles non-linear irradiance × temp interaction |
| XGBoost | 500 trees, η=0.05, max_depth=6, early stopping | Best performer; used in production path |

All models are trained on the same feature set (default: `lag_aware`) and compared
on the held-out test split using RMSE as the primary ranking metric.

---

## Evaluation metrics

| Metric | Formula | Why |
|---|---|---|
| MAE (kW) | mean\|actual - predicted\| | Interpretable: "off by X kW on average" |
| RMSE (kW) | √mean(actual - predicted)² | Penalises large errors; sensitive to fault spikes |
| MAPE (%) | mean\|error/actual\| on daylight only | % error; div-by-zero excluded at night |
| R² | 1 - SS_res/SS_tot | Fraction of variance explained |

**Stratified metrics** break these down by hour of day, season, and irradiance bin.
This reveals whether the model underperforms at dawn/dusk (common for linear models)
or in winter (when sun elevation is low and the model may not have seen enough examples).

---

## Anomaly detection

### Current: residual sigma thresholding

```
residual = actual_kw - predicted_kw
normalised = (residual - mean_hour) / std_hour     # per-hour bucket
flag if normalised < -k                            # default k = 2.0
```

Per-hour normalisation prevents false positives at peak irradiance (where absolute
residuals are naturally larger) and missed alarms at dawn/dusk.

**Limitation:** No formal coverage guarantee. The threshold k is a heuristic.

### Next step: conformal lower prediction bound

`src/gridguard/anomaly/conformal.py` implements the foundation:

```python
from gridguard.anomaly.conformal import compute_conformal_bound, flag_conformal_anomalies

# Compute bound from training-split residuals
bound = compute_conformal_bound(train_residuals, alpha=0.1)

# Flag evaluation intervals
flags = flag_conformal_anomalies(eval_residuals, train_residuals, alpha=0.1)
```

**Guarantee:** At miscoverage level α, at most ⌈(n+1)α⌉/n of calibration intervals
fall below the bound. For α=0.1, roughly 90% of non-fault intervals will *not* be
flagged, regardless of the residual distribution.

**Reference:** Angelopoulos & Bates (2022), *A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.

Integrating the conformal bound into the production `detect_anomalies()` path (replacing
or complementing sigma thresholding) is the next concrete ML improvement.

---

## What is synthetic vs real

All generation data in the default workflow is **synthetic** — produced by a
physics-motivated simulator in `ingestion/download.py`. The simulator models:

- Sun elevation angle at site latitude (simplified, no pvlib)
- Clear-sky GHI × log-normal cloud factor
- Temperature with diurnal + seasonal cycle calibrated to Northern Virginia
- Panel efficiency penalty: −0.4%/°C above 25°C
- Injected faults on ~5% of days (40–80% output reduction)

The synthetic data is good enough for model development and demo. It is not suitable
for yield studies, degradation analysis, or regulatory reporting. Real data paths
(NREL PVDAQ, NSRDB) are documented in `docs/data_sources.md`.
