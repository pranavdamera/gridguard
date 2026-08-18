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

**Limitation:** No formal coverage guarantee. The threshold *k* is a heuristic, and
it is badly miscalibrated on heavy-tailed residuals — outliers inflate the σ that
defines the threshold, so a "2σ" rule can cover ~99% instead of the ~95% that
choice of *k* implies, silently missing faults. The sigma detector is retained
only as a benchmark.

### Conformal lower prediction bound (production default)

Split conformal prediction replaced sigma thresholding as the default detector.
`src/gridguard/anomaly/conformal.py` implements it with Mondrian (per-irradiance-
bucket) calibration.

```python
from gridguard.anomaly.conformal import ConformalCalibration

calibration = ConformalCalibration.fit(
    residuals=held_out_residuals,      # NOT training residuals — see below
    irradiance_wm2=held_out_irradiance,
    alpha=0.05,
)
lower_bound = calibration.lower_bound_for(irradiance_wm2=820)
coverage = calibration.coverage_on(test_residuals, test_irradiance)
```

**Guarantee.** Given calibration residuals exchangeable with future residuals,
`P(actual − predicted ≥ bound) ≥ 1 − α`, with no distributional assumption. At
α = 0.05 that means at most ~5% of healthy intervals should breach the bound —
a design parameter, not a hope.

**Two implementation details that turned out to matter:**

1. *Calibrate on held-out data.* Calibrating on the model's own training
   residuals produces bounds that are too tight, because in-sample errors are
   biased small. Coverage sat below the stated guarantee until this was fixed.
2. *Calibrate on data close in time.* Array performance drifts, so
   exchangeability decays with temporal separation. Calibration uses the most
   recent healthy validation intervals; widening the validation window (which
   increases separation) measurably *degraded* coverage.

**Measured coverage** on held-out healthy intervals: 0.962 / 0.989 / 0.991 on the
three real arrays against a 0.95 target. Two simulated sites currently sit below
target (0.86, 0.90) — the exchangeability limitation manifesting under modelled
drift, reported per-site rather than tuned away.

**References.** Angelopoulos & Bates (2023), *Conformal Prediction: A Gentle
Introduction*, arXiv:2107.07511. Vovk et al. (2005), *Algorithmic Learning in a
Random World* — Mondrian conformal prediction, ch. 4.

---

## What is measured vs simulated

GridGuard runs on both, and never blurs them.

**Measured** — three NIST arrays in Gaithersburg MD, a full year of 15-minute
telemetry from NREL PVDAQ via the OEDI data lake, with irradiance, temperature
and wind from instruments at the array. **Forecast accuracy is reported on this
data.**

**Simulated** — seven illustrative DMV campus sites from a pvlib-based generator
with autocorrelated cloud attenuation and a drifting performance factor whose
persistence is calibrated against the real arrays' own autocorrelation.

The split of responsibilities is deliberate: measured telemetry has no fault
labels, so **detection accuracy** can only be measured against injected faults —
which are injected into the measured data too, keeping generation and weather
real while only the failures are simulated.

Simulated data remains unsuitable for yield studies, degradation analysis, or
regulatory reporting. Full provenance in [data_sources.md](data_sources.md).
