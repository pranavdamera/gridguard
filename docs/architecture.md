# GridGuard Architecture

## Overview

GridGuard is a batch ML pipeline with a REST API serving layer and a Streamlit
dashboard. It is designed to be runnable on a laptop with no cloud dependencies.

```
┌──────────────────────────────────────────────────────────────┐
│                        Data Sources                          │
│   NREL PVDAQ API  ──or──  Synthetic Generator (site-aware)   │
│                   config/sites.csv (DMV site registry)       │
└─────────────────────────┬────────────────────────────────────┘
                          │ 15-min interval DataFrames
┌─────────────────────────▼────────────────────────────────────┐
│                   Feature Engineering                         │
│  src/gridguard/features/engineer.py                          │
│                                                              │
│  Time: hour_sin/cos, doy_sin/cos, month                      │
│  Weather: irradiance_wm2, irradiance_sq, temperature_c       │
│  Lags: ac_power_lag1 (15 min), ac_power_lag4 (1 hr)          │
│  Rolling: irradiance_roll1h                                   │
└─────────────────────────┬────────────────────────────────────┘
                          │ X_train / X_test
                          │ TEMPORAL SPLIT (no leakage)
┌─────────────────────────▼────────────────────────────────────┐
│                    Model Training                             │
│  src/gridguard/models/train.py                               │
│                                                              │
│  Persistence → Ridge(LinearRegression) → RandomForest         │
│  → XGBoost (early stopping on test eval_set)                  │
│                                                              │
│  Artifacts:                                                   │
│    artifacts/models/{name}.pkl                               │
│    artifacts/models/residual_stats.json   ← calibration      │
│    data/processed/test_df.parquet                            │
└────────────────────────┬─────────────────────────────────────┘
                         │ fitted model + residual_stats.json
┌────────────────────────▼─────────────────────────────────────┐
│                   Anomaly Detection                           │
│  src/gridguard/anomaly/detect.py                             │
│                                                              │
│  1. Predict expected generation with best model              │
│  2. residual = actual − predicted                            │
│  3. Normalise by per-hour σ from TRAINING data only          │
│  4. Flag if residual_sigma < −k (configurable threshold)     │
│  5. Compute lost_energy_kwh per flagged interval             │
│                                                              │
│  Event grouping  src/gridguard/anomaly/events.py             │
│  Consecutive flagged intervals → events (start/end/severity) │
└────────────────────────┬─────────────────────────────────────┘
                         │ anomaly_df, events_df
         ┌───────────────┤
         │               │
┌────────▼───────┐  ┌────▼─────────────────────────────────────┐
│ Explainability │  │         FastAPI Backend                   │
│ shap_explain.py│  │  src/gridguard/api/main.py               │
│                │  │                                           │
│ TreeExplainer  │  │  GET  /health                            │
│ per-alert SHAP │  │  POST /forecast                          │
│ waterfall      │  │  GET  /anomalies                         │
└────────────────┘  │  GET  /events                            │
                    │  GET  /explain?timestamp=…               │
                    │  GET  /metrics                           │
                    └────┬─────────────────────────────────────┘
                         │ HTTP (localhost:8000)
                    ┌────▼─────────────────────────────────────┐
                    │       Streamlit Dashboard                  │
                    │  dashboard/app.py                         │
                    │                                           │
                    │  KPI row · Actual vs Predicted plot       │
                    │  Daily loss · Events table                │
                    │  Stratified metrics · SHAP explain panel  │
                    │  Weather drivers scatter                  │
                    └──────────────────────────────────────────┘
```

## Key design decisions

### Temporal split, not random split
All train/test splits are by date. Random splits on time-series data leak
future information into the training set, producing over-optimistic metrics.

### Residual calibration from training data only
The per-hour σ used to normalise anomaly residuals is computed from the TRAINING
split and saved as `residual_stats.json`. `detect_anomalies()` loads this frozen
calibration at inference time. If calibration were computed from the test/live data
being evaluated, the threshold would adapt to that data's characteristics — masking
true anomalies or generating spurious alerts depending on how unusual the period is.

### Modular model zoo
All models implement the sklearn `.fit(X, y)` / `.predict(X)` interface.
This makes adding a new model (LightGBM, LSTM) a one-file change in `baseline.py`.

### Graceful SHAP degradation
If `shap` is not installed or the model is not XGBoost-compatible, the API and
dashboard fall back to XGBoost built-in feature importances. The response schema
includes `shap_available: bool` so callers can distinguish.

## Data flow

```
config/sites.csv
    ↓ registry.py
ingestion/download.py → data/processed/raw_{site_id}.parquet
    ↓ feature engineering
features/engineer.py → X_train, X_test, y_train, y_test
    ↓ training
models/train.py → artifacts/models/*.pkl + residual_stats.json
    ↓ anomaly detection (frozen calibration)
anomaly/detect.py → anomaly_df
    ↓ event grouping
anomaly/events.py → events_df
    ↓ API + dashboard
api/main.py + dashboard/app.py
```

## Limitations

- **Synthetic data** — The default dataset is simulated, not real. Real PV systems
  have additional noise sources (shading geometry, soiling rates, inverter clipping)
  that the simulator does not model.
- **No streaming** — The pipeline is batch. Live fault detection would require a
  streaming ingestion layer (Kafka, MQTT) and a sliding-window inference loop.
- **Single-site model** — The current model trains on pooled data. A multi-site
  model would need site-specific normalisation and possibly a hierarchical architecture.
- **Residual thresholding** — Statistical thresholding has no coverage guarantee.
  A conformal prediction interval would provide well-calibrated alert rates.
- **No ground-truth labels** — For synthetic data we inject faults and can evaluate
  detection rate. For real data, manual labelling would be needed to measure precision/recall.
