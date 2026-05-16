# Demo Script

A guided walkthrough for showing GridGuard to recruiters, reviewers, or teammates.
Estimated time: 8–10 minutes.

---

## Before you start

```bash
# Make sure everything is built and running
make install
cp .env.example .env
make data-synthetic      # ~5 seconds
make train               # ~60–90 seconds

# Terminal 1 — API
make api

# Terminal 2 — Dashboard
make dashboard
```

Open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Talking points

### 1. The problem (30 seconds)

> "Solar panels can lose 10–30% of annual output to undetected faults — soiling,
> partial shading, inverter failures. Utilities and O&M teams often find out weeks
> later from quarterly reports. GridGuard flags underperformance within 15 minutes."

### 2. Architecture (1 minute)

Point to the dashboard and walk through:
- **What the model sees**: weather + time features, not the actual output
- **What it predicts**: expected generation for this site at this moment
- **How anomalies are flagged**: actual < predicted by > 2σ (training-calibrated)
- **What an event is**: consecutive flagged intervals grouped together

### 3. Show the dashboard (3 minutes)

1. Point to the **KPI row** — highlight "anomaly events" vs "anomaly intervals"
2. Scroll to **Actual vs Predicted** chart — point out the X markers on anomaly intervals
3. Show the **Daily Energy Loss** bar chart — explain the colour is % loss relative to expected
4. Open **Anomaly Events table** — explain severity tiers (kWh threshold), one row per event
5. Show **Stratified Metrics** — change from "hour_of_day" to "season" to show seasonal gap
6. Select an anomalous timestamp in the **Explain panel** — walk through SHAP bar chart

### 4. Show the API (2 minutes)

In the Swagger UI (http://localhost:8000/docs):

1. **GET /health** — show `residual_calibration_loaded: true`
2. **GET /events** — paste the response, point to `severity: "high"` rows
3. **POST /forecast** — submit a request:
   ```json
   {
     "timestamp": "2023-07-15T13:00:00",
     "irradiance_wm2": 850,
     "temperature_c": 32,
     "wind_speed_ms": 2.0
   }
   ```
4. **GET /explain** — paste a timestamp from an anomalous interval

### 5. Model comparison (1 minute)

Show the Streamlit model comparison table. Key point:

> "Any model that doesn't beat the persistence baseline — 'predict the same as
> the last measurement' — isn't worth deploying. XGBoost gets R²=0.98 on this
> dataset; persistence gets ~0.85."

### 6. Site registry (1 minute)

```bash
python scripts/download_data.py --list-sites
```

> "The site registry in `config/sites.csv` means adding a new site is just
> adding a row. Each site gets its own cache file and latitude-adjusted synthetic
> data. In production you'd pull from the real inverter API."

### 7. Engineering decisions worth highlighting

If asked to go deeper:

- **Why temporal split not random?** — Random splits on time series let future
  data leak into training. The model would appear to predict things it already
  "saw", inflating R² artificially.

- **Why per-hour σ calibration?** — Absolute residuals are 5× larger at noon
  than at dawn. Without normalisation, the threshold catches noisy midday intervals
  and misses quiet morning failures.

- **Why is calibration computed from training data only?** — If we calibrated
  on the test period, the threshold would adapt to that period's characteristics,
  masking systematic underperformance. This is the same principle as fitting a
  StandardScaler on training data only.

- **SHAP vs feature importance** — XGBoost's built-in importance counts splits,
  which is biased toward high-cardinality features. SHAP values are consistent,
  sum to the prediction, and give per-interval explanations, not just global ranks.

---

## Questions to be ready for

**"Is this production-ready?"**
> No — and I'm explicit about that in the model card and README. The main gaps
> are: trained on synthetic data, no streaming ingestion, single-site, no
> coverage-guaranteed prediction intervals. Each of these is a concrete next step.

**"How would you handle real-time data?"**
> Replace the batch pipeline with a streaming loop: an MQTT/REST listener ingests
> inverter data every 15 minutes, calls `/forecast`, computes the residual, and
> calls `/events` to update the event log. The model itself doesn't change.

**"How do you know the anomaly detector is working?"**
> On synthetic data, I inject faults on 5% of days and measure recall. The test
> in `test_anomaly.py::test_injected_faults_detected` asserts that fault days
> have a higher anomaly rate than normal days. For real data you'd need labelled
> fault events from maintenance logs.
