# GridGuard DMV — Demo Script

A 2–3 minute walkthrough for the GMU Fairfax underperformance scenario.
Suitable for demo videos, recruiter calls, or peer reviews.

**Honest framing:** All data shown is synthetic. GMU Fairfax is an illustrative site
(lat 38.83°N, 250 kW nameplate). The underperformance event on 2023-06-15 is scripted.

---

## Setup (run before the demo)

```bash
make install
cp .env.example .env

# Generate GMU Fairfax demo data (deterministic, ~3 seconds)
python scripts/download_data.py --source synthetic --site-id gmu_fairfax --demo

# Train models (~60–90 seconds)
python scripts/run_pipeline.py --site-id gmu_fairfax --demo

# Terminal 1 — API
make api

# Terminal 2 — Dashboard
make dashboard
```

Open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Demo flow (~2.5 minutes)

### 0:00 — One-sentence framing (15 seconds)

> "GridGuard DMV is an open-source ML system that forecasts expected solar output
> for campus-scale PV systems, detects underperformance, and explains the events
> in plain English. I'll show it on a scripted GMU Fairfax scenario."

---

### 0:15 — KPI row (20 seconds)

Point to the header metrics:

- **Anomaly events** — grouped continuous underperformance periods
- **Lost energy (kWh)** — modelled energy that should have been generated
- **Best RMSE** — model accuracy against withheld test data

> "The key number here is lost energy — that's what an O&M team would use to
> prioritise which sites to dispatch a crew to."

---

### 0:35 — Anomaly Events table (45 seconds)

Scroll to the **Anomaly Events** table:

1. Find the row for **2023-06-15** — this is the scripted underperformance event
2. Point to the **Severity: high** badge
3. Read the **Explanation** column aloud:

> "During this 3h 15min event on 2023-06-15, the model expected 45 kW average
> output but actual generation was 13 kW — 70% below forecast.
> Estimated lost energy: ~23 kWh. Severity: high."

> "The explanation is auto-generated from the model's residuals — no manual
> annotation needed."

---

### 1:20 — Actual vs Predicted chart (30 seconds)

Scroll to **Actual vs Predicted Generation**:

> "The X markers are flagged intervals. You can see the model tracked actual
> output closely on clear days, but when the fault hit on June 15th the actual
> line drops sharply while the forecast stays up — that gap is the lost energy."

---

### 1:50 — Next-Day Forecast panel (20 seconds)

Scroll to **Next-Day Solar Forecast**:

> "This panel shows a clear-sky upper-bound forecast for the next 24 hours.
> It's not weather-aware yet — that would come from NSRDB or a weather API —
> but it gives the operator a baseline to compare against when real readings
> start arriving."

---

### 2:10 — API (20 seconds)

Switch to http://localhost:8000/docs:

1. **GET /events** → expand a `severity: "high"` record → point to `explanation` field
2. **GET /health** → show `residual_calibration_loaded: true`

> "Every event the API returns includes a human-readable explanation string —
> same content as the dashboard, consumable by any downstream notification system."

---

### 2:30 — Wrap-up (30 seconds)

> "The full pipeline runs with `make data-gmu && make train-gmu`. Adding a new
> site is one row in `config/sites.csv`. The next steps for a real deployment
> would be connecting real inverter telemetry, adding NSRDB irradiance, and
> swapping the residual threshold for conformal prediction intervals to get
> a coverage guarantee on the alert rate."

---

## Key talking points (if asked to go deeper)

**Why temporal split not random?**
Random splits let future data leak into training, inflating R² artificially.

**Why per-hour σ calibration?**
Residuals are 5× larger at noon than at dawn. Normalising by hour means
a morning fault doesn't need to match a midday threshold.

**Why calibration from training data only?**
Same principle as fitting a StandardScaler on training data — calibrating
on the test period would mask systematic underperformance in that window.

**Is this production-ready?**
No — explicitly noted in the model card and README. Main gaps: synthetic data,
no streaming ingestion, single-site model, no coverage-guaranteed intervals.
Each is a concrete next step, not a hand-wave.

**Demo event timestamp:** 2023-06-15 09:00 – 12:15 (scripted, 70% reduction)
