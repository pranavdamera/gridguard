# ⚡ GridGuard DMV — Solar Generation Forecasting & Underperformance Detection

> Open-source ML system for forecasting solar generation and detecting underperformance
> in DC/Northern Virginia campus-style solar assets.

---

## What is this

GridGuard DMV forecasts expected solar output for campus-scale PV systems, flags
intervals where actual generation falls significantly below the model's prediction,
and groups those intervals into human-readable underperformance events.

The core detection loop:

```
Weather + Time → Forecast expected output → Compare vs actual → Alert if delta > 2σ
```

The system ships with a site registry for seven illustrative DMV institutions
(GMU Fairfax, NOVA campuses, DC Community Solar). Coordinates and capacities are
approximate estimates from public records. **All data in the default workflow is
synthetic** — physically simulated, not measured telemetry. Connect a real inverter
or SCADA feed to use real data.

---

## Why it matters

Campus energy managers and solar O&M teams need to know:
- **Did the system underperform today, and how much energy was lost?**
- **Is the deviation a real fault or just a cloudy afternoon?**
- **Which site in the fleet needs attention first?**

GridGuard addresses this by learning what a *healthy* system should produce given the weather, then flagging deviations that exceed a calibrated threshold. The pipeline is designed to connect to real inverter telemetry (SCADA, NREL PVDAQ, or a simple CSV) without structural changes.

Relevant real-world contexts:
- **Campus energy management** — universities tracking solar contribution to net-zero goals
- **Solar O&M** — fault triage and lost revenue estimation across a distributed fleet
- **DER visibility** — aggregating rooftop solar for grid operators who need generation visibility
- **Energy resilience** — detecting degradation before it becomes a critical outage

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                 Data Layer                  │
│  NREL PVDAQ API  ──or──  Synthetic Generator│
│       (15-min intervals, site-aware)        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            Feature Engineering              │
│  Time cyclic · Irradiance² · Lag-1 · Roll  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│             Model Training                  │
│  Persistence · Linear · RandomForest · XGB  │
│           (temporal split)                  │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴───────────┐
        │                      │
┌───────▼───────┐    ┌─────────▼────────┐
│   Anomaly     │    │  Explainability  │
│  Detection    │    │  (SHAP values)   │
│ (residual σ)  │    └──────────────────┘
└───────┬───────┘
        │
┌───────▼───────────────────────────────────┐
│           FastAPI Backend                  │
│  /health  /forecast  /anomalies  /events   │
│  /explain  /metrics                        │
└───────┬───────────────────────────────────┘
        │
┌───────▼───────────────────────────────────┐
│         Streamlit Dashboard                │
│  Actual vs Predicted · Anomaly timeline   │
│  Daily loss · Event explanations          │
│  Model comparison · SHAP · Forecast panel │
└───────────────────────────────────────────┘
```

---

## DMV site registry

GridGuard ships with seven Northern Virginia / DC campus sites:

```
GMU Fairfax · NOVA Annandale · NOVA Alexandria · NOVA Loudoun
NOVA Manassas · NOVA Woodbridge · DC Community Solar
```

Sites are defined in [config/sites.csv](config/sites.csv). Each has a latitude
that the synthetic generator uses to compute the correct sun elevation angle and
seasonal irradiance profile.

```bash
# List all sites
python scripts/download_data.py --list-sites

# Generate site-specific synthetic data
python scripts/download_data.py --source synthetic --site-id gmu_fairfax
python scripts/download_data.py --source synthetic --site-id nova_loudoun
```

**Note:** Site capacities and coordinates are approximate values from public records.
Synthetic data is a physics simulation, not real measured output from these campuses.

---

## What is real vs simulated

| Component                       | Status                                                           |
| ------------------------------- | ---------------------------------------------------------------- |
| Site IDs, names, region         | Real institution names (illustrative only)                       |
| Coordinates, capacity_kw        | Approximate estimates from public records                        |
| Solar generation data           | **Synthetic** — physics simulation, not measured                 |
| Sun geometry, seasonal swing    | Physically motivated (latitude-parameterised)                    |
| Fault events                    | **Injected** at ~5% of training days with known labels           |
| GMU demo underperformance event | **Scripted** — see [Deterministic demo](#deterministic-gmu-demo) |
| NREL PVDAQ option               | Real data (requires free API key)                                |

---

## Repo structure

```
gridguard/
├── config/
│   └── sites.csv              # DMV site registry (lat, lon, capacity)
├── src/gridguard/
│   ├── config.py              # Pydantic settings (loaded from .env)
│   ├── sites/
│   │   └── registry.py        # Site loader and Site dataclass
│   ├── ingestion/
│   │   └── download.py        # NREL PVDAQ + site-aware synthetic generator
│   ├── features/
│   │   └── engineer.py        # Feature engineering, temporal split
│   ├── models/
│   │   ├── baseline.py        # Persistence, Linear, RF, XGBoost
│   │   ├── train.py           # Training pipeline + residual calibration artifact
│   │   └── evaluate.py        # Overall + stratified metrics; report saving
│   ├── anomaly/
│   │   ├── detect.py          # Residual-based anomaly detection (frozen calibration)
│   │   └── events.py          # Group consecutive intervals into events + explain
│   ├── explainability/
│   │   └── shap_explain.py    # SHAP global + per-alert explanation
│   └── api/
│       ├── main.py            # FastAPI app
│       └── schemas.py         # Pydantic request/response schemas
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── scripts/
│   ├── download_data.py       # CLI: --source, --site-id, --demo, --list-sites
│   └── run_pipeline.py        # CLI: end-to-end training pipeline (--site-id)
├── tests/
│   └── ...                    # pytest suite
├── docs/
│   ├── architecture.md
│   ├── data_sources.md
│   ├── model_card.md
│   └── demo_script.md         # 2-3 minute MVP demo walkthrough
└── ...
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/pranav-damera/gridguard.git
cd gridguard
python -m venv .venv && source .venv/bin/activate
make install

# Also install frontend dependencies
cd web && npm install && cd ..

# 2. Copy env files
cp .env.example .env
cp web/.env.example web/.env.local

# 3. Reset demo — generates all artifacts deterministically (run once)
make demo-reset

# 4. Start backend (terminal 1)
make api
# → http://localhost:8000/docs

# 5. Start Next.js frontend (terminal 2)
make web
# → http://localhost:3000

# 6. (Optional) Start Streamlit research dashboard (terminal 3)
make dashboard
# → http://localhost:8501
```

---

## Repo structure

```
gridguard/
├── web/                         # Next.js public frontend (TypeScript + Tailwind)
│   ├── app/
│   │   ├── page.tsx             # Landing page
│   │   ├── demo/page.tsx        # Main demo dashboard
│   │   ├── sites/[siteId]/      # Site detail
│   │   ├── events/[eventId]/    # Event drilldown
│   │   └── methodology/         # Transparent model docs
│   ├── components/              # Reusable dashboard components
│   └── lib/api.ts               # API client wrapper
├── scripts/
│   ├── reset_demo.py            # One-step deterministic demo reset
│   ├── download_data.py         # CLI: --source, --site-id, --demo
│   └── run_pipeline.py          # CLI: end-to-end training pipeline
├── src/gridguard/
│   └── api/main.py              # FastAPI — /health /sites /events /demo/scenario …
├── dashboard/app.py             # Streamlit research dashboard
├── docs/
│   ├── methodology.md           # Feature modes, leakage, anomaly detection
│   ├── deployment.md            # Vercel + Render/Railway deployment
│   └── demo_script.md           # 2-min demo walkthrough
└── tests/                       # 92 pytest tests
```

---

## GMU Fairfax demo workflow

---

## Deterministic GMU demo

The `--demo` flag generates a **repeatable** underperformance scenario with a
known daylight fault window, suitable for demo videos.

```bash
python scripts/download_data.py --source synthetic --site-id gmu_fairfax --demo
python scripts/run_pipeline.py --site-id gmu_fairfax --demo
```

**Scripted underperformance event:**

- **Site:** GMU Fairfax (250 kW, 38.83°N)
- **Date:** 2023-06-15 (clear summer day)
- **Window:** 09:00 – 12:15 local time (13 intervals, ~3.25 hours)
- **Effect:** ~70% generation reduction during peak irradiance hours
- **Cache file:** `data/processed/raw_gmu_fairfax_demo.parquet`

This event is **scripted for demonstration purposes** — it does not represent
a real fault at GMU. After running the demo pipeline, look for this event in the
Anomaly Events table with `severity: high`.

---

## Commands

| Command                                                        | Description                                           |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| `make install`                                                 | Install all dependencies                              |
| `make data-synthetic`                                          | Generate 2 years of generic synthetic solar data      |
| `make data-gmu`                                                | Generate site-specific synthetic data for GMU Fairfax |
| `make data-nrel`                                               | Download real NREL PVDAQ data (requires API key)      |
| `make train`                                                   | Run full training pipeline (generic data)             |
| `make train-gmu`                                               | Train models using GMU Fairfax synthetic data         |
| `make api`                                                     | Start FastAPI server                                  |
| `make dashboard`                                               | Start Streamlit dashboard                             |
| `make test`                                                    | Run pytest suite with coverage                        |
| `make lint`                                                    | Run ruff + black check                                |
| `make format`                                                  | Auto-fix lint issues                                  |
| `make docker-up`                                               | Start API + dashboard via Docker                      |
| `python scripts/download_data.py --list-sites`                 | Print the DMV site registry                           |
| `python scripts/download_data.py --site-id gmu_fairfax`        | Generate site-specific data                           |
| `python scripts/download_data.py --site-id gmu_fairfax --demo` | Generate deterministic demo data                      |

---

## API reference

After `make api`, docs are at http://localhost:8000/docs

**GET /health**

```json
{ "status": "ok", "model_loaded": true, "data_rows": 17520, "version": "0.1.0" }
```

**POST /forecast**

```json
{
  "timestamp": "2023-07-15T14:00:00",
  "irradiance_wm2": 820,
  "temperature_c": 31,
  "wind_speed_ms": 2.5
}
```

```json
{ "timestamp": "...", "predicted_kw": 7.84, "model_name": "xgboost" }
```

**GET /events?limit=20**

```json
{
  "total_events": 18,
  "total_lost_kwh": 48.7,
  "events": [
    {
      "event_id": 1,
      "start_time": "2023-06-15T09:00:00",
      "end_time": "2023-06-15T12:15:00",
      "duration_minutes": 210,
      "total_lost_kwh": 23.6,
      "severity": "high",
      "explanation": "During this 3h 30min event, the model expected 45.2 kW average output but actual generation was 13.5 kW — 70% below forecast. Estimated lost energy: 23.6 kWh. Severity: high."
    }
  ]
}
```

**GET /anomalies?limit=50**, **GET /metrics**, **GET /explain?timestamp=...**
— see http://localhost:8000/docs

---

## ML approach

### Two feature modes

GridGuard distinguishes two feature sets for different modelling goals:

| Mode | Purpose | Allowed features |
|---|---|---|
| `weather_only` | Expected-generation baseline / anomaly detection | Weather, time, site metadata. No actual-power lags. |
| `lag_aware` | Short-horizon operational forecasting | All weather + time features + `ac_power_lag1`, `ac_power_lag4` |

**Why the distinction matters:** If you train an anomaly detector with lag features, a system that has been degraded for days produces low power → lag₁ is low → the model learns to predict low → the degradation is invisible. The `weather_only` mode forces the model to answer "what should a healthy system produce given today's weather?" independently of recent observed output.

```python
# Weather-only model for anomaly detection (avoids masking persistent faults)
from gridguard.features.engineer import get_X_y
X_train, y_train = get_X_y(train_df, mode="weather_only")

# Lag-aware model for next-interval operational forecast
X_train, y_train = get_X_y(train_df, mode="lag_aware")
```

### Models

| Model | Notes |
|---|---|
| Persistence (lag-1) | Naive baseline — predicts next interval = previous interval |
| Linear (Ridge) | Regularised linear baseline on all features |
| RandomForest | Ensemble, handles non-linear irradiance × temperature interactions |
| XGBoost | Best performer; used for anomaly detection and API by default |

All models use a **temporal train/test split** (not random). Random splits leak future information into training, inflating R² artificially.

### Anomaly detection

The default detector uses **residual sigma thresholding** calibrated on training data:

```
flag if:  (actual - predicted) < -k × σ_hour
```

where `σ_hour` is the per-hour residual standard deviation estimated from the training split only. Using training-set stats prevents test-period degradation from contaminating the threshold.

A **conformal lower prediction bound** is implemented in `anomaly/conformal.py` as the principled next step. It provides a coverage guarantee: at miscoverage level α, at most α × n+1 calibration points fall below the bound. See `docs/modeling.md` for details.

### Explainability

SHAP values are computed for the XGBoost model, attributing each prediction to specific features. The `/explain` API endpoint returns the top contributors for any flagged interval.

See [docs/modeling.md](docs/modeling.md) for full model design rationale.

---

## Limitations

This is a portfolio and learning project. Be explicit about what it is not:

- **All data is synthetic** — Real sensor noise, soiling ramp, inverter clipping,
  and row-to-row shading are not modelled. Retrain on real data before drawing conclusions.
- **Site data is illustrative** — GMU/NOVA/DC capacities and coordinates are
  approximate estimates from public records, not official specifications.
- **No streaming** — Batch pipeline only. Live fault detection needs a streaming layer.
- **Residual thresholding** — No coverage guarantee. Conformal prediction intervals
  are the right next step.
- **Single-site model** — No cross-site transfer or fleet-normalised features.
- **Not safety-critical** — Do not use as the sole basis for dispatch, maintenance,
  or financial decisions without independent validation.

See [docs/model_card.md](docs/model_card.md) for the full model card.

---

## How to run tests

```bash
# From repo root (no PYTHONPATH required — pytest.ini sets pythonpath = ["src"])
pytest

# With coverage
pytest --cov=gridguard --cov-report=term-missing
```

---

## Roadmap

**Real data ingestion**
- [ ] NSRDB integration — real irradiance/weather from NREL's National Solar Radiation Database (`/api/solar/nsrdb_psm3`)
- [ ] PJM Data Miner integration — regional LMP and dispatch signals as grid-context features
- [ ] PVDAQ/OEDI ingestion — real PV generation records from NREL Open Energy Data Initiative

**ML improvements**
- [ ] Conformal prediction intervals — replace sigma thresholding with coverage-guaranteed bounds (`anomaly/conformal.py` foundation is ready)
- [ ] Weather-only model as default anomaly baseline — train separate XGBoost on `mode="weather_only"` features
- [ ] Event-level precision/recall evaluation — need labeled fault datasets
- [ ] pvlib clear-sky model — physics-based GHI as a feature (currently using simplified sun geometry)
- [ ] PyTorch LSTM baseline — sequence model on sliding windows
- [ ] Degradation trend detection — slow long-term output decline via residual trend

**System / demo**
- [ ] Multi-site fleet aggregation — cross-site metrics and correlation
- [ ] Live data mode — poll a real inverter API, run forecast + anomaly in real time
- [ ] Email/Slack alerts — notify when high-severity event detected
- [ ] pvlib integration — replace custom sun geometry with well-tested pvlib computations

See [docs/research_angle.md](docs/research_angle.md) for the research collaboration context.

---

## Stack

| Layer          | Library                          |
| -------------- | -------------------------------- |
| Data           | pandas, numpy, pyarrow           |
| ML             | scikit-learn, xgboost            |
| Explainability | shap                             |
| API            | fastapi, uvicorn, pydantic       |
| Dashboard      | streamlit, plotly                |
| Config         | pydantic-settings, python-dotenv |
| Infra          | Docker, docker-compose           |
| Dev            | pytest, ruff, black, make        |

---

## License

MIT
