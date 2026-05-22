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
git clone https://github.com/YOUR_USERNAME/gridguard.git
cd gridguard
python -m venv .venv && source .venv/bin/activate
make install

# 2. Copy env file
cp .env.example .env

# 3. Generate synthetic data and train
make data-synthetic
make train

# 4. Start the API
make api
# → http://localhost:8000/docs

# 5. Start the dashboard (new terminal)
make dashboard
# → http://localhost:8501
```

---

## GMU Fairfax demo workflow

```bash
# Generate site-specific synthetic data for GMU Fairfax (250 kW system, 38.83°N)
make data-gmu

# Train models on GMU data
make train-gmu

# Start API and dashboard
make api
make dashboard
```

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

## Roadmap

- [ ] NSRDB integration — real irradiance data from NREL's National Solar Radiation Database
- [ ] OpenEI building load — add campus load profiles to model net metering impact
- [ ] PJM grid context — regional dispatch signals and day-ahead LMP as features
- [ ] Conformal prediction intervals — replace sigma thresholding with coverage-guaranteed intervals
- [ ] LightGBM comparison — add to model zoo
- [ ] PyTorch LSTM baseline — sequence model on sliding windows
- [ ] Multi-site fleet view — parameterise by site_id, aggregate fleet-level metrics
- [ ] Live data mode — poll a real inverter API, run forecast + anomaly in real time
- [ ] Email/Slack alerts — notify when high-severity event is detected
- [ ] pvlib clear-sky model — physics-based irradiance as feature
- [ ] Degradation trend — detect slow long-term output decline via residual trend

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
