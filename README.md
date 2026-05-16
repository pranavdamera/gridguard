# ⚡ GridGuard — Solar Asset Fault Detection & Energy Forecasting

> A production-leaning ML system that predicts expected solar generation,
> detects underperformance anomalies, explains alerts with SHAP, and ships
> a live dashboard. Built for portfolio demonstration and MLE skill development.

---

## Why this matters

Solar is the fastest-growing energy source globally — but a significant portion
of potential output is lost every year to undetected faults: soiling, partial
shading, inverter failures, and wiring degradation. Traditional monitoring waits
for alarms or manual inspection. ML-powered anomaly detection can flag
underperformance within minutes of onset, enabling proactive O&M and recovering
lost revenue.

This project models the core technical loop used by companies like Raptor Maps,
AlsoEnergy, Enercast, and utility-scale asset managers:

```
Weather + Time → Forecast expected output → Compare vs actual → Alert if delta is significant
```

---

## What ML skills this demonstrates

| Skill | Where |
|---|---|
| Time-series feature engineering | `features/engineer.py` — cyclic encodings, lag features, rolling stats |
| Temporal train/test splitting | Prevents future leakage; standard for TS problems |
| Model benchmarking pipeline | Persistence → Linear → RF → XGBoost, same eval harness |
| Residual-based anomaly detection | Threshold on normalised residuals, per-hour sigma |
| SHAP explainability | `explainability/shap_explain.py` — global + per-alert breakdown |
| ML system design | FastAPI serving layer, Pydantic schemas, artifact management |
| Evaluation discipline | MAE, RMSE, MAPE, R², skill score vs persistence |
| Reproducible pipelines | `make train` runs end-to-end from data to artifacts |
| Containerisation | Dockerfile + docker-compose for API + dashboard |
| Testing ML code | pytest for feature transforms, anomaly logic, API contracts |

---

## How this maps to energy/infrastructure AI roles

**ML Engineer at a solar O&M company** — you would own exactly this stack:
data ingestion from SCADA/inverters, forecasting models, alert logic, and the
internal dashboard that field engineers use.

**Applied Scientist at a grid analytics company** — the residual analysis,
conformal prediction intervals (next step in roadmap), and SHAP waterfall
explanations are the kinds of artefacts you would present to operators.

**Data Scientist at a utility** — energy loss estimation and daily fault
reports map directly to PPA contract monitoring and performance guarantees.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                 Data Layer                  │
│  NREL PVDAQ API  ──or──  Synthetic Generator│
│                 (15-min intervals)          │
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
│  /health  /forecast  /anomalies  /metrics  │
└───────┬───────────────────────────────────┘
        │
┌───────▼───────────────────────────────────┐
│         Streamlit Dashboard                │
│  Actual vs Predicted · Anomaly timeline   │
│  Daily loss · Model comparison · SHAP     │
└───────────────────────────────────────────┘
```

---

## DMV Region use case

GridGuard ships with a site registry for seven Northern Virginia / DC institutions:

```
GMU Fairfax · NOVA Annandale · NOVA Alexandria · NOVA Loudoun
NOVA Manassas · NOVA Woodbridge · DC Community Solar
```

Sites are defined in [config/sites.csv](config/sites.csv). Each has a latitude
that the synthetic generator uses to compute the correct sun elevation angle and
seasonal irradiance profile — so GMU Fairfax gets ~38.8°N solar geometry, not
the generic 37°N default.

```bash
# List all sites
python scripts/download_data.py --list-sites

# Generate site-specific data (uses site latitude and capacity)
python scripts/download_data.py --source synthetic --site-id gmu_fairfax
python scripts/download_data.py --source synthetic --site-id nova_loudoun
```

**Limitations:** Site capacities and coordinates are approximate values based on
public records — they are illustrative, not official specifications. The synthetic
data is a physical simulation, not real measured output from these campuses.

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
│   │   └── events.py          # Group consecutive intervals into events
│   ├── explainability/
│   │   └── shap_explain.py    # SHAP global + per-alert explanation
│   └── api/
│       ├── main.py            # FastAPI app (/health /forecast /anomalies /events /explain /metrics)
│       └── schemas.py         # Pydantic request/response schemas
├── dashboard/
│   └── app.py                 # Streamlit dashboard (events, explain, stratified metrics)
├── scripts/
│   ├── download_data.py       # CLI: --source, --site-id, --list-sites
│   └── run_pipeline.py        # CLI: end-to-end training pipeline
├── tests/
│   ├── test_features.py       # Feature engineering unit tests
│   ├── test_anomaly.py        # Anomaly detection tests
│   ├── test_events.py         # Event grouping tests
│   ├── test_sites.py          # Site registry + site-aware generation tests
│   ├── test_calibration.py    # Train/test calibration separation tests
│   ├── test_evaluate.py       # Stratified metrics tests
│   └── test_api.py            # FastAPI integration tests
├── docs/
│   ├── architecture.md        # System design and data flow
│   ├── data_sources.md        # Dataset options and setup
│   ├── model_card.md          # Model details, performance, limitations
│   └── demo_script.md         # 8-minute recruiter walkthrough
├── data/                      # gitignored; created at runtime
│   └── processed/
├── artifacts/                 # gitignored; model pkl + residual_stats.json
│   ├── models/
│   └── reports/               # Stratified metric CSVs
├── .github/workflows/ci.yml   # GitHub Actions: lint + test on push/PR
├── Makefile
├── pyproject.toml
├── LICENSE
├── CONTRIBUTING.md
├── Dockerfile
└── docker-compose.yml
```

---

## Datasets

### Option A — Synthetic (recommended to start)

No setup required. A physically-motivated generator creates 2 years of 15-minute data:
- Clear-sky irradiance from sun elevation geometry
- Log-normal cloud attenuation
- Temperature with diurnal + seasonal cycle
- Panel efficiency as a function of temperature
- 5% of days have injected fault events (ground-truth labels for evaluation)

```bash
make data-synthetic
```

### Option B — NREL PVDAQ (real data)

Real residential and commercial PV system data from the National Renewable
Energy Laboratory. Free API key required.

1. Sign up at https://developer.nrel.gov/signup/
2. Add key to `.env`: `NREL_API_KEY=your_key_here`
3. Run: `make data-nrel`

Browse available systems: https://developer.nrel.gov/docs/solar/pvdaq-v3/

### Option C — Open Power System Data

Hourly solar generation for European countries:
https://open-power-system-data.org/data-packages/time_series

Download the CSV and adapt `ingestion/download.py` to parse it.

### Option D — Ausgrid Solar Home Dataset

Australian residential solar + load data, 30-minute intervals:
https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data

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

# 3. Generate synthetic data and train all models
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

## Commands

| Command | Description |
|---|---|
| `make install` | Install all dependencies |
| `make data-synthetic` | Generate 2 years of synthetic solar data |
| `make data-nrel` | Download real NREL PVDAQ data |
| `make train` | Run full training pipeline |
| `make api` | Start FastAPI server |
| `make dashboard` | Start Streamlit dashboard |
| `make test` | Run pytest suite with coverage |
| `make lint` | Run ruff + black check |
| `make format` | Auto-fix lint issues |
| `make docker-up` | Start API + dashboard via Docker |
| `python scripts/download_data.py --list-sites` | Print the DMV site registry |
| `python scripts/download_data.py --site-id gmu_fairfax` | Generate site-specific data |

---

## API reference

After `make api`, docs are at http://localhost:8000/docs

**GET /health**
```json
{"status": "ok", "model_loaded": true, "data_rows": 17520, "version": "0.1.0"}
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
{"timestamp": "...", "predicted_kw": 7.84, "model_name": "xgboost"}
```

**GET /anomalies?limit=50**
```json
{
  "total_anomalies": 312,
  "total_lost_kwh": 48.7,
  "records": [...]
}
```

**GET /metrics**
```json
{
  "models": [
    {"model": "xgboost", "mae_kw": 0.21, "rmse_kw": 0.38, "mape_pct": 4.2, "r2": 0.97}
  ],
  "best_model": "xgboost"
}
```

**GET /events?limit=20**
```json
{
  "total_events": 18,
  "total_lost_kwh": 48.7,
  "events": [
    {
      "event_id": 1,
      "start_time": "2023-07-04T10:00:00",
      "end_time": "2023-07-04T13:45:00",
      "duration_minutes": 225,
      "interval_count": 15,
      "total_lost_kwh": 12.3,
      "max_residual_sigma": -4.1,
      "severity": "high"
    }
  ]
}
```

**GET /explain?timestamp=2023-07-04T11:00:00**
```json
{
  "timestamp": "2023-07-04T11:00:00",
  "predicted_kw": 7.84,
  "contributors": [
    {"feature": "irradiance_wm2", "feature_value": 420.0, "shap_value": -2.1, "direction": "negative"},
    {"feature": "ac_power_lag1",  "feature_value": 3.2,   "shap_value": -1.4, "direction": "negative"}
  ],
  "model_name": "xgboost",
  "shap_available": true
}
```

---

## Limitations

This is a portfolio and learning project. Be explicit about what it is not:

- **Trained on synthetic data** — Real sensor noise, clipping, soiling ramp, and
  row-to-row shading are not modelled. Retrain on real data before drawing conclusions.
- **No streaming** — Batch pipeline only. Live fault detection needs a streaming layer.
- **Residual thresholding** — No coverage guarantee. Conformal prediction intervals
  would be the right next step for calibrated alert rates.
- **Single-site model** — No cross-site transfer or fleet-normalised features.
- **Site registry is illustrative** — DMV site capacities and coordinates are
  approximate estimates from public records, not official specifications.
- **Not safety-critical** — Do not use as the sole basis for dispatch, maintenance,
  or financial decisions without independent validation.

See [docs/model_card.md](docs/model_card.md) for the full model card.

---

## Learning plan

Work through these in order. Each builds directly on the previous.

### Week 1 — Data & features
1. **Pandas time-series basics** — `resample`, `rolling`, `shift`, `dt` accessors
2. **Feature engineering for time series** — why lag features, why cyclic encodings
3. **Temporal cross-validation** — why random splits break TS models (data leakage)
4. **Practical task**: Add a `clearsky_ratio` feature using the `pvlib` library

### Week 2 — Models
5. **sklearn Pipeline** — why to use it, how StandardScaler fits inside
6. **RandomForest internals** — bagging, feature importance, overfitting risk
7. **XGBoost / gradient boosting** — boosting vs bagging, learning rate, early stopping
8. **Practical task**: Grid search hyperparameters for XGBoost using `TimeSeriesSplit`

### Week 3 — Evaluation & anomaly detection
9. **Regression metrics** — MAE vs RMSE vs MAPE: when each matters
10. **Skill score** — how to contextualise model improvement over a naive baseline
11. **Residual analysis** — check for heteroskedasticity, autocorrelation in errors
12. **Practical task**: Plot prediction intervals using quantile regression

### Week 4 — Explainability & MLOps
13. **SHAP values** — why TreeExplainer is faster than KernelExplainer for tree models
14. **FastAPI lifespan** — how startup/shutdown events work for loading ML models
15. **Model artifact management** — pickle vs joblib vs ONNX trade-offs
16. **Practical task**: Add an `/explain` endpoint that returns SHAP breakdown per alert

---

## 7-day sprint plan

| Day | Goal | Deliverable |
|---|---|---|
| **1** | Environment + data | `make data-synthetic` works; inspect DataFrame in notebook |
| **2** | Feature engineering | All FEATURE_COLS generated; `test_features.py` passes |
| **3** | Train baseline models | `make train` runs; metrics table printed to console |
| **4** | Anomaly detection | `detect_anomalies()` running; injected faults detected at >70% rate |
| **5** | FastAPI backend | `/health`, `/forecast`, `/anomalies`, `/metrics` all returning 200 |
| **6** | Streamlit dashboard | Dashboard loads, shows actual vs predicted, anomaly table |
| **7** | Polish + GitHub | Clean README, tests passing, `.env.example` complete, push to GitHub |

---

## Roadmap

- [ ] Conformal prediction intervals — replace sigma thresholding with coverage-guaranteed intervals (`mapie`)
- [ ] LightGBM comparison — add to model zoo, compare to XGBoost
- [ ] PyTorch LSTM baseline — sequence model on sliding windows
- [ ] Multi-site support — parameterise by `system_id`, aggregate fleet view
- [ ] DuckDB caching — replace parquet cache with queryable local DB
- [ ] MLflow tracking — log runs, compare experiments in UI
- [ ] Live data mode — poll a real inverter API, run forecast + anomaly in real time
- [ ] Email/Slack alerts — notify when fault day detected
- [ ] pvlib clear-sky model — physics-based irradiance as feature
- [ ] Degradation trend — detect slow long-term output decline via residual trend

---

## Stack

| Layer | Library |
|---|---|
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, xgboost |
| Explainability | shap |
| API | fastapi, uvicorn, pydantic |
| Dashboard | streamlit, plotly |
| Config | pydantic-settings, python-dotenv |
| Infra | Docker, docker-compose |
| Dev | pytest, ruff, black, make |

---

## License

MIT
