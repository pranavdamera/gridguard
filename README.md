# GridGuard

[![CI](https://github.com/pranavdamera/gridguard/actions/workflows/ci.yml/badge.svg)](https://github.com/pranavdamera/gridguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Open-source spatial intelligence and anomaly detection for distributed solar.**

GridGuard forecasts what a healthy photovoltaic array *should* produce, puts a
calibrated uncertainty band around that expectation, flags generation that falls
outside it, and uses nearby sites to separate an equipment fault from a cloud
that fooled the weather model.

It runs on **real measured telemetry** from three NIST arrays in Gaithersburg,
Maryland, alongside a **simulated** DMV campus fleet used for controlled fault
injection. The two are never blurred: every API response, chart, and site card
states which it is showing.

---

## Live demo

| | |
|---|---|
| **Frontend** | `<set after deploying — Vercel>` |
| **API docs** | `<set after deploying — Render>/docs` |
| **Health** | `<set after deploying — Render>/health` |

Deployment is documented in [docs/deployment.md](docs/deployment.md). To run it
locally, see [Quick start](#quick-start) — it takes one command and no
credentials.

> **Screenshots / GIF.** Not committed to keep the repository small. To capture
> them: run the [90-second demo script](docs/demo_script.md), which is
> deterministic — the same commit reproduces the same events and the same
> numbers. Capture `/demo` (fleet map), `/events/nist_ground:53` (the real
> outage, with spatial attribution), and `/data` (provenance). Place them in
> `docs/images/` and link them here.

---

## What problem it solves

An underperforming solar array is invisible. It still produces a plausible
bell curve; it is just a smaller one than it should be. Three things have to be
true before that becomes actionable:

1. **You need a model of expected output.** Otherwise there is nothing to
   compare against.
2. **The alert threshold needs a known false-alarm rate.** A threshold picked by
   eye produces alerts nobody trusts. GridGuard uses conformal prediction: at
   α = 0.05, at most ~5% of healthy intervals should breach the bound — with no
   distributional assumption.
3. **A drop is not always a fault.** If neighbouring arrays under the same sky
   dropped too, it was the weather. GridGuard checks the neighbourhood before
   pointing at equipment.

---

## Architecture

```
                       ┌──────────────────────────────────────┐
   OEDI data lake ────►│  data/  — one canonical schema        │
   (NREL PVDAQ)        │    RealPVDataSource   (measured)      │
                       │    SyntheticDMVSource (simulated)     │
   simulator ─────────►│  + DatasetProvenance on every frame   │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  features/  weather_only | lag_aware  │
                       └──────────────────┬───────────────────┘
                                          │
              ┌───────────────────────────┼──────────────────────────┐
              ▼                           ▼                          ▼
     ┌────────────────┐        ┌────────────────────┐      ┌──────────────────┐
     │ forecasting    │        │ expected generation│      │ physics (pvlib)  │
     │ RF/XGB/Ridge/  │        │ weather-only XGB   │      │ + hybrid residual│
     │ persistence    │        │ healthy data only  │      │                  │
     └────────────────┘        └─────────┬──────────┘      └──────────────────┘
                                         │
                       ┌─────────────────▼────────────────────┐
                       │ anomaly/  conformal bounds (default)  │
                       │           sigma detector (benchmark)  │
                       │           event grouping + severity   │
                       └─────────────────┬────────────────────┘
                                         │
                       ┌─────────────────▼────────────────────┐
                       │ spatial/  haversine neighbourhoods    │
                       │           site-specific vs regional   │
                       │ fleet/    health rollup + KPIs        │
                       └─────────────────┬────────────────────┘
                                         │
        scripts/build_artifacts.py ──────┴──────► artifacts/ + manifest.json
                                                          │
                       ┌──────────────────────────────────▼──┐
                       │ FastAPI — loads artifacts, never     │
                       │ trains                               │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │ Next.js — fleet map, site analytics,  │
                       │ event investigation, provenance       │
                       └──────────────────────────────────────┘
```

---

## Real vs simulated data

This distinction is the organising principle of the project, because the two
answer questions the other cannot.

| | Measured | Simulated |
|---|---|---|
| **Sites** | 3 NIST arrays, Gaithersburg MD | 7 DMV campus sites |
| **Source** | NREL PVDAQ via the OEDI data lake | GridGuard simulator (pvlib clear-sky + AR(1) clouds) |
| **Weather** | Instruments at the array | Modelled |
| **Fault labels** | None — real telemetry has none | Known by construction |
| **Used for** | **Forecast accuracy** | **Detection accuracy** |

**Forecast accuracy is measured on real data.** The target is genuine
generation, so error metrics describe real predictive skill.

**Detection accuracy needs ground truth**, which measured data lacks. So faults
are injected with known labels — *including into the measured telemetry*, where
the generation and weather stay real and only the failures are simulated. Every
detection number in this repository describes performance against injected
faults, and is labelled as such. None of it is field-validated.

The DMV site names (GMU, NOVA, DC) are **illustrative**. No named institution
supplied telemetry to this project. Every simulated surface says so.

### Real-data provenance

| | |
|---|---|
| **Dataset** | NREL PVDAQ, distributed via the [Open Energy Data Initiative](https://openei.org/wiki/PVDAQ) data lake |
| **Systems** | 4901 (canopy, 242.5 kW), 4902 (ground, 270.7 kW), 4903 (roof, 73.7 kW) |
| **Location** | NIST Gaithersburg, MD (39.13°N, 77.21°W) |
| **Period** | 2016-01-01 → 2016-12-31, resampled 1-min → 15-min |
| **Weather** | Co-located pyranometer/reference cell, ambient temperature, wind |
| **Credentials** | **None.** Anonymous HTTPS. |

The legacy `developer.nrel.gov/api/pvdaq/v3` REST API has been **decommissioned**.
The data remains public through the OEDI data lake, which is what GridGuard
reads. NSRDB is implemented ([`data/nsrdb.py`](src/gridguard/data/nsrdb.py)) for
sites without on-site instruments, but is **not** on the shipped path and needs
no key for anything GridGuard ships.

Three details that matter, because getting them wrong produces plausible but
wrong numbers:

- **Channels are selected by declared units, not by name.** On these systems two
  of four irradiance channels are raw pyranometer millivolts, the column named
  `ac_power_meter_1864` is reactive energy in kVARh, and wind speed is filed
  under "AC other". Millivolt channels are rejected rather than converted — the
  calibration constant is not published, and guessing it would corrupt
  everything downstream.
- **The timezone is derived, not assumed.** PVDAQ publishes local and UTC
  columns; the offset is read from them and confirmed constant across the year
  (local standard time, no daylight-saving shift).
- **Every channel is validated against a clear-sky model** transposed into the
  sensor's own plane. A tilted sensor compared against a horizontal reference
  fails every winter — that check caught exactly that bug during development.

---

## ML methodology

### Two feature sets

| Mode | Purpose | Includes lagged power? |
|---|---|---|
| `weather_only` | Expected generation, for anomaly detection | **No** |
| `lag_aware` | Short-horizon forecasting | Yes |

The exclusion is the constraint the detector rests on. A system degraded for
days produces low output → its lagged power is low → a lag-aware model predicts
low output → the degradation is declared normal. The failure is silent and
total, so it is enforced in code and covered by tests, not left to convention.

### Models

| Model | Notes |
|---|---|
| Persistence | Naive lag-1 baseline |
| Ridge | Regularised linear |
| RandomForest / XGBoost | Non-linear irradiance × temperature interactions |
| **pvlib physics** | PVWatts chain from published tilt/azimuth/nameplate. **One** fitted parameter (an overall derate), so the comparison is fair |
| **Physics + ML hybrid** | Gradient boosting on the physics *residual*, not on power — limits how much of a real fault it can absorb |

All evaluation uses **temporal** splits. Random splits on time series leak the
future into training and inflate every score.

### Conformal prediction (production default)

Split conformal with Mondrian calibration per irradiance bucket. Given
exchangeable calibration residuals:

```
P(actual − predicted ≥ lower_bound) ≥ 1 − α
```

with **no distributional assumption**. PV residuals are heavy-tailed, where a
"2σ" rule is badly miscalibrated — the outliers inflate the σ that defines it,
so it covers ~99% instead of the ~95% the choice of *k* implies, silently
missing faults. The sigma detector is retained as a benchmark so that comparison
stays visible rather than asserted.

**What the guarantee does not say.** Coverage is marginal within each bucket,
not conditional on every interval. Exchangeability is an assumption the world
breaks — degradation, sensor drift and seasonal shift all violate it, which is
why calibration is recomputed per build from the *most recent* held-out healthy
data. And a breach means "outside the calibrated range of healthy behaviour",
not "this component failed".

### Geospatial methodology

Haversine distances (spherical Earth — under 0.5% from WGS-84 at fleet scale,
far below the uncertainty in whether two arrays share weather at all). Residuals
are normalised by nameplate so a 70 kW roof and a 270 kW field are comparable,
then compared against neighbours within 60 km using both an unweighted median
and an inverse-distance weighting.

Neighbours are restricted to the **same data mode**: the simulator draws each
site's weather independently, so a simulated neighbour carries no information
about a measured array's cloud field, and mixing them would manufacture
agreement out of noise.

This is **not a fault classifier**. It answers one question — is this deviation
shared with nearby sites — and always reports the evidence behind its answer.
With fewer than two neighbours it returns `indeterminate` rather than guessing.

---

## Results

All figures come from the artifact manifest of the build that produced the
served models. Nothing here is hand-entered.

### Forecast accuracy — measured NIST arrays, held out from 2016-10-01

| Site | Best model | MAE (kW) | RMSE (kW) | R² | Capacity |
|---|---|---|---|---|---|
| NIST Roof | XGBoost | 0.54 | 1.11 | 0.992 | 73.7 kW |
| NIST Canopy | XGBoost | 1.22 | 3.20 | 0.993 | 242.5 kW |
| NIST Ground | RandomForest | 1.57 | 5.26 | 0.992 | 270.7 kW |

### Weather-only models (no lagged power — not comparable with the above)

| Site | pvlib physics | Physics + ML hybrid | Weather-only XGBoost |
|---|---|---|---|
| NIST Roof | 0.56 | **0.56** | 0.63 |
| NIST Canopy | 2.27 | **1.48** | 1.70 |
| NIST Ground | 5.48 | **4.48** | 4.71 |

MAE in kW. The hybrid beats pure ML on two of three arrays, which is the
interesting result: anchoring a learned model to a physical one improves it
*and* constrains how much fault it can absorb. Physics numbers are reported only
on measured sites — on simulated data the baseline inverts the same clear-sky
model that generated the series, so its accuracy there is tautological.

### Conformal coverage (target ≥ 0.95, held-out healthy intervals)

| NIST Ground | NIST Canopy | NIST Roof |
|---|---|---|
| 0.962 | 0.989 | 0.991 |

All three measured sites meet the guarantee. Two *simulated* sites sit below it
(0.86, 0.90) — that is the exchangeability limitation manifesting on data with
deliberately-modelled performance drift. It is reported per-site in the UI
rather than tuned away.

### Detection — injected faults on measured telemetry

| Site | Precision | Recall | F1 | Events found | False alarms/day |
|---|---|---|---|---|---|
| NIST Ground | 0.58 | 0.63 | 0.60 | 5/5 | 0.96 |
| NIST Roof | 0.68 | 0.51 | 0.58 | 5/5 | 0.80 |
| NIST Canopy | 0.57 | 0.34 | 0.43 | 5/5 | 0.52 |

**Every event was detected**, but only ~half of the individual intervals within
them. That gap is real and worth stating plainly: GridGuard reliably notices
that something is wrong, and is much less reliable about precisely which
intervals were affected. Per-class breakdowns on `/methodology` show why — total
outages are caught at 100% recall while gradual degradation is caught at ~35%,
which is the honest ranking of difficulty. Detection is consistently better on
simulated data (F1 ≈ 0.87) than on real data, which is exactly what you should
expect and why the real numbers are the ones quoted here.

---

## Quick start

```bash
git clone https://github.com/pranavdamera/gridguard
cd gridguard
python -m venv .venv && source .venv/bin/activate

make install            # backend + frontend dependencies
make build-artifacts    # ~4 min. No network, no credentials.

make api                # terminal 1 → http://localhost:8000/docs
make web                # terminal 2 → http://localhost:3000
```

`make build-artifacts` needs no download because a year of curated measured
telemetry (~3 MB) is committed to `data/curated/`, and the simulated fleet is
regenerated deterministically from a fixed seed.

Optional research dashboard (`pip install -e ".[dashboard]"` first):

```bash
make dashboard          # → http://localhost:8501
```

Streamlit is an internal model-diagnostics surface. **The Next.js app is the
product.**

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, artifact commit, what loaded |
| `GET /sites`, `/sites/{id}` | Registry with provenance disclaimers |
| `GET /sites/{id}/timeseries` | Actual, expected, calibrated band |
| `GET /fleet/summary` | Fleet KPIs, per-site health, spatial scope, map bounds |
| `GET /events`, `/events/{id}` | Events and full investigation |
| `GET /anomalies` | Interval-level detector output |
| `POST /forecast` | Expected generation for supplied weather |
| `GET /metrics` | Model evaluation, read from the manifest |
| `GET /methodology` | How detection works, and its limits |
| `GET /data` | Dataset provenance, real vs simulated |
| `GET /demo/scenario` | Guided walkthrough entry point |

Interactive docs at `/docs`. Every response carrying generation numbers also
carries the `data_mode` that produced them.

---

## Reproducible artifacts

One command does everything:

```bash
make build-artifacts
```

→ verify/load data → preprocess → train → calibrate → evaluate → write
`artifacts/models/manifest.json` recording the git commit, dataset window,
library versions, hyperparameters, calibration configuration, and every
evaluation score.

**The backend never trains.** It loads artifacts and reports the manifest
alongside every metric, so any number in the UI traces to the build that
produced it. Artifacts (~150 MB) are built during deploy rather than committed;
the curated measured data that makes that possible offline *is* committed.

---

## Tests

```bash
make test          # 224 tests
make lint          # ruff, black, eslint, tsc
```

Covers real-data channel resolution and unit handling, timezone derivation,
missing-value policy, temporal splits, the no-leakage constraint, conformal
calibration and coverage, spatial logic, fault injection, provenance, artifact
loading, and API contracts. CI runs the Python suite on 3.11 and 3.12 plus the
frontend lint/typecheck/build.

---

## Limitations

Stated plainly, because a demo that oversells is worse than one that doesn't.

- **Detection metrics are not field-validated.** They come from injected faults.
  No one has confirmed a GridGuard alert against a real maintenance record.
- **Interval-level recall is mediocre on real data** (~0.5). Event-level recall
  is 5/5, so it notices problems; it is imprecise about their exact extent.
- **Gradual degradation is caught poorly** (~35% of intervals) — the hardest and
  most economically important case.
- **It does not diagnose.** It detects and localises deviation. It cannot tell
  you which component failed, and does not claim to.
- **Spatial attribution needs fleet density.** Three arrays 400 m apart give a
  strong signal; sites 50 km apart under different cloud fields do not.
- **One year of measured data, one site cluster.** No seasonal generalisation
  across years, and no diversity of climate, mounting, or equipment.
- **Simulated data is a simulation.** Row-to-row shading, spectral effects,
  snow, and real inverter dynamics are not modelled.
- **Conformal coverage degrades under drift** — demonstrated, not hypothetical:
  two simulated sites currently sit below target.
- **Three npm advisories remain**, all inside Next.js itself. Fixing them
  requires moving past the pinned major version; deferred deliberately.
- **Not safety-critical.** Not a basis for dispatch, maintenance, or financial
  decisions without independent validation.

---

## Roadmap

- Multi-year, multi-climate measured data — the single biggest credibility gain
- Event-level precision/recall against a real maintenance log
- Conformal under distribution shift (weighted / adaptive conformal) to address
  the drift limitation directly
- Degradation-trend detection: slow year-over-year decline, distinct from faults
- PJM market context for lost-revenue estimation (architecturally ready; needs
  real price data, and GridGuard will not fabricate prices)
- Streaming ingestion for live detection

---

## Documentation

| Document | Contents |
|---|---|
| [docs/methodology.md](docs/methodology.md) | Feature modes, leakage, detection design |
| [docs/modeling.md](docs/modeling.md) | Model design rationale |
| [docs/data_sources.md](docs/data_sources.md) | Dataset provenance in detail |
| [docs/model_card.md](docs/model_card.md) | Model card |
| [docs/deployment.md](docs/deployment.md) | Vercel + Render deployment |
| [docs/demo_script.md](docs/demo_script.md) | 90-second walkthrough |
| [docs/research_angle.md](docs/research_angle.md) | Research collaboration context |

---

## Stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, xgboost, **pvlib** |
| Uncertainty | split conformal prediction (implemented here) |
| Explainability | shap |
| API | fastapi, uvicorn, pydantic |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind, **MapLibre GL**, Recharts |
| Research UI | streamlit, plotly (optional extra) |
| Dev | pytest, ruff, black |

---

## Attribution

Measured data: **NREL PVDAQ**, via the Open Energy Data Initiative. Basemap:
**OpenStreetMap** contributors, tiles by **CARTO**. GridGuard is not affiliated
with NREL, NIST, CARTO, or any institution named in the simulated fleet.

## License

MIT — see [LICENSE](LICENSE).
