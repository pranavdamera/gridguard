# Architecture

GridGuard is a batch pipeline that produces versioned artifacts, a stateless API
that serves them, and a web application that renders them. It runs end to end on
a laptop with no cloud dependencies and no credentials.

---

## The central constraint

**The backend never trains.** Training on request would be slow,
non-deterministic across library versions, and would make every restart a silent
experiment. Instead a single build command produces every artifact, records what
it did in a manifest, and the API loads that.

```
  scripts/build_artifacts.py                    gridguard.api
  ─────────────────────────                     ─────────────
  data → features → models → calibration   ──►  load once at startup
       → evaluation → manifest.json             serve, never fit
```

Everything else follows from this: metrics come from the manifest rather than
being recomputed, `/health` reports the artifact commit, and a deployment can be
traced to the build that produced it.

---

## Package layout

```
src/gridguard/
├── config.py              Pydantic settings (env / .env)
├── sites/registry.py      Site catalogue — real and simulated in one registry
│
├── data/                  ── Data layer ──────────────────────────────
│   ├── schema.py          Canonical schema, resampling, missing values
│   ├── provenance.py      DatasetProvenance — attached to every dataset
│   ├── oedi.py            NREL PVDAQ via the OEDI data lake
│   ├── nsrdb.py           Optional weather API (unused by the shipped path)
│   ├── synthetic.py       DMV fleet simulator
│   ├── faults.py          Fault taxonomy and injection
│   └── sources.py         DataSource ABC + dispatch by data_mode
│
├── features/engineer.py   weather_only | lag_aware feature sets
│
├── models/                ── Modelling ────────────────────────────────
│   ├── baseline.py        Persistence, Ridge, RandomForest, XGBoost
│   ├── physics.py         pvlib PVWatts chain + hybrid residual model
│   ├── train.py           Legacy training entry point
│   └── evaluate.py        Metrics, stratified reporting
│
├── anomaly/               ── Detection ────────────────────────────────
│   ├── conformal.py       Split conformal, Mondrian calibration
│   ├── detect.py          Detection (conformal default, sigma benchmark)
│   ├── events.py          Grouping, severity
│   └── evaluate.py        Detection metrics against injected faults
│
├── spatial/               ── Geospatial ───────────────────────────────
│   ├── geo.py             Haversine, neighbour graphs, bounds
│   └── context.py         Site-specific vs regional attribution
│
├── fleet/aggregate.py     Per-site health rollup, fleet KPIs
├── explainability/        SHAP attribution
├── artifacts/manifest.py  Build manifest
├── pipeline.py            build_site() — the per-site build
└── api/                   FastAPI app, schemas, artifact store
```

---

## Data flow

```
┌─────────────────────────────────────────────────────────────────┐
│  SOURCES                                                        │
│                                                                 │
│   OEDI data lake ──► RealPVDataSource ──┐                       │
│   (NREL PVDAQ)                          │                       │
│                                         ├──► canonical frame    │
│   simulator ──────► SyntheticDMVSource ─┘    + DatasetProvenance│
│                                                                 │
│   Canonical schema: timestamp (local standard, naive), site_id, │
│   ac_power_kw, irradiance_wm2, temperature_c, wind_speed_ms,    │
│   data_mode.  Optional: injection labels, module temperature.   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │  Everything downstream is source-agnostic;
                             │  data_mode keeps provenance explicit.
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  TEMPORAL SPLIT   train | val | test    (never random)          │
└────────────┬──────────────────────┬─────────────────────────────┘
             │                      │
             ▼                      ▼
   ┌──────────────────┐   ┌──────────────────────────────────┐
   │ lag_aware        │   │ weather_only                     │
   │ forecasting      │   │ expected generation              │
   │                  │   │ (healthy intervals only)         │
   │ persistence      │   │                                  │
   │ ridge            │   │ XGBoost  ← detection runs on this│
   │ random forest    │   │ pvlib physics                    │
   │ xgboost          │   │ physics + ML hybrid              │
   └──────────────────┘   └───────────────┬──────────────────┘
                                          │
                          ┌───────────────▼──────────────────┐
                          │ CALIBRATION (held-out val tail)  │
                          │  conformal bounds per irradiance │
                          │  bucket + legacy per-hour sigma  │
                          └───────────────┬──────────────────┘
                                          │
                          ┌───────────────▼──────────────────┐
                          │ DETECTION over the test split    │
                          │  actual < calibrated lower bound │
                          │  → group into events + severity  │
                          └───────────────┬──────────────────┘
                                          │
             ┌────────────────────────────┼────────────────────────┐
             ▼                            ▼                        ▼
   ┌──────────────────┐        ┌────────────────────┐   ┌──────────────────┐
   │ spatial context  │        │ fleet aggregation  │   │ detection eval   │
   │ site vs regional │        │ health + KPIs      │   │ (injected faults)│
   └──────────────────┘        └────────────────────┘   └──────────────────┘
                                          │
                                          ▼
                            artifacts/models/ + manifest.json
```

---

## Serving

```
┌────────────────────────────────────────────────────────────┐
│  ArtifactStore — loaded once at startup                    │
│                                                            │
│  per site:  detected frame · events · provenance           │
│             anomaly model · conformal calibration          │
│  global:    manifest                                       │
│                                                            │
│  Forecast-model pickles are deliberately NOT loaded:       │
│  /metrics reads the manifest and /forecast uses the        │
│  weather-only model, so holding tree ensembles resident    │
│  would cost hundreds of MB for no served response.         │
│                                                            │
│  Loading is tolerant — a site with missing artifacts is    │
│  skipped with a warning and /health reports it, rather     │
│  than taking the service down.                             │
└────────────────────────────┬───────────────────────────────┘
                             │
                    FastAPI (stateless)
                             │
                             ▼
              Next.js (server components, fetch + revalidate)
```

---

## Storage strategy

| What | Location | Size | Committed |
|---|---|---|---|
| Curated measured telemetry | `data/curated/` | ~3 MB | **Yes** |
| Working cache | `data/processed/` | ~25 MB | No |
| Models, calibration, manifest | `artifacts/models/` | ~150 MB | No |

Committing the curated measured data is what makes the whole deployment story
simple: the build step needs no network access and no credentials, because the
hard-to-obtain input is already in the repository and the simulated fleet is
regenerated deterministically. Artifacts are rebuilt at deploy time.

---

## Design decisions worth knowing

**One registry, two data modes.** Real and simulated sites live in the same
registry with a `data_mode` column, so fleet views, the map, and spatial logic
operate on one uniform collection while provenance stays explicit everywhere it
surfaces.

**Provenance is a first-class object, not a comment.** Every dataset carries a
`DatasetProvenance` record — instrument, units basis, timezone derivation,
processing steps, known limitations — and the `/data` page renders it directly.

**Channels resolved by declared units.** PVDAQ channel *names* are unreliable;
its metrics table declares units, and that is what selection uses. See
[data_sources.md](data_sources.md).

**Detection is structurally prevented from seeing lagged power.** Not a
convention — the columns are absent from the feature set, and tests enforce it.

**Neighbours never cross data modes.** A simulated site's residuals carry no
information about a measured site's weather.

**The API is stateless.** No database, no session state, no background jobs.
Everything is derived from immutable artifacts, so horizontal scaling is trivial
and a restart is a no-op.

---

## What is deliberately absent

No message queue, no feature store, no orchestrator, no Kubernetes, no database.
GridGuard is a batch pipeline plus a read-only API. Adding infrastructure would
add operational surface without answering any question the project poses.

The one genuine architectural gap is **streaming**: detection currently runs over
a completed dataset, not a live feed. That is the honest next step, and it is a
real change rather than a configuration one.
