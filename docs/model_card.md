# Model card — GridGuard

Covers the models GridGuard ships and the detector built on them. Numbers come
from the artifact manifest of the build that produced the served models
(`artifacts/models/manifest.json`); none are hand-entered.

---

## Overview

GridGuard ships **three** kinds of model per site, plus a calibration artifact.

| Artifact | Type | Purpose |
|---|---|---|
| `{name}.pkl` | Persistence / Ridge / RandomForest / XGBoost | Short-horizon forecasting (**lag-aware** features) |
| `anomaly_detector.pkl` | XGBoost | **Expected generation** (weather-only features) — what detection runs against |
| `physics.pkl`, `physics_hybrid.pkl` | pvlib PVWatts chain, and that chain + a learned residual correction | Physically-grounded expected generation |
| `conformal_calibration.json` | Split-conformal bounds per irradiance bucket | The calibrated healthy range |
| `residual_stats.json` | Per-hour residual mean/σ | Legacy sigma detector (benchmark) |

Version 0.2.0 · MIT · Explainability via SHAP TreeExplainer.

---

## Intended use

**Intended.** Research and engineering demonstration of solar underperformance
detection. Estimating expected generation from weather. Flagging deviation with
a stated false-alarm rate. Distinguishing site-specific from regional deviation
across a fleet. Teaching and evaluating detection methodology.

**Not intended.** Operational dispatch. Maintenance scheduling without
independent confirmation. Financial settlement or lost-revenue claims. Warranty
or performance-guarantee adjudication. Safety-critical decisions of any kind.

**Out of scope by construction.** GridGuard does **not** diagnose which component
failed. It detects and localises deviation. Any statement stronger than that is
not something this system supports.

---

## Feature sets

| Mode | Features | Used by |
|---|---|---|
| `weather_only` | hour, day-of-year, month, cyclic encodings, irradiance, irradiance², temperature, wind, 1h rolling irradiance | Expected generation / detection |
| `lag_aware` | all of the above **+** `ac_power_lag1`, `ac_power_lag4` | Forecasting only |

**The exclusion of lagged power from the detection model is the single most
important design decision in this project.** A system degraded for days produces
low output, so its lagged power is low, so a lag-aware model predicts low output
and reports the degradation as normal. The failure is silent and complete. It is
enforced in code and covered by tests.

The expected-generation model is additionally trained on **healthy intervals
only**, so it cannot absorb known faults into its notion of normal.

---

## Training data

| | Measured | Simulated |
|---|---|---|
| Source | NREL PVDAQ (OEDI data lake) | GridGuard simulator |
| Sites | 3 (NIST Gaithersburg MD) | 7 (illustrative DMV) |
| Period | 2016-01-01 → 2016-12-31 | 2016-01-01 → 2016-12-31 |
| Interval | 15 min (from 1 min) | 15 min |
| Rows | 31,702 – 35,136 per site | 35,041 per site |
| Split | Temporal, held out from 2016-10-01 | Same |

Splits are always temporal. Random splits leak the future into training.
XGBoost early stopping validates on the chronological tail of the *training*
window, never on the test split.

See [data_sources.md](data_sources.md) for full provenance.

---

## Performance

### Forecast accuracy — measured arrays, held-out test period

| Site | Best model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|---|
| NIST Roof (73.7 kW) | XGBoost | 0.54 | 1.11 | 0.992 |
| NIST Canopy (242.5 kW) | XGBoost | 1.22 | 3.20 | 0.993 |
| NIST Ground (270.7 kW) | RandomForest | 1.57 | 5.26 | 0.992 |

Persistence, as the honest floor: MAE 1.14 / 3.60 / 5.52 kW respectively.

### Weather-only models (no lagged power — not comparable with the above)

MAE in kW:

| Site | pvlib physics | Physics + ML hybrid | Weather-only XGBoost |
|---|---|---|---|
| NIST Roof | 0.56 | **0.56** | 0.63 |
| NIST Canopy | 2.27 | **1.48** | 1.70 |
| NIST Ground | 5.48 | **4.48** | 4.71 |

The hybrid beats pure ML on two of three arrays. Physics results are reported
**only on measured data**: the simulator and the physics baseline share a pvlib
clear-sky model, so physics accuracy on simulated data is near-tautological and
is not evidence of anything.

The physics baseline fits exactly **one** free parameter (an overall derate).
Everything else is fixed physics or published metadata, which is what makes the
comparison against multi-hundred-parameter ensembles meaningful.

### Conformal coverage (α = 0.05, target ≥ 0.95)

Measured on held-out healthy intervals that neither the model nor the
calibration saw:

| NIST Ground | NIST Canopy | NIST Roof |
|---|---|---|
| 0.962 | 0.989 | 0.991 |

Two *simulated* sites currently sit below target (0.86, 0.90). That is the
exchangeability assumption breaking under modelled performance drift — reported
per-site in the UI rather than suppressed.

### Detection — injected faults on measured telemetry

| Site | Precision | Recall | F1 | Events | False alarms/day |
|---|---|---|---|---|---|
| NIST Ground | 0.58 | 0.63 | 0.60 | 5/5 | 0.96 |
| NIST Roof | 0.68 | 0.51 | 0.58 | 5/5 | 0.80 |
| NIST Canopy | 0.57 | 0.34 | 0.43 | 5/5 | 0.52 |

Per fault class (typical), which is the more informative view:

| Class | Detected? | Typical recall |
|---|---|---|
| Complete outage | ✅ | 1.00 |
| Partial outage | ✅ | 1.00 |
| Shading | ✅ | 0.83 |
| Persistent derate | ⚠️ | 0.51 |
| Gradual degradation | ⚠️ | 0.35 |
| Clipping *(not a fault)* | ✅ correctly ignored | 0.00 false-alarm rate |
| Comm dropout *(not a fault)* | ✅ correctly ignored | 0.00 |
| Sensor dropout *(not a fault)* | ❌ **often false-alarms** | up to 0.59 |

**How to read these.** Every injected event was detected, but interval-level
recall is around half — GridGuard reliably notices that something is wrong and is
much less precise about exactly which intervals were affected. The ranking is the
honest one: total outages are easy, gradual degradation is hard, and a frozen
irradiance sensor still fools it into reporting underperformance that is not
happening.

**These are not field-validated numbers.** They describe behaviour against
*injected* faults. No GridGuard alert has been confirmed against a real
maintenance record.

---

## Evaluation methodology

GridGuard reports two kinds of accuracy and never conflates them:

- **Forecast accuracy** on *measured* telemetry, where the target is genuine.
- **Detection accuracy** on *injected* faults, because measured telemetry has no
  trustworthy fault labels.

Detection faults are injected into the **held-out test split**, after training
and calibration, including into measured telemetry — so the generation, weather
and noise are real and only the failures are simulated.

Three injected classes are **not** generation losses (clipping, sensor dropout,
comm dropout). A detector that flags them is producing false alarms, and they
are scored that way. Without that, a detector that alarmed constantly would post
excellent recall.

---

## Ethical and practical considerations

**Institution names.** The simulated fleet uses real institution names (GMU,
NOVA, DC) for illustration. No named institution supplied telemetry or endorsed
this project, and every surface displaying simulated data says so explicitly.

**False alarms have a cost.** Each one is a wasted site visit and a small
withdrawal from operator trust. This is why the detector targets a *stated*
false-alarm rate rather than a threshold chosen for good-looking recall, why
event grouping requires sustained deviation, and why false alarms per day is a
headline metric.

**Missed detections have a larger one.** Undetected degradation compounds. The
~35% recall on gradual degradation is the most consequential weakness here and
is stated as such rather than buried.

**Attribution restraint.** Calling a deviation "site-specific" could send a
technician on a pointless trip. The classifier returns `indeterminate` rather
than guessing when neighbours are too few, and always shows the evidence.

---

## Known limitations

- One year, one site cluster, one climate. No cross-year or cross-climate
  generalisation is demonstrated.
- Coverage degrades under drift — observed, not hypothetical.
- Sensor-dropout false alarms are a real weakness: a frozen irradiance reading
  produces confident, wrong underperformance reports.
- The canopy array's dual east/west orientation is modelled as a single east
  mount, so its physics baseline is approximate.
- Physics assumptions (temperature coefficient, thermal model, inverter
  efficiency, DC/AC ratio) are generic crystalline-silicon defaults, not
  manufacturer data for these specific arrays. They are exposed on
  `/methodology`.
- No streaming. Batch evaluation only.

---

## Maintenance

Artifacts are rebuilt by `make build-artifacts`, which records the git commit,
library versions, dataset window and every metric in
`artifacts/models/manifest.json`. Calibration is recomputed on every build,
which matters: conformal validity depends on calibration data resembling what it
is applied to, and that resemblance decays with time.
