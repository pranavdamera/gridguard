# Methodology

How GridGuard decides that an array underperformed, how confident it is, and what
it refuses to claim.

---

## 1. Two feature sets, and why the split exists

| Mode | Purpose | Lagged power? |
|---|---|---|
| `weather_only` | Expected generation — the detection baseline | **No** |
| `lag_aware` | Short-horizon forecasting | Yes |

Consider an array that has been running at 50% for a week.

With **lag-aware** features, the model sees `ac_power_lag1` ≈ half of normal and
predicts accordingly. Residual ≈ 0. The detector reports everything is fine. The
fault is invisible — not because detection is badly tuned, but because the model
was asked the wrong question. It answered "what will this array produce next?",
which for a degraded array is "not much", correctly.

With **weather-only** features, the model can only answer "what *should* an array
produce in this weather?" It keeps predicting healthy output, the residual stays
large and negative, and the fault stays visible for as long as it persists.

Two safeguards enforce this:

1. The detector is structurally incapable of seeing lagged power — those columns
   are not in its feature set.
2. It is trained on **healthy intervals only**, so known faults never enter its
   notion of normal.

Both are covered by tests, including one that demonstrates the masking mechanism
directly.

> **On effect size.** In GridGuard's simulation the masking effect is directional
> but modest (a few percent), because irradiance explains most of the variance,
> leaving lagged power little to add. The hazard scales with how much output
> variance the weather channels *fail* to explain — soiling, partial shading,
> inverter derating, sensor drift — which is larger on real installations. The
> tests assert direction and structural immunity, not a magnitude the data would
> not support.

---

## 2. Temporal splits

Everything is split by time, never randomly.

```
|<------------ train ------------>|<-- val -->|<------ test ------>|
                                   ^^^^^^^^^^^
                          early stopping + conformal calibration
```

A random split on time series puts intervals from the same afternoon on both
sides, which inflates every score. XGBoost early stopping uses the validation
split, never the test split — using test data for model selection makes it part
of training.

---

## 3. Expected generation

Four families, evaluated together:

| Model | What it is |
|---|---|
| Persistence | Naive lag-1. The floor any model must beat. |
| Ridge | Regularised linear on all features. |
| RandomForest / XGBoost | Non-linear irradiance × temperature interactions. |
| **pvlib physics** | POA irradiance → cell temperature (Sandia) → DC power (PVWatts) → AC power with inverter clipping. |
| **Physics + ML hybrid** | Gradient boosting on the physics *residual*. |

The physics baseline earns its place for three reasons a statistical model
cannot match:

1. **It cannot learn a fault as normal.** It has no memory of history, so it
   always answers "what should this produce in this weather?"
2. **It transfers to a site with no history.** A newly commissioned array has no
   training data but known geometry on day one.
3. **It is auditable.** Every term maps to a documented physical effect with a
   stated assumption.

It fits exactly **one** free parameter — an overall derate absorbing the gap
between generic loss assumptions and this array's actual performance. One free
parameter keeps the comparison against tree ensembles fair without turning the
baseline into a fitted model in disguise.

The hybrid ordering matters for detection: the learned component only ever sees
the *residual* of a physical model, so it has far less capacity to absorb a real
fault as normal than a model fitted directly to power.

---

## 4. Conformal prediction

### The guarantee

Given calibration residuals exchangeable with future residuals, the lower bound
*q* satisfies

```
P(actual − predicted ≥ q) ≥ 1 − α
```

with **no distributional assumption**. At α = 0.05, at most ~5% of healthy
intervals should fall below the bound. That is a design parameter you set, not a
property you hope for.

### Why not a sigma threshold

The legacy detector flags when the residual falls below −*k*σ. That implies a
false-alarm rate *only if residuals are Gaussian*, and PV residuals emphatically
are not — they are heavy-tailed and sharply skewed by cloud transients.

On heavy-tailed residuals, "2σ" is badly miscalibrated in a way that is easy to
miss: outliers inflate the σ that defines the threshold, so the bound drifts deep
into the tail and covers ~99% instead of the ~95% the choice of *k* implies. That
is not usefully conservative — it is silently missing faults. The sigma detector
is retained as a benchmark so the comparison stays visible.

### Mondrian calibration

A single global bound would be far too wide at dawn and far too tight at noon,
because residual spread scales with available irradiance. Bounds are therefore
calibrated per irradiance bucket:

| Bucket | Range (W/m²) |
|---|---|
| night | < 50 |
| low | 50 – 250 |
| medium | 250 – 550 |
| high | 550 – 850 |
| peak | ≥ 850 |

Bucketing on irradiance rather than clock hour makes the conditioning physical,
so it transfers across seasons and latitudes instead of encoding one site's
daylight pattern. Buckets with fewer than 100 calibration points fall back to the
global bound — a quantile from a dozen samples is noise wearing the costume of a
guarantee.

### Calibration data

Calibration uses **held-out** residuals, from the most recent healthy validation
intervals.

Both halves of that are load-bearing, and both were established empirically:

- Calibrating on the model's own **training** residuals produces bounds that are
  too tight, because in-sample errors are biased small. Coverage fell below the
  stated guarantee until this was fixed.
- Calibrating on data **further away in time** also degrades coverage, because
  array performance drifts and exchangeability decays with separation. Widening
  the validation window — which increases that separation — made coverage worse
  across the fleet, not better.

### What the guarantee does not say

1. **Coverage is marginal, not conditional.** It holds on average within each
   bucket, not for every individual interval.
2. **Exchangeability is an assumption about the world.** Degradation, sensor
   drift and seasonal shift all break it. Calibration is recomputed per build and
   its window recorded in the manifest — and two simulated sites currently sit
   below target, which is this limitation being visible rather than hidden.
3. **A breach is not a diagnosis.** It means the interval fell outside the
   calibrated range of healthy behaviour. Cause is a separate question.

---

## 5. Events

Contiguous flagged intervals are grouped into events, because operators think in
"a two-hour outage", not "312 flags".

**Persistence is required.** The detector is *designed* to flag α (5%) of healthy
intervals — that is what the coverage guarantee means. Promoting every isolated
flag to an event yields roughly one "event" per twenty healthy daylight
intervals, burying real faults under hundreds of single-interval blips (observed:
~1000 events, of which one was real). Requiring three consecutive intervals
converts a per-interval false-positive rate into a far lower per-event one — for
roughly independent intervals, about α³ — and matches the physics: a cloud edge
produces one bad interval, an inverter fault persists.

**Severity is a fraction, not a quantity.** An absolute kWh threshold is
meaningless across a fleet spanning 60 kW to 500 kW — 10 kWh lost is a rounding
error for one and a total outage for the other. Severity is the fraction of
expected generation lost: ≥50% high, ≥20% medium, below that low, with a small
absolute floor so an energetically trivial blip is never escalated.

---

## 6. Spatial attribution

A single-site detector can say an array underperformed. It cannot say why, and
the two explanations demand opposite responses:

- **Site-specific** — this array underperformed while its neighbours, under the
  same sky, did not. Dispatch someone.
- **Regional** — several nearby arrays dropped together. The weather model was
  wrong, not the hardware. Do nothing.

### Method

Haversine great-circle distance on a spherical Earth. At fleet scale the
difference from WGS-84 is under 0.5%, far below the uncertainty in whether two
arrays share weather at all.

Residuals are normalised by nameplate capacity, making a 70 kW roof and a 270 kW
field directly comparable. Neighbour agreement is measured both as an unweighted
median (robust to one odd neighbour) and inverse-distance weighted (closer sites
share weather more often).

A deviation is called **site-specific** when the site is worse than its
neighbourhood median by at least 5 percentage points of nameplate *and* fewer
than half its neighbours are also affected. It is called **regional** when at
least half the neighbours dropped and this site did too. Otherwise:
`indeterminate`.

### Constraints

- **Neighbours must share the site's data mode.** The simulator draws each site's
  weather independently, so a simulated neighbour carries no information about a
  measured array's cloud field. Mixing them manufactures agreement out of noise.
- **Fewer than two neighbours → no verdict.** `indeterminate`, with an
  explanation, rather than a guess.
- **The evidence is always reported**: this site's residual, the neighbour
  median, the distance-weighted residual, how many neighbours were affected, and
  the excess deviation.

### What it is not

Not a fault classifier. It does not diagnose inverter failure, soiling or
shading, and makes no claim to. It answers one narrower question and shows its
working so a human can disagree.

---

## 7. Detection evaluation

Measured PV telemetry has **no trustworthy fault labels**. Deriving labels from
the same residuals the detector consumes would be circular. So:

- **Forecast accuracy** → measured on *real* telemetry, where the target is
  genuine.
- **Detection accuracy** → measured against *injected* faults, where ground truth
  is known by construction.

Faults are injected into the **held-out test split**, after training and
calibration — including into measured telemetry, so generation, weather and noise
stay real and only the failures are simulated.

### Taxonomy

Eight classes with distinct signatures. Five are generation losses:

`complete_outage` · `partial_outage` · `persistent_derate` · `shading` ·
`gradual_degradation`

Three are **not**, and are scored as negatives:

| Class | Why it is not a loss |
|---|---|
| `clipping` | Normal, healthy inverter behaviour on high irradiance |
| `sensor_dropout` | Irradiance sensor freezes; generation continues normally |
| `comm_dropout` | Telemetry stops; no energy is lost |

Scoring these separately is what stops headline recall from hiding a detector
that simply alarms constantly. It also surfaces a genuine weakness: GridGuard
does false-alarm on frozen irradiance sensors.

### Metrics

Interval-level precision/recall/F1; **event-level recall** (closer to how an
operator experiences the system — catching a four-hour outage twenty minutes late
is a success, not 80 failures); detection latency; false alarms per day over
healthy periods; and lost-energy error, computed exactly because the generator
retains pre-fault generation in `ac_power_baseline_kw`.

---

## 8. What GridGuard does not claim

- It does not identify which component failed.
- Its detection numbers are not field-validated against maintenance records.
- It does not estimate financial loss (no electricity prices are assumed —
  fabricating them would be worse than omitting them).
- It does not generalise across climates or years on the evidence presented.
- It is not a basis for dispatch, maintenance, or financial decisions without
  independent validation.
