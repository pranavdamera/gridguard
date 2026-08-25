# Experiments

Offline research work: analyses that belong in a report, not in the operational
API. Everything here reads prebuilt artifacts and writes to
`artifacts/reports/`. Nothing here is on the serving path, and nothing here is
imported by `src/gridguard/api/`.

```bash
pip install -e ".[experiments]"    # adds matplotlib
make build-artifacts               # once, if you have not already
make diagnostics                   # or: python experiments/run_diagnostics.py
```

## `run_diagnostics.py`

Per-site diagnostic figures and the CSVs behind them, written to
`artifacts/reports/diagnostics/<site_id>/`.

| Output | What it shows |
| --- | --- |
| `power_curve.{png,csv}` | AC power against irradiance, with flagged intervals overlaid. Underperformance appears as points below the main locus at a given irradiance. |
| `power_curve_summary.csv` | The same data binned by irradiance: healthy median, flagged median, and the gap between them. A working detector separates the two medians. |
| `daily_loss.{png,csv}` | Energy attributed to detected loss, by day. |
| `feature_importance.{png,csv}` | Global importance by mean absolute SHAP value over the site's own data. |
| `clear_sky_projection.{png,csv}` | Generation ceiling for the day after the data ends, under clear skies. |
| `index.csv` | One row per site, summarising all of the above. |

Pass `--no-figures` to write CSVs only, which needs no matplotlib. Pass
`--site <id>` (repeatable) to restrict which sites run.

### Reading the clear-sky projection honestly

It is a **physical ceiling, not a forecast**. It answers "what could this site
produce tomorrow if nothing were in the way" — irradiance comes from pvlib's
Ineichen clear-sky model transposed into the array plane, and temperature from a
seasonal climatology rather than any weather service. Actual output will sit at
or below this line whenever there is cloud. Every projection frame carries an
`attrs["assumptions"]` dict saying exactly this, and the CSV is written beside
the figure so the assumption travels with the number.

## `run_simulation.py`

Runs the layered site simulator and writes three views of the same run, kept
separate on purpose:

| Output | What it is |
| --- | --- |
| `truth.parquet` | What physically happened, per asset: real weather, real generation, real equipment state. The answer key. Never fed to a model. |
| `observed.parquet` | What the instruments reported. What a model would see if the network were perfect. |
| `canonical.parquet` | What reached the collector, in the shipped schema. What a model actually sees. |
| `run.json` | Config, seed, fingerprint, throughput, and a digest of each frame. |

```bash
python experiments/run_simulation.py --days 30                    # 7 sites, ~1s
python experiments/run_simulation.py --site nist_roof --seed 3
python experiments/run_simulation.py --days 1 --speedup 40000 --replay
```

### Why three frames and not one

Three different things can make a value wrong or absent at the collector: the
plant produced nothing, the instrument failed, or the reading never arrived.
An architecture that merges them cannot tell them apart, and telling them apart
is the research question. So the simulator keeps a layer boundary at each
transition and retains the output of each.

`result.observation_error()` is the direct consequence: the measured gap between
what happened and what was reported. A phase 8 attribution experiment scores a
sensor fault against that baseline rather than estimating it.

### Injected faults

`--inject-faults` schedules the full taxonomy against each site's real assets.
The key difference from the batch injector in `gridguard.data.faults` is that a
fault is applied by **the layer whose behaviour it changes**:

| Fault | Layer | What it changes |
| --- | --- | --- |
| `complete_outage`, `partial_outage`, `persistent_derate`, `gradual_degradation` | equipment | Asset state, and therefore real generation |
| `shading` | equipment | Real generation, within the same hours each day |
| `clipping` | equipment | AC cap. Healthy behaviour, not a fault |
| `sensor_dropout` | sensing | A channel freezes. Ground truth untouched |
| `comm_dropout` | transport | Records never arrive. Truth *and* observation untouched |

That routing is the whole point. Rewriting `ac_power_kw` to `NaN` for a lost
message — which is what a single-frame injector must do — makes it
indistinguishable from a dead inverter that reported honestly. The information
that tells them apart is destroyed at the moment of injection. Here the plant
keeps generating, the meter keeps reading correctly, and only delivery changes.

`result.fault_summary()` reports what each fault actually cost. `lost_kwh` is
exact rather than modelled: it is the recorded gap between potential and actual
generation, which exists only because the equipment layer keeps both.

Two results in that table are worth reading carefully:

- **`sensor_dropout` and `comm_dropout` cost 0 kWh.** Correct — neither touches
  the plant.
- **`clipping` costs real energy but is not a generation loss.** Not a
  contradiction. `is_generation_loss` means "the detector should flag this",
  which is false for clipping; it is not a claim that no energy was withheld.

Two defaults in the scenario builder exist because the obvious version of each
silently tested nothing:

- Fault windows are anchored to solar noon. Placed by tick arithmetic alone they
  landed mostly at night, where a fault costs nothing and exercises no detector —
  shading cost exactly zero, and a 45% partial outage over twelve hours cost 9 kWh.
- The clipping cap is a percentile of the array's *own* daylight potential, not a
  fraction of nameplate. These arrays peak near 75% of DC nameplate after derate
  and temperature losses, and how far below varies with tilt and latitude, so a
  nameplate-relative cap engaged at some sites and sat above the peak at others.

These are **simulated failure modes, not observed field incidents.** No rate
derived from them describes real equipment.

### Speed and reproducibility

A run is a pure function of `(SimulationConfig, seed)` — no wall-clock, no
environment, no process-dependent hashing. Every layer of every asset draws from
its own named random stream, so enabling a fault in one layer provably cannot
perturb another. Without that, injecting a sensor fault would come bundled with
different weather and no detector response could be attributed to either; the
test suite asserts it directly.

Computation is vectorised — 7 sites over 30 simulated days takes about a second,
roughly two million times real time. Pacing is therefore applied to `--replay`,
*after* the values exist, which makes "pacing cannot change results" true by
construction rather than by discipline. `run.json` records the frame digests so
a paced demo and an unpaced experiment can be checked against each other by eye.

---

## Relationship to the retired Streamlit dashboard

GridGuard used to carry a second UI in `dashboard/app.py`. It was retired
because it duplicated the web application's content while being untested, and
would have diverged from the domain model as assets were introduced.

Most of what it displayed already existed elsewhere — actual-vs-predicted
charts, the event queue, model metrics, the fleet map all have API and web
equivalents. Two of its analyses were computed by package functions that
happened to be surfaced nowhere else (`compute_daily_loss` and
`global_feature_importance`); those are now called from here. The power curve
was genuinely absent from the project and is new code in
`src/gridguard/analysis/diagnostics.py`.

One thing deliberately did **not** carry over unchanged. The dashboard's
next-day forecast computed solar declination, hour angle and a cosine-of-zenith
approximation inline, duplicating — and diverging from — the pvlib model the
rest of the project uses. The projection here is rebuilt on that shared model,
so its numbers will not match the dashboard's. That is the point.

---

## Planned: conformal validity under degraded telemetry

**Not yet run. No results exist for this, and none are claimed.** This section
specifies the experiment so that it stays an explicit research question rather
than being quietly designed out of the system.

### The question

Split conformal prediction guarantees `P(actual − predicted ≥ q) ≥ 1 − α` given
one assumption: that calibration residuals are **exchangeable** with the
residuals the bound will be applied to. GridGuard already knows that assumption
strains under performance drift — widening the validation fraction from 0.15 to
0.25 measurably *worsened* fleet coverage, because a longer calibration window
spans more drift.

Distributed sensing breaks exchangeability a second, independent way. Samples go
missing, arrive late, or arrive corrupted — and **which** samples that happens to
is not random with respect to site conditions. A link saturating under load, or
a sensor drifting in heat, correlates with precisely the intervals a detector
cares about. So the question is not whether the guarantee degrades. It is how
fast, in which direction, and whether any of it is recoverable.

### Design

Hold the model, the calibration set and the test split fixed. Vary one
degradation regime at a time over the held-out period, and re-measure empirical
coverage per irradiance bucket:

| Regime | Parameter swept |
| --- | --- |
| Random sample loss | Fraction dropped, 0 → 50% |
| Bursty loss | Outage length, 1 → 96 intervals, at fixed total loss |
| Delayed arrival | Arrival lag, 0 → 24 h, scored at a fixed decision deadline |
| Frozen sensor | Fraction of intervals with a stale irradiance channel |
| Correlated loss | Loss probability conditioned on irradiance, to break the missing-at-random assumption on purpose |

The reported output is a **coverage curve per regime**, with the nominal
`1 − α` line drawn on it, and the gap between marginal and per-bucket coverage
shown separately — Mondrian calibration restores the guarantee within buckets
under exchangeability, and whether it still does under loss is its own question.

### Rules for this experiment

1. **Report degradation; do not patch it.** Inflating the interval until the
   guarantee reappears destroys the finding.
2. **Bursty and correlated loss are the interesting arms.** Random loss is the
   easy case and is included only as a control.
3. Numbers go in the report only from a recorded run, with the manifest commit
   attached — as with everything else here.

Depends on the edge agent and collector (phases 5–7), which is what will make
these regimes reproducible rather than merely simulated in a notebook.
