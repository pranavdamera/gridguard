# Demo script — 60 to 90 seconds

A deterministic walkthrough. Every number below is reproducible from a clean
clone: `make build-artifacts && make api && make web`.

The flow deliberately opens on a **real measured outage** rather than the
scripted one. Leading with genuine data is stronger, and the simulated event
then earns its place as the controlled case where ground truth is known.

---

## Setup

```bash
make build-artifacts     # ~4 min, no network or credentials needed
make api                 # terminal 1
make web                 # terminal 2
```

Open `http://localhost:3000`.

---

## The walkthrough

### 0:00 — Landing (10s)

> "GridGuard forecasts what a healthy solar array *should* produce, then
> explains why it didn't."

Point at the two cards: **3 arrays with measured telemetry** (NIST Gaithersburg,
via NREL PVDAQ) and **7 simulated DMV campus sites**. Say the distinction out
loud — it is the thing the whole project is organised around.

Click **Open the live fleet**.

### 0:10 — Fleet map (15s)

The map shows ten sites across the DC/Maryland/Virginia region on a real
basemap. Note without dwelling:

- Marker **size** is nameplate capacity, **colour** is health.
- **Solid ring = measured, dashed ring = simulated.**
- The view opens on the day of the fleet's most severe event, not on a
  months-long average — health is a statement about a moment.

One site is pulsing red: **NIST Gaithersburg — Ground Array**.

### 0:25 — A real outage (20s)

Click the red marker, then **Open site analytics**.

> "This is measured data. On 15 October 2016 this 270 kW array produced
> essentially nothing during daylight."

On the chart, point at:

- The **white line** (actual) collapsing to the floor.
- The **dashed cyan line** (expected) continuing at full height.
- The **shaded band** — the calibrated range of healthy output.

> "The band isn't decoration. It's a conformal prediction interval: at α = 0.05,
> at most about 5% of healthy intervals should fall below it. So 'outside the
> band' is a statement with a false-alarm rate attached, not a threshold someone
> picked by eye."

### 0:45 — Was it the array, or the weather? (20s)

Open the event from the site page.

> "A drop like this has two very different explanations, and they call for
> opposite responses."

Point at the spatial attribution panel: **Likely site-specific**.

> "GridGuard checked the two other NIST arrays — 400 metres away, same sky. They
> were producing normally. A cloud does not stop at one array and spare its
> neighbours, so this is equipment, not weather. Dispatch someone."

Show the neighbour comparison table: the canopy and roof arrays near 100% of
expected while the ground array sits at 0%.

### 1:05 — How it knows (10s)

Scroll to **What drove the expectation** (SHAP contributors) and the
**Suggested next steps**.

> "The explanation is scoped honestly: these features explain why the model
> expected what it did. They don't diagnose which component failed — GridGuard
> detects and localises, it doesn't diagnose."

### 1:15 — The controlled case (15s)

Back to the fleet, open **GMU Fairfax** (simulated).

> "Measured data has no fault labels, so detection accuracy can't be measured on
> it directly. GridGuard injects known faults — eight classes, including three
> that are *not* generation losses, like inverter clipping — and scores itself
> against those. Injected into the measured data too, so the generation and
> weather stay real and only the failures are simulated."

Open **/methodology** and point at the fault taxonomy, then **/data** for
provenance down to the instrument and channel.

---

## What to say if asked

**"Is this real data?"**
Three of the ten sites are real measured arrays — NIST's campus PV in
Gaithersburg, from NREL's PVDAQ collection, a full year at 15-minute resolution
with weather from instruments at the array. The other seven are simulated, and
labelled as such everywhere they appear.

**"Why simulate anything?"**
Because measured telemetry carries no fault labels. Forecast accuracy is
measured on the real arrays; detection accuracy needs known ground truth, so
faults are injected.

**"Does it need an API key?"**
No. The legacy PVDAQ REST API was decommissioned; the data is now distributed
through the OEDI data lake and read over anonymous HTTPS. The arrays publish
their own weather, so no NSRDB request is needed either.

**"How accurate is it?"**
Forecast error on the measured arrays is reported per-site on the site page,
read from the artifact manifest. Detection metrics are per fault class on
`/methodology`. Both come from the build that produced the loaded models — no
hand-entered numbers anywhere.

---

## Recording notes

- The build is deterministic: the same commit produces the same events, the same
  demo date, and the same numbers.
- Take the fleet map at ≥1440px wide; the marker legend crowds below ~900px.
- `/health` shows the artifact commit, which is worth a frame if the audience is
  technical.
