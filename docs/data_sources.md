# Data sources

GridGuard runs on two clearly-separated kinds of data. This document says
exactly what each is, where it comes from, and what was done to it.

---

## Measured: NREL PVDAQ via the OEDI data lake

### Why not the PVDAQ API

Earlier versions of this project targeted
`https://developer.nrel.gov/api/pvdaq/v3/...`. **Those endpoints have been
decommissioned.** The PVDAQ *data* remains public and is distributed through the
Open Energy Data Initiative (OEDI) data lake as an anonymously-readable S3
bucket, which is what GridGuard reads over plain HTTPS.

Two consequences, both good for a public demo:

- **No API key anywhere in the real-data path.**
- Data is partitioned one object per system per day, so a curated slice can be
  assembled without downloading the archive.

Landing page: <https://openei.org/wiki/PVDAQ>

### The systems

Three arrays on the NIST campus in Gaithersburg, Maryland — real installations
in the DMV region, all within 700 m of each other, which is what makes them
useful for demonstrating spatial attribution.

| System | Site id | Nameplate (DC) | Mount | Tilt / azimuth | Modules |
|---|---|---|---|---|---|
| 4901 | `nist_canopy` | 242.5 kW | Parking canopy | 5° / 90° (east) | 1032 × Sharp NU-U235F2 |
| 4902 | `nist_ground` | 270.7 kW | Fixed ground | 20° / 180° | 1152 × Sharp NU-U235F2 |
| 4903 | `nist_roof` | 73.7 kW | Low-tilt roof | 10° / 180° | 312 × Sharp NU-U235F2 |

Coordinates, elevation, tilt, azimuth, module model and DC nameplate all come
from each system's published metadata document, which is what lets the physics
baseline run on real geometry rather than assumptions.

> The canopy is a **dual east/west** array (mounts at azimuth 90° and 270°). The
> registry records the east mount only, so physics-model output for that site is
> approximate. This is recorded in its provenance and in the registry notes.

### Weather

These systems publish **their own** irradiance, ambient temperature and wind,
measured by instruments at the array. That is better than satellite or reanalysis
weather for this purpose — it is what the array actually experienced — and it is
why GridGuard needs no weather API.

### Layout

```
pvdaq/csv/systems.csv                                   system catalogue
pvdaq/csv/system_metadata/{id}_system_metadata.json     geometry, modules, inverter
pvdaq/parquet/metrics/metrics__system_{id}__part000.parquet   metric dictionary
pvdaq/parquet/pvdata/system_id={id}/year={Y}/month={M}/day={D}/
    system_{id}__date_{Y}_{MM}_{DD}.snappy.000.parquet  telemetry, long format
```

Note the asymmetry in the telemetry path: directory components are unpadded
(`month=6`) while the filename is zero-padded (`date_2016_06_01`).

Telemetry parquet is **long** — `(measured_on, utc_measured_on, metric_id,
value)` — and carries no channel names. The per-system **metrics table** is the
authority.

### Channel selection: by units, not by name

This is the part that silently produces wrong numbers if done casually. On
system 4902:

| Column name | What it actually is | Units |
|---|---|---|
| `ac_power_inv_14538` | instantaneous AC power ✅ | kW |
| `ac_power_meter_1864` | **reactive energy** ❌ | kVARh |
| `ac_power_meter_1864_2` | **cumulative energy** ❌ | kWh |
| `irradiance_ghi_o_2202` | raw pyranometer signal ❌ | **mV** |
| `irradiance_poa_o_2203` | raw pyranometer signal ❌ | **mV** |
| `irradiance_poa_o_2204` | reference cell ✅ | W/m² |
| `wind_speed_o_2206` | wind speed ✅ (filed under "AC other") | m/s |

So GridGuard selects on **declared units** first and treats names as a
tiebreaker:

- **AC power** — instantaneous power in `kW`, preferring an inverter source.
- **Irradiance** — plane-of-array in `W/m²` before global horizontal. Channels
  published in raw `mV` are **rejected**, not converted: the instrument
  calibration constant is not published, and inventing one would corrupt every
  downstream number.
- **Temperature** — ambient air in `C`.
- **Wind speed** — the averaging channel, not the max.

Each channel's `calc_scale` / `calc_offset` is applied, and the exact selection
is recorded in provenance so any number can be traced to the instrument that
measured it.

### Timezone

PVDAQ publishes both `measured_on` (local) and `utc_measured_on`. The offset is
**derived** from that pair rather than assumed, and confirmed constant across the
full year — UTC−5, local standard time, no daylight-saving shift.

### Validation

Every selected irradiance channel is checked against a pvlib Ineichen clear-sky
model **transposed into the sensor's own plane**. The 95th-percentile clear-sky
index must land in [0.55, 1.35]; outside that, the load fails loudly rather than
producing plausible-looking nonsense.

The transposition matters: comparing a 20°-tilted sensor against a *horizontal*
clear-sky reference gives a clear-sky index above 1.6 in winter, when the low sun
strikes the tilted plane far more directly. That check caught exactly that bug
during development.

### Processing

1. Fetch only the resolved channels (filtered before transfer).
2. Pivot long → wide, applying scale/offset.
3. Clamp negative irradiance and power to zero — pyranometers read slightly
   negative at night from thermal offset, and meters report small negative
   values from inverter tare draw. Neither is generation.
4. Resample 1-minute → 15-minute by mean.
5. Interpolate gaps up to 2 intervals; **leave longer gaps missing**. A
   communications outage is a real operating condition, not a value to impute.
6. Drop rows still missing power or irradiance, and record how many.

Coverage after processing, 2016:

| Site | Usable intervals | Dropped |
|---|---|---|
| `nist_ground` | 31,702 | 3,434 (9.8%) |
| `nist_roof` | 34,663 | 473 (1.3%) |
| `nist_canopy` | 35,136 | 0 |

### Licence

NREL PVDAQ data published via OEDI as a public dataset. Attribution to
NREL/OEDI required; consult the landing page for current terms before
redistribution.

### The curated copy

A year of processed 15-minute data for all three systems (~3 MB total) is
**committed** to `data/curated/`, each with its provenance record. That is what
lets a fresh clone run the real pipeline with no network access.

Regenerate with:

```bash
make data-real     # python scripts/download_data.py --refresh-curated
```

Review `git diff --stat data/curated/` before committing — these are the
repository's only committed data files.

---

## Simulated: the GridGuard DMV fleet

Seven illustrative campus sites across the DC/Maryland/Virginia region.

**Nothing here is measured data.** Institution names (GMU, NOVA, DC) are
illustrative; **no named institution supplied telemetry to this project**. Every
frame carries `data_mode="synthetic"`, and every surface that displays it says so.

### Why simulate at all

Because measured telemetry has no fault labels. Nobody annotated the NIST arrays
interval-by-interval, and deriving labels from the same residuals the detector
consumes would be circular. Detection performance can only be measured against
known ground truth, so ground truth has to be constructed.

### What the simulator models

1. Clear-sky GHI from pvlib's Ineichen model at the site's real coordinates and
   elevation.
2. Transposition to the plane of array using the site's tilt and azimuth.
3. Cloud attenuation as an **AR(1) process**, so cloudiness persists across
   intervals the way weather actually does. Independent per-sample noise would
   produce a shimmer no model could learn, making the task artificially hard and
   the residual distribution unrealistic.
4. Ambient temperature: seasonal + diurnal cycle for the Mid-Atlantic, plus noise.
5. AC power from POA irradiance with a temperature efficiency derate, system
   derate, and inverter cap.
6. A **slowly-drifting performance factor** — soiling, spectral mismatch, sensor
   drift. Its persistence is calibrated against the measured arrays' own daily
   autocorrelation (~0.6 at one day, <0.15 at a week) rather than guessed.

That last term is not decoration. Without it the simulated array is a
deterministic function of irradiance, which makes lagged power carry *no*
information — and the leakage hazard that motivates the whole weather-only
feature set could not even arise. Getting its persistence wrong also visibly
breaks conformal coverage, which is why it is calibrated against real data.

### Determinism

Fully reproducible. The per-site seed uses a stable digest, **not** Python's
builtin `hash()`, which is salted per process — using it made the "deterministic"
fleet silently different on every run. There is a regression test that spawns
subprocesses to confirm this.

### The window

2016-01-01 → 2016-12-31, deliberately **aligned with the measured data** so a
single fleet view can show both. Without alignment one fleet would always report
"no data". Simulated dates are arbitrary, so the simulation is what moves.

---

## Optional: NSRDB

[`src/gridguard/data/nsrdb.py`](../src/gridguard/data/nsrdb.py) implements NSRDB
PSM v3 access for PV systems that report generation but carry **no on-site
weather instrumentation**. GridGuard's shipped fleet does not need it.

If you extend the fleet to such a site:

```env
NREL_API_KEY=your-key        # free: https://developer.nrel.gov/signup/
NREL_API_EMAIL=you@example.com
```

The module raises rather than falling back to a demonstration key, which would
silently rate-limit. No key is committed anywhere in this repository.
