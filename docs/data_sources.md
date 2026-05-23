# Data Sources

## Supported sources

### 1. Synthetic (default)

No setup needed. Generates 2 years of 15-minute physically-motivated data.

```bash
python scripts/download_data.py --source synthetic
python scripts/download_data.py --source synthetic --site-id gmu_fairfax
python scripts/download_data.py --list-sites  # show all DMV sites
```

**What the simulator models:**
- Sun elevation angle from latitude and day-of-year (no pvlib dependency)
- Clear-sky GHI × log-normal cloud factor
- Temperature: seasonal mean + diurnal ±5°C swing + Gaussian noise
- Panel efficiency: −0.4%/°C above 25°C (standard datasheet specification)
- Injected faults on 5% of days (40–80% output reduction, random severity)

**Limitations:** No bifacial gain, no row-to-row shading, no spectral effects,
no soiling ramp. The generator is good enough for model development and demo,
not for yield studies.

### 2. NREL PVDAQ

Real residential and commercial PV systems in the US. Free API key required.

1. Register at https://developer.nrel.gov/signup/
2. Set `NREL_API_KEY=your_key` in `.env`
3. Optionally set `PVDAQ_SYSTEM_ID` (default 2 — a well-documented 5 kW residential system)

```bash
python scripts/download_data.py --source nrel
```

Browse available systems (metadata, location, capacity):
https://developer.nrel.gov/docs/solar/pvdaq-v3/

**Note:** The DEMO_KEY has rate limits. For large downloads get a personal key.

### 3. NREL NSRDB — National Solar Radiation Database (future)

High-resolution (4 km, 30-min) satellite-derived solar resource data for the US.

- API: `https://developer.nrel.gov/api/solar/nsrdb_psm3_download.json`
- Variables: GHI, DNI, DHI, temperature, wind speed, cloud type
- Coverage: 1998–present (updated annually)
- Free with NREL API key; bulk download limits apply

**Why this matters for GridGuard:** The current synthetic generator approximates sun
geometry from latitude alone. NSRDB provides measured irradiance at each site location,
which would improve the `weather_only` expected-generation model significantly —
especially for cloudy days and winter periods where the simple geometric model diverges
from reality.

**Integration path:** Add `nsrdb` source to `load_raw_data()`. The PSM3 API returns
a CSV with standardised column names that would map cleanly to GridGuard's schema.

### 4. PJM Data Miner — Grid Context (future)

PJM operates the wholesale electricity market for DC, Maryland, Virginia, and 12 other
states. Real-time and day-ahead LMP (locational marginal price) data is publicly available.

- URL: `https://dataminer2.pjm.com/`
- API: REST, no authentication required for public data
- Variables: LMP ($/MWh) by node and 5-min interval; load forecasts; interchange

**Why this matters for GridGuard:** LMP transforms "23 kWh lost" into "23 kWh × $85/MWh
= $2 of lost revenue." During peak-price hours, a fault is economically 5–10× more
impactful than during off-peak. Integrating LMP makes severity rankings actionable for
energy managers, not just O&M teams.

**Integration path:** Add a `pjm_lmp` module that fetches 5-min LMP for the nearest
node to each site, joins by timestamp, and adds `lmp_dollar_per_mwh` to the anomaly
DataFrame. Update `compute_daily_loss()` to compute `lost_revenue_usd`.

### 5. Open Power System Data (future)

Hourly national solar generation for European countries (Germany, France, GB, etc.).

- URL: https://open-power-system-data.org/data-packages/time_series
- Download the `time_series_60min_stacked.csv` file (~500 MB)
- Add an adapter in `ingestion/download.py` that parses it and normalises column names

### 4. Ausgrid Solar Home Dataset (future)

30-minute interval data for ~300 Australian residential customers with rooftop solar.

- URL: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
- Useful for studying household-scale fault patterns
- Requires column normalisation (different schema from PVDAQ)

### 5. PVOutput.org (future)

Community-contributed live and historical PV system data. Free API for non-commercial use.

- URL: https://pvoutput.org/help/api_specification.html
- 5-minute resolution available for some systems
- Good for real-time demo mode (poll every 5 minutes)

## DMV Site Registry

Sites in `config/sites.csv` are illustrative — capacity and exact coordinates
are approximate values based on public records and satellite imagery, not
official specifications. They are provided for demonstration purposes only.

| Site | Capacity (kW) | Notes |
|---|---|---|
| GMU Fairfax | 250 | Estimated from campus sustainability reports |
| NOVA Annandale | 80 | Illustrative |
| NOVA Alexandria | 60 | Illustrative |
| NOVA Loudoun | 120 | Illustrative |
| NOVA Manassas | 75 | Illustrative |
| NOVA Woodbridge | 90 | Illustrative |
| DC Community | 500 | Placeholder for community solar aggregation |

To add a real site, append a row to `config/sites.csv` — no code changes needed.
The registry loader is `src/gridguard/sites/registry.py`.
