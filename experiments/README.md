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
