"""
GridGuard Streamlit dashboard.

Sections:
  1. Header + system summary KPIs
  2. Actual vs Predicted generation (time series)
  3. Daily energy loss chart
  4. Anomaly Events table (grouped consecutive intervals)
  5. Model performance comparison + stratified metrics
  6. Feature importance / SHAP explain panel
  7. Weather drivers overlay

Run:
    streamlit run dashboard/app.py

TODO: Add date-range picker sidebar filter.
TODO: Add live-data mode that polls /forecast every 15 minutes.
TODO: Add map view for multi-site deployments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridguard.anomaly.detect import compute_daily_loss, detect_anomalies
from gridguard.anomaly.events import group_anomaly_events
from gridguard.config import settings
from gridguard.features.engineer import get_X_y
from gridguard.models.evaluate import compare_models, stratified_metrics
from gridguard.models.train import load_model
from gridguard.sites.registry import load_sites

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GridGuard DMV — Solar Asset Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_COLOR = "#F5A623"
ANOMALY_COLOR = "#E74C3C"
PRED_COLOR = "#3498DB"
ACTUAL_COLOR = "#2ECC71"
SEVERITY_COLOR = {"high": "#E74C3C", "medium": "#F39C12", "low": "#27AE60"}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models …")
def load_artifacts():
    """Load models. Cached across reruns."""
    models = {}
    from gridguard.models.baseline import ALL_MODELS

    for name in ALL_MODELS:
        try:
            models[name] = load_model(name)
        except FileNotFoundError:
            pass

    test_path = Path(settings.data_processed_dir) / "test_df.parquet"
    if not test_path.exists():
        return None, None, None

    test_df = pd.read_parquet(test_path)
    return test_df, models, None


def compute_all(test_df, models, threshold):
    """Run anomaly detection + event grouping with the current sidebar threshold."""
    best_model = models.get("xgboost") or next(iter(models.values()))
    anomaly_df = detect_anomalies(test_df, best_model, threshold_sigma=threshold)
    daily_df = compute_daily_loss(anomaly_df)
    events_df = group_anomaly_events(anomaly_df)
    X_test, y_test = get_X_y(test_df)
    metrics_df = compare_models(models, X_test, y_test)
    return anomaly_df, daily_df, events_df, metrics_df, best_model


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/solar-panel.png", width=80)
    st.title("GridGuard")
    st.caption("Solar Asset Intelligence · DMV Region")
    st.divider()

    threshold = st.slider(
        "Anomaly threshold (σ)",
        min_value=1.0,
        max_value=4.0,
        value=float(settings.anomaly_threshold_sigma),
        step=0.25,
        help="Flag intervals where residual < −threshold × σ. Lower = more sensitive.",
    )

    # Site selector (informational for now — future: load per-site data)
    try:
        sites = load_sites()
        site_names = {s.site_id: s.name for s in sites.values()}
        site_options = ["(all / default)"] + sorted(site_names.values())
        selected_site_name = st.selectbox("Site filter", site_options, index=0)
        selected_site = next(
            (sid for sid, nm in site_names.items() if nm == selected_site_name), None
        )
    except Exception:
        selected_site = None

    st.divider()
    st.markdown("**Data source**")
    st.code(settings.data_source, language=None)
    st.markdown("**Links**")
    st.markdown("- [API docs](http://localhost:8000/docs)")
    st.markdown("- [GitHub](https://github.com/YOUR_USERNAME/gridguard)")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("⚡ GridGuard DMV — Solar Asset Intelligence")
st.caption(
    "Forecasts solar generation · Detects underperformance · Estimates lost energy · "
    "DC / Northern Virginia campus sites · "
    "Synthetic data unless real telemetry is configured"
)

test_df, models, _ = load_artifacts()

if test_df is None or not models:
    st.error("No model artifacts found. Run `make train` to generate them, then refresh.")
    st.code("make data-synthetic\nmake train", language="bash")
    st.stop()

anomaly_df, daily_df, events_df, metrics_df, best_model = compute_all(test_df, models, threshold)

# Apply site filter if a specific site is selected
if selected_site and "site_id" in anomaly_df.columns:
    anomaly_df = anomaly_df[anomaly_df["site_id"] == selected_site]
    daily_df = compute_daily_loss(anomaly_df)
    events_df = group_anomaly_events(anomaly_df)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

n_anomalies = int(anomaly_df["is_anomaly"].sum())
total_lost_kwh = anomaly_df["lost_energy_kwh"].sum()
total_actual_kwh = (anomaly_df["ac_power_kw"] * 0.25).sum()
fault_days = int((daily_df["anomaly_count"] > 0).sum())
best_rmse = metrics_df.iloc[0]["rmse_kw"]
n_events = len(events_df)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Anomaly events", f"{n_events:,}")
col2.metric("Anomaly intervals", f"{n_anomalies:,}")
col3.metric("Lost energy", f"{total_lost_kwh:.1f} kWh")
col4.metric("Fault days", f"{fault_days}")
col5.metric("Generation", f"{total_actual_kwh/1000:.1f} MWh")
col6.metric("Best RMSE", f"{best_rmse:.3f} kW")

st.divider()

# ---------------------------------------------------------------------------
# Section 1: Actual vs Predicted
# ---------------------------------------------------------------------------

st.subheader("Actual vs Predicted Generation")

# Limit to 1 week for rendering performance
plot_df = anomaly_df.head(4 * 24 * 7)

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(x=plot_df["timestamp"], y=plot_df["ac_power_kw"],
               name="Actual", line=dict(color=ACTUAL_COLOR, width=1), opacity=0.8)
)
fig1.add_trace(
    go.Scatter(x=plot_df["timestamp"], y=plot_df["predicted_kw"],
               name="Predicted", line=dict(color=PRED_COLOR, width=1.5, dash="dot"))
)
anom_plot = plot_df[plot_df["is_anomaly"]]
fig1.add_trace(
    go.Scatter(x=anom_plot["timestamp"], y=anom_plot["ac_power_kw"],
               mode="markers", name="Anomaly",
               marker=dict(color=ANOMALY_COLOR, size=6, symbol="x"))
)
fig1.update_layout(
    xaxis_title="Timestamp", yaxis_title="AC Power (kW)",
    legend=dict(orientation="h", y=1.02), height=380,
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Daily energy loss
# ---------------------------------------------------------------------------

st.subheader("Daily Energy Loss from Anomalies")

loss_plot = daily_df[daily_df["lost_energy_kwh"] > 0].tail(60)
if not loss_plot.empty:
    fig2 = px.bar(
        loss_plot, x="date", y="lost_energy_kwh",
        color="loss_pct", color_continuous_scale="Reds",
        labels={"lost_energy_kwh": "Lost Energy (kWh)", "loss_pct": "Loss %"},
        height=300,
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No energy loss detected at this threshold.")

# ---------------------------------------------------------------------------
# Section 3: Anomaly Events table
# ---------------------------------------------------------------------------

st.subheader("Anomaly Events (grouped consecutive intervals)")

if events_df.empty:
    st.info("No anomaly events at the current threshold.")
else:
    # Colour-code severity
    def severity_badge(s: str) -> str:
        colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        return f"{colors.get(s, '')} {s}"

    display_events = events_df.head(30).copy()
    display_events["severity"] = display_events["severity"].apply(severity_badge)
    display_events["start_time"] = display_events["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    display_events["end_time"] = display_events["end_time"].dt.strftime("%Y-%m-%d %H:%M")
    display_events = display_events.rename(columns={
        "event_id": "ID", "start_time": "Start", "end_time": "End",
        "duration_minutes": "Duration (min)", "interval_count": "Intervals",
        "total_lost_kwh": "Lost (kWh)", "max_residual_sigma": "Max σ",
        "mean_actual_kw": "Actual (kW)", "mean_predicted_kw": "Predicted (kW)",
        "severity": "Severity", "explanation": "Explanation",
    })
    cols_to_show = ["ID", "Start", "End", "Duration (min)", "Lost (kWh)", "Max σ", "Severity", "Explanation"]
    if "site_id" in display_events.columns:
        cols_to_show.insert(1, "site_id")
    # Only include columns that actually exist (explanation added in events.py)
    cols_to_show = [c for c in cols_to_show if c in display_events.columns]
    st.dataframe(display_events[cols_to_show], use_container_width=True)

    # Show plain-English explanation for the most recent high-severity event
    high_events = events_df[events_df["severity"] == "high"]
    if not high_events.empty and "explanation" in high_events.columns:
        top = high_events.iloc[0]
        st.info(f"**Most recent high-severity event:** {top['explanation']}")

    # Severity breakdown
    sev_counts = events_df["severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig_sev = px.bar(
        sev_counts, x="Severity", y="Count",
        color="Severity",
        color_discrete_map={"high": ANOMALY_COLOR, "medium": "#F39C12", "low": ACTUAL_COLOR},
        height=220,
    )
    fig_sev.update_layout(margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig_sev, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Model performance + stratified metrics
# ---------------------------------------------------------------------------

st.subheader("Model Performance")

left, right = st.columns(2)

with left:
    st.markdown("**Overall metrics (test set)**")
    display_metrics = metrics_df.rename(columns={
        "model": "Model", "mae_kw": "MAE (kW)", "rmse_kw": "RMSE (kW)",
        "mape_pct": "MAPE (%)", "r2": "R²",
    })
    st.dataframe(display_metrics.set_index("Model"), use_container_width=True)

with right:
    st.markdown("**Stratified metrics**")
    strata_choice = st.selectbox(
        "Break down by",
        options=["hour_of_day", "season", "irradiance_bin"],
        index=0,
    )
    try:
        strat_df = stratified_metrics(test_df, best_model, strata=strata_choice)
        st.dataframe(
            strat_df.rename(columns={
                strata_choice: strata_choice.replace("_", " ").title(),
                "mae_kw": "MAE (kW)", "rmse_kw": "RMSE (kW)",
                "mape_pct": "MAPE (%)", "r2": "R²", "n_samples": "n",
            }),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not compute stratified metrics: {e}")

# ---------------------------------------------------------------------------
# Section 5: Feature importance + explain panel
# ---------------------------------------------------------------------------

st.subheader("Feature Importance & Explain")

fi_col, exp_col = st.columns(2)

with fi_col:
    st.markdown("**XGBoost feature importance**")
    try:
        xgb_model = models.get("xgboost")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            from gridguard.features.engineer import FEATURE_COLS

            fi_df = pd.DataFrame(
                {"feature": FEATURE_COLS, "importance": xgb_model.feature_importances_}
            ).sort_values("importance", ascending=True)
            fig4 = px.bar(
                fi_df.tail(10), x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Blues", height=320,
            )
            fig4.update_layout(margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")

with exp_col:
    st.markdown("**Explain a specific interval**")
    anom_timestamps = anomaly_df[anomaly_df["is_anomaly"]]["timestamp"].dt.to_pydatetime().tolist()
    if not anom_timestamps:
        st.info("No anomalies to explain at the current threshold.")
    else:
        selected_ts = st.selectbox(
            "Select anomalous timestamp",
            options=anom_timestamps[:50],
            format_func=lambda t: t.strftime("%Y-%m-%d %H:%M"),
        )
        if selected_ts:
            try:
                from gridguard.features.engineer import FEATURE_COLS, build_features

                row_df = anomaly_df[anomaly_df["timestamp"] == pd.Timestamp(selected_ts)].head(1).copy()
                row_df = build_features(row_df)
                present = [c for c in FEATURE_COLS if c in row_df.columns]
                X_row = row_df[present]

                try:
                    from gridguard.explainability.shap_explain import explain_anomaly

                    shap_df = explain_anomaly(best_model, X_row, top_n=8)
                    shap_df["color"] = shap_df["shap_value"].apply(
                        lambda v: ANOMALY_COLOR if v < 0 else ACTUAL_COLOR
                    )
                    fig_shap = px.bar(
                        shap_df.sort_values("shap_value"),
                        x="shap_value", y="feature", orientation="h",
                        color="shap_value",
                        color_continuous_scale=["#E74C3C", "#ECF0F1", "#2ECC71"],
                        color_continuous_midpoint=0,
                        height=320,
                        labels={"shap_value": "SHAP value (kW impact)"},
                    )
                    fig_shap.update_layout(margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception:
                    # Fallback: show raw feature values for the anomalous interval
                    st.dataframe(
                        X_row.T.rename(columns={X_row.index[0]: "value"}).round(3),
                        use_container_width=True,
                    )
                    st.caption("Install `shap` for SHAP waterfall explanations.")
            except Exception as e:
                st.warning(f"Explanation failed: {e}")

# ---------------------------------------------------------------------------
# Section 6: Weather drivers
# ---------------------------------------------------------------------------

st.subheader("Weather Drivers")

fig3 = px.scatter(
    anomaly_df.sample(min(2000, len(anomaly_df)), random_state=42),
    x="irradiance_wm2", y="ac_power_kw",
    color="is_anomaly",
    color_discrete_map={True: ANOMALY_COLOR, False: ACTUAL_COLOR},
    labels={"irradiance_wm2": "Irradiance (W/m²)", "ac_power_kw": "AC Power (kW)"},
    height=320, opacity=0.5,
)
fig3.update_layout(margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 7: Next-day solar forecast (clear-sky approximation)
# ---------------------------------------------------------------------------

st.subheader("Next-Day Solar Forecast")
st.caption(
    "Clear-sky forecast using sun geometry for the site latitude. "
    "This is a physics-based upper bound, not a weather-aware prediction. "
    "Actual output will vary with cloud cover."
)

try:
    # Derive site latitude: use GMU Fairfax default if no site selected or registry unavailable
    _site_lat = 38.8316  # GMU Fairfax default
    _site_cap = 250.0
    if selected_site:
        try:
            from gridguard.sites.registry import get_site as _get_site
            _s = _get_site(selected_site)
            _site_lat = _s.latitude
            _site_cap = _s.capacity_kw
        except Exception:
            pass

    # "Tomorrow" = day after the last timestamp in the test set
    _last_ts = test_df["timestamp"].max()
    _next_day = (_last_ts + pd.Timedelta(days=1)).normalize()
    _forecast_idx = pd.date_range(_next_day, periods=96, freq="15min")

    # Clear-sky irradiance using the same sun-geometry model as the synthetic generator
    _doy = _forecast_idx.day_of_year.values
    _hour = _forecast_idx.hour.values + _forecast_idx.minute.values / 60
    _lat_rad = np.radians(_site_lat)
    _decl = np.radians(23.45 * np.sin(np.radians(360 / 365 * (_doy - 81))))
    _ha = np.radians(15 * (_hour - 12))
    _cos_z = np.clip(
        np.sin(_lat_rad) * np.sin(_decl) + np.cos(_lat_rad) * np.cos(_decl) * np.cos(_ha),
        0, 1,
    )
    _clearsky_irr = np.clip(1000 * _cos_z, 0, 1200)

    # Seasonal temperature for this day of year
    _temp = 16 + 11 * np.sin(np.radians(360 / 365 * (_doy - 80))) + 5 * np.sin(np.radians(15 * (_hour - 14)))

    # Apply model to predict from clean weather inputs
    _forecast_df = pd.DataFrame({
        "timestamp": _forecast_idx,
        "irradiance_wm2": _clearsky_irr.round(2),
        "temperature_c": _temp.round(2),
        "wind_speed_ms": np.full(96, 3.5),
        "ac_power_kw": np.zeros(96),
    })
    from gridguard.features.engineer import build_features as _build_features
    _forecast_df = _build_features(_forecast_df, include_lags=False)
    _present = [c for c in FEATURE_COLS if c in _forecast_df.columns]
    _preds = np.clip(best_model.predict(_forecast_df[_present]), 0, None)
    _forecast_df["forecast_kw"] = _preds

    _total_kwh = float((_forecast_df["forecast_kw"] * 0.25).sum())
    _peak_kw = float(_forecast_df["forecast_kw"].max())

    _fc1, _fc2 = st.columns(2)
    _fc1.metric("Forecasted generation", f"{_total_kwh:.1f} kWh")
    _fc2.metric("Peak forecast", f"{_peak_kw:.1f} kW")

    _fig_fc = go.Figure()
    _fig_fc.add_trace(go.Scatter(
        x=_forecast_df["timestamp"], y=_forecast_df["forecast_kw"],
        name="Forecast (clear-sky)", line=dict(color=PRED_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
    ))
    _fig_fc.update_layout(
        xaxis_title="Time", yaxis_title="Forecast AC Power (kW)",
        height=300, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(_fig_fc, use_container_width=True)

except Exception as _e:
    st.warning(f"Could not generate next-day forecast: {_e}")

st.divider()

st.caption(
    "GridGuard DMV v0.1.0 · DC / Northern Virginia Solar Asset Intelligence · "
    "Synthetic data unless real telemetry is configured · Not for production use without validation."
)
