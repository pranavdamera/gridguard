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
from gridguard.features.engineer import FEATURE_COLS, get_X_y
from gridguard.models.evaluate import compare_models, stratified_metrics
from gridguard.models.train import load_model
from gridguard.sites.registry import load_sites

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GridGuard DMV — Solar Asset Intelligence",
    page_icon="GG",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_COLOR = "#F4B740"
INK = "#10201A"
MUTED = "#66756E"
SURFACE = "#FFFFFF"
CANVAS = "#F6F4EE"
LINE = "#D9DDD5"
ANOMALY_COLOR = "#D1493F"
PRED_COLOR = "#2F6F8F"
ACTUAL_COLOR = "#2E8B57"
SEVERITY_COLOR = {"high": ANOMALY_COLOR, "medium": "#C47B24", "low": ACTUAL_COLOR}

PLOT_TEMPLATE = "plotly_white"


def inject_design_system() -> None:
    """Apply lightweight visual polish without changing dashboard behavior."""
    st.markdown(
        f"""
        <style>
            :root {{
                --gg-ink: {INK};
                --gg-muted: {MUTED};
                --gg-line: {LINE};
                --gg-surface: {SURFACE};
                --gg-canvas: {CANVAS};
                --gg-brand: {BRAND_COLOR};
                --gg-good: {ACTUAL_COLOR};
                --gg-info: {PRED_COLOR};
                --gg-bad: {ANOMALY_COLOR};
            }}

            .stApp {{
                background:
                    linear-gradient(180deg, rgba(246,244,238,0.98), rgba(250,249,245,1) 340px);
                color: var(--gg-ink);
            }}

            [data-testid="stSidebar"] {{
                background: #101D18;
                border-right: 1px solid rgba(255,255,255,0.08);
            }}

            [data-testid="stSidebar"] * {{
                color: rgba(255,255,255,0.88);
            }}

            [data-testid="stSidebar"] .stSlider p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stCaptionContainer {{
                color: rgba(255,255,255,0.72) !important;
            }}

            [data-testid="stSidebar"] [data-baseweb="select"] > div {{
                background: rgba(255,255,255,0.08);
                border-color: rgba(255,255,255,0.16);
            }}

            [data-testid="stSidebar"] code {{
                color: #F7E7B7 !important;
                background: rgba(255,255,255,0.08) !important;
            }}

            [data-testid="stHeader"] {{
                background: rgba(246,244,238,0.78);
                backdrop-filter: blur(10px);
            }}

            .block-container {{
                padding-top: 2.1rem;
                padding-bottom: 2.8rem;
                max-width: 1280px;
            }}

            h1, h2, h3 {{
                letter-spacing: 0;
                color: var(--gg-ink);
            }}

            h3 {{
                margin-top: 0.35rem;
                font-size: 1.1rem;
            }}

            div[data-testid="stMarkdownContainer"] p {{
                color: var(--gg-muted);
            }}

            .gg-sidebar-brand {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin: 0.35rem 0 1.35rem;
            }}

            .gg-logo {{
                width: 42px;
                height: 42px;
                border-radius: 8px;
                display: grid;
                place-items: center;
                background: linear-gradient(135deg, #F8C85B, #2E8B57);
                color: #0C1814;
                font-weight: 800;
                letter-spacing: 0;
                box-shadow: 0 12px 24px rgba(0,0,0,0.22);
            }}

            .gg-sidebar-brand strong {{
                display: block;
                font-size: 1.02rem;
                line-height: 1.1;
            }}

            .gg-sidebar-brand span {{
                display: block;
                margin-top: 0.18rem;
                color: rgba(255,255,255,0.62);
                font-size: 0.78rem;
            }}

            .gg-hero {{
                padding: 1.45rem 1.55rem;
                border: 1px solid rgba(16,32,26,0.08);
                border-radius: 8px;
                background:
                    linear-gradient(135deg, rgba(16,32,26,0.96), rgba(33,62,52,0.94)),
                    linear-gradient(90deg, rgba(244,183,64,0.22), transparent);
                box-shadow: 0 18px 48px rgba(30,43,34,0.10);
                margin-bottom: 1rem;
            }}

            .gg-hero-topline {{
                color: rgba(255,255,255,0.68);
                text-transform: uppercase;
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }}

            .gg-hero h1 {{
                color: white;
                font-size: clamp(2rem, 4vw, 3.45rem);
                line-height: 1;
                margin: 0;
            }}

            .gg-hero p {{
                max-width: 760px;
                margin: 0.8rem 0 0;
                color: rgba(255,255,255,0.74) !important;
                font-size: 1rem;
            }}

            .gg-hero-footer {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1.2rem;
            }}

            .gg-pill {{
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 999px;
                padding: 0.36rem 0.64rem;
                color: rgba(255,255,255,0.82);
                background: rgba(255,255,255,0.07);
                font-size: 0.78rem;
            }}

            .gg-metric-grid {{
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 0.7rem;
                margin: 0.65rem 0 1.15rem;
            }}

            .gg-metric {{
                min-height: 116px;
                padding: 0.88rem;
                background: var(--gg-surface);
                border: 1px solid var(--gg-line);
                border-radius: 8px;
                box-shadow: 0 10px 28px rgba(45,55,48,0.06);
            }}

            .gg-metric .label {{
                color: var(--gg-muted);
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}

            .gg-metric .value {{
                color: var(--gg-ink);
                font-size: 1.7rem;
                line-height: 1.1;
                font-weight: 760;
                margin-top: 0.55rem;
                white-space: nowrap;
            }}

            .gg-metric .hint {{
                color: var(--gg-muted);
                font-size: 0.78rem;
                margin-top: 0.35rem;
            }}

            .gg-section-label {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin: 1.05rem 0 0.35rem;
                color: var(--gg-muted);
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 800;
            }}

            .gg-panel {{
                background: var(--gg-surface);
                border: 1px solid var(--gg-line);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 10px 28px rgba(45,55,48,0.06);
                margin-bottom: 0.85rem;
            }}

            .gg-alert {{
                border-left: 4px solid var(--gg-bad);
                background: #FFF7F4;
                border-radius: 8px;
                padding: 0.9rem 1rem;
                color: var(--gg-ink);
                margin: 0.4rem 0 0.9rem;
            }}

            .gg-alert strong {{
                color: var(--gg-bad);
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--gg-line);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 8px 22px rgba(45,55,48,0.05);
            }}

            hr {{
                margin: 1.4rem 0;
                border-color: rgba(16,32,26,0.08);
            }}

            @media (max-width: 1100px) {{
                .gg-metric-grid {{
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                }}
            }}

            @media (max-width: 720px) {{
                .block-container {{
                    padding-left: 1rem;
                    padding-right: 1rem;
                }}
                .gg-metric-grid {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
                .gg-hero {{
                    padding: 1.1rem;
                }}
                .gg-metric .value {{
                    font-size: 1.35rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="gg-section-label">{text}</div>', unsafe_allow_html=True)


def open_panel() -> None:
    st.markdown('<div class="gg-panel">', unsafe_allow_html=True)


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def format_metric_value(value: float | int | str, unit: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        formatted = f"{value:,}"
    elif abs(value) >= 100:
        formatted = f"{value:,.0f}"
    else:
        formatted = f"{value:,.1f}"
    return f"{formatted}{unit}"


def metric_card(label: str, value: str, hint: str) -> str:
    return f"""
        <div class="gg-metric">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
    """


def apply_plot_style(fig: go.Figure, *, height: int, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0) if legend else None,
    )
    fig.update_xaxes(gridcolor="rgba(16,32,26,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(16,32,26,0.08)", zeroline=False)
    return fig


inject_design_system()


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
    st.markdown(
        """
        <div class="gg-sidebar-brand">
            <div class="gg-logo">GG</div>
            <div>
                <strong>GridGuard</strong>
                <span>Solar asset intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown("**Telemetry mode**")
    st.code(settings.data_source, language=None)
    st.markdown("**Links**")
    st.markdown("- [API docs](http://localhost:8000/docs)")
    st.markdown("- [GitHub](https://github.com/pranav-damera/gridguard)")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    """
    <section class="gg-hero">
        <div class="gg-hero-topline">DMV campus solar monitoring</div>
        <h1>GridGuard</h1>
        <p>
            Forecast expected generation, surface underperformance events, and estimate
            lost energy across DC and Northern Virginia solar assets.
        </p>
        <div class="gg-hero-footer">
            <span class="gg-pill">15-minute intervals</span>
            <span class="gg-pill">Residual-calibrated alerts</span>
            <span class="gg-pill">Synthetic demo telemetry</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
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

st.markdown(
    f"""
    <div class="gg-metric-grid">
        {metric_card("Anomaly events", format_metric_value(n_events), "Grouped fault windows")}
        {metric_card("Flagged intervals", format_metric_value(n_anomalies), "15-minute detections")}
        {metric_card("Lost energy", f"{total_lost_kwh:,.1f} kWh", "Estimated recoverable output")}
        {metric_card("Fault days", format_metric_value(fault_days), "Days with any alert")}
        {metric_card("Generation", f"{total_actual_kwh / 1000:,.1f} MWh", "Observed test output")}
        {metric_card("Best RMSE", f"{best_rmse:.3f} kW", "Lowest model error")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1: Actual vs Predicted
# ---------------------------------------------------------------------------

section_label("Generation Forecast")
st.subheader("Actual vs Predicted Generation")

# Limit to 1 week for rendering performance
plot_df = anomaly_df.head(4 * 24 * 7)

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["ac_power_kw"],
        name="Actual",
        line=dict(color=ACTUAL_COLOR, width=1),
        opacity=0.8,
    )
)
fig1.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["predicted_kw"],
        name="Predicted",
        line=dict(color=PRED_COLOR, width=1.5, dash="dot"),
    )
)
anom_plot = plot_df[plot_df["is_anomaly"]]
fig1.add_trace(
    go.Scatter(
        x=anom_plot["timestamp"],
        y=anom_plot["ac_power_kw"],
        mode="markers",
        name="Anomaly",
        marker=dict(color=ANOMALY_COLOR, size=6, symbol="x"),
    )
)
fig1.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="AC Power (kW)",
)
apply_plot_style(fig1, height=390)
st.plotly_chart(fig1, width="stretch")

# ---------------------------------------------------------------------------
# Section 2: Daily energy loss
# ---------------------------------------------------------------------------

section_label("Loss Accounting")
st.subheader("Daily Energy Loss from Anomalies")

loss_plot = daily_df[daily_df["lost_energy_kwh"] > 0].tail(60)
if not loss_plot.empty:
    fig2 = px.bar(
        loss_plot,
        x="date",
        y="lost_energy_kwh",
        color="loss_pct",
        color_continuous_scale="Reds",
        labels={"lost_energy_kwh": "Lost Energy (kWh)", "loss_pct": "Loss %"},
        height=300,
    )
    apply_plot_style(fig2, height=310, legend=False)
    st.plotly_chart(fig2, width="stretch")
else:
    st.info("No energy loss detected at this threshold.")

# ---------------------------------------------------------------------------
# Section 3: Anomaly Events table
# ---------------------------------------------------------------------------

section_label("Operations Queue")
st.subheader("Anomaly Events")

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
    display_events = display_events.rename(
        columns={
            "event_id": "ID",
            "start_time": "Start",
            "end_time": "End",
            "duration_minutes": "Duration (min)",
            "interval_count": "Intervals",
            "total_lost_kwh": "Lost (kWh)",
            "max_residual_sigma": "Max σ",
            "mean_actual_kw": "Actual (kW)",
            "mean_predicted_kw": "Predicted (kW)",
            "severity": "Severity",
            "explanation": "Explanation",
        }
    )
    cols_to_show = [
        "ID",
        "Start",
        "End",
        "Duration (min)",
        "Lost (kWh)",
        "Max σ",
        "Severity",
        "Explanation",
    ]
    if "site_id" in display_events.columns:
        cols_to_show.insert(1, "site_id")
    # Only include columns that actually exist (explanation added in events.py)
    cols_to_show = [c for c in cols_to_show if c in display_events.columns]
    st.dataframe(display_events[cols_to_show], width="stretch")

    # Show plain-English explanation for the most recent high-severity event
    high_events = events_df[events_df["severity"] == "high"]
    if not high_events.empty and "explanation" in high_events.columns:
        top = high_events.iloc[0]
        st.markdown(
            f"""
            <div class="gg-alert">
                <strong>Most recent high-severity event</strong><br>
                {top['explanation']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Severity breakdown
    sev_counts = events_df["severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig_sev = px.bar(
        sev_counts,
        x="Severity",
        y="Count",
        color="Severity",
        color_discrete_map={"high": ANOMALY_COLOR, "medium": "#F39C12", "low": ACTUAL_COLOR},
        height=220,
    )
    apply_plot_style(fig_sev, height=230, legend=False)
    fig_sev.update_layout(showlegend=False)
    st.plotly_chart(fig_sev, width="stretch")

# ---------------------------------------------------------------------------
# Section 4: Model performance + stratified metrics
# ---------------------------------------------------------------------------

section_label("Model Quality")
st.subheader("Model Performance")

left, right = st.columns(2)

with left:
    st.markdown("**Overall metrics (test set)**")
    display_metrics = metrics_df.rename(
        columns={
            "model": "Model",
            "mae_kw": "MAE (kW)",
            "rmse_kw": "RMSE (kW)",
            "mape_pct": "MAPE (%)",
            "r2": "R²",
        }
    )
    st.dataframe(display_metrics.set_index("Model"), width="stretch")

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
            strat_df.rename(
                columns={
                    strata_choice: strata_choice.replace("_", " ").title(),
                    "mae_kw": "MAE (kW)",
                    "rmse_kw": "RMSE (kW)",
                    "mape_pct": "MAPE (%)",
                    "r2": "R²",
                    "n_samples": "n",
                }
            ),
            width="stretch",
        )
    except Exception as e:
        st.warning(f"Could not compute stratified metrics: {e}")

# ---------------------------------------------------------------------------
# Section 5: Feature importance + explain panel
# ---------------------------------------------------------------------------

section_label("Explainability")
st.subheader("Feature Importance & Alert Context")

fi_col, exp_col = st.columns(2)

with fi_col:
    st.markdown("**XGBoost feature importance**")
    try:
        xgb_model = models.get("xgboost")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            fi_df = pd.DataFrame(
                {"feature": FEATURE_COLS, "importance": xgb_model.feature_importances_}
            ).sort_values("importance", ascending=True)
            fig4 = px.bar(
                fi_df.tail(10),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                height=320,
            )
            apply_plot_style(fig4, height=320, legend=False)
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, width="stretch")
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
                from gridguard.features.engineer import build_features

                row_df = (
                    anomaly_df[anomaly_df["timestamp"] == pd.Timestamp(selected_ts)].head(1).copy()
                )
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
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="shap_value",
                        color_continuous_scale=["#E74C3C", "#ECF0F1", "#2ECC71"],
                        color_continuous_midpoint=0,
                        height=320,
                        labels={"shap_value": "SHAP value (kW impact)"},
                    )
                    apply_plot_style(fig_shap, height=320, legend=False)
                    fig_shap.update_layout(showlegend=False)
                    st.plotly_chart(fig_shap, width="stretch")
                except Exception:
                    # Fallback: show raw feature values for the anomalous interval
                    st.dataframe(
                        X_row.T.rename(columns={X_row.index[0]: "value"}).round(3),
                        width="stretch",
                    )
                    st.caption("Install `shap` for SHAP waterfall explanations.")
            except Exception as e:
                st.warning(f"Explanation failed: {e}")

# ---------------------------------------------------------------------------
# Section 6: Weather drivers
# ---------------------------------------------------------------------------

section_label("Weather Signal")
st.subheader("Weather Drivers")

fig3 = px.scatter(
    anomaly_df.sample(min(2000, len(anomaly_df)), random_state=42),
    x="irradiance_wm2",
    y="ac_power_kw",
    color="is_anomaly",
    color_discrete_map={True: ANOMALY_COLOR, False: ACTUAL_COLOR},
    labels={"irradiance_wm2": "Irradiance (W/m²)", "ac_power_kw": "AC Power (kW)"},
    height=320,
    opacity=0.5,
)
apply_plot_style(fig3, height=330, legend=True)
st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------------------
# Section 7: Next-day solar forecast (clear-sky approximation)
# ---------------------------------------------------------------------------

section_label("Forward View")
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
        0,
        1,
    )
    _clearsky_irr = np.clip(1000 * _cos_z, 0, 1200)

    # Seasonal temperature for this day of year
    _temp = (
        16
        + 11 * np.sin(np.radians(360 / 365 * (_doy - 80)))
        + 5 * np.sin(np.radians(15 * (_hour - 14)))
    )

    # Apply model to predict from clean weather inputs
    _forecast_df = pd.DataFrame(
        {
            "timestamp": _forecast_idx,
            "irradiance_wm2": _clearsky_irr.round(2),
            "temperature_c": _temp.round(2),
            "wind_speed_ms": np.full(96, 3.5),
            "ac_power_kw": np.zeros(96),
        }
    )
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
    _fig_fc.add_trace(
        go.Scatter(
            x=_forecast_df["timestamp"],
            y=_forecast_df["forecast_kw"],
            name="Forecast (clear-sky)",
            line=dict(color=PRED_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.12)",
        )
    )
    _fig_fc.update_layout(xaxis_title="Time", yaxis_title="Forecast AC Power (kW)")
    apply_plot_style(_fig_fc, height=310, legend=True)
    st.plotly_chart(_fig_fc, width="stretch")

except Exception as _e:
    st.warning(f"Could not generate next-day forecast: {_e}")

# ---------------------------------------------------------------------------
# Section 8: Fleet map
# ---------------------------------------------------------------------------

section_label("Fleet Context")
st.subheader("Fleet Map")
st.caption(
    "Site locations, capacities, and illustrative health status. "
    "Coordinates are approximate estimates from public records."
)

try:
    sites_dict = load_sites()

    # Build per-site health summary from anomaly data if available
    site_lost: dict[str, float] = {}
    if "site_id" in anomaly_df.columns:
        site_lost = anomaly_df.groupby("site_id")["lost_energy_kwh"].sum().to_dict()

    map_rows = []
    for s in sites_dict.values():
        lost = site_lost.get(s.site_id, 0.0)
        # Simple severity label for hover display
        if lost > 10:
            status = "High loss"
        elif lost > 1:
            status = "Medium loss"
        else:
            status = "Normal"
        map_rows.append(
            {
                "site_id": s.site_id,
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "capacity_kw": s.capacity_kw,
                "lost_kwh": round(lost, 2),
                "status": status,
            }
        )

    map_df = pd.DataFrame(map_rows)

    status_color = {"High loss": "red", "Medium loss": "orange", "Normal": "green"}

    fig_map = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        text="site_id",
        size="capacity_kw",
        color="status",
        color_discrete_map=status_color,
        hover_name="name",
        hover_data={
            "capacity_kw": True,
            "lost_kwh": True,
            "status": True,
            "latitude": False,
            "longitude": False,
        },
        scope="usa",
        title="DMV Solar Fleet (illustrative sites, synthetic data)",
        height=440,
    )
    fig_map.update_geos(
        center={"lat": 38.85, "lon": -77.2},
        projection_scale=12,
        showland=True,
        landcolor="rgb(240, 240, 240)",
        showcoastlines=True,
        coastlinecolor="rgb(180, 180, 180)",
        showstates=True,
        statecolor="rgb(200, 200, 200)",
    )
    apply_plot_style(fig_map, height=440, legend=True)
    fig_map.update_layout(legend_title_text="Status")
    st.plotly_chart(fig_map, width="stretch")

    # Summary table
    st.dataframe(
        map_df.rename(
            columns={
                "site_id": "Site ID",
                "name": "Name",
                "capacity_kw": "Capacity (kW)",
                "lost_kwh": "Est. Lost (kWh)",
                "status": "Status",
            }
        )[["Site ID", "Name", "Capacity (kW)", "Est. Lost (kWh)", "Status"]],
        width="stretch",
    )

except Exception as _map_err:
    st.warning(f"Fleet map unavailable: {_map_err}")

st.divider()

st.caption(
    "GridGuard DMV v0.1.0 · DC / Northern Virginia Solar Asset Intelligence · "
    "Synthetic data unless real telemetry is configured · Not for production use without validation."
)
