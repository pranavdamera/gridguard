#!/usr/bin/env python3
"""
Per-site diagnostic report.

Replaces the retired Streamlit dashboard. Where that was an interactive surface
that recomputed everything on every page load, this writes a fixed set of
figures and the CSVs behind them, so a number in the technical report can be
traced to a file rather than to a screenshot of a session nobody can reproduce.

    python experiments/run_diagnostics.py                 # every site
    python experiments/run_diagnostics.py --site nist_roof
    python experiments/run_diagnostics.py --no-figures    # CSVs only

Output goes to ``artifacts/reports/diagnostics/<site_id>/``. Requires artifacts
to have been built first (``make build-artifacts``) and matplotlib
(``pip install -e ".[experiments]"``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from gridguard.analysis import (
    clear_sky_projection,
    feature_importance,
    power_curve,
    power_curve_summary,
)
from gridguard.anomaly.detect import compute_daily_loss
from gridguard.artifacts.manifest import MANIFEST_FILENAME, load_manifest
from gridguard.config import settings
from gridguard.features.engineer import get_X_y
from gridguard.sites.registry import get_site, list_site_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-6s %(message)s")
logger = logging.getLogger("diagnostics")


def _load_site_inputs(site_id: str) -> tuple[pd.DataFrame, object]:
    """The detected frame and weather-only model an artifact build produced."""
    detected_path = Path(settings.data_processed_dir) / f"detected_{site_id}.parquet"
    model_path = Path(settings.model_dir) / site_id / "anomaly_detector.pkl"

    if not detected_path.exists():
        raise FileNotFoundError(
            f"{detected_path} not found. Run `make build-artifacts` before diagnostics."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `make build-artifacts` first.")

    return pd.read_parquet(detected_path), joblib.load(model_path)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _figure_power_curve(curve: pd.DataFrame, site_name: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    healthy = curve[~curve["is_anomaly"]]
    flagged = curve[curve["is_anomaly"]]

    ax.scatter(
        healthy["irradiance_wm2"],
        healthy["ac_power_kw"],
        s=4,
        alpha=0.25,
        color="#3D6390",
        label=f"Healthy (n={len(healthy):,})",
        linewidths=0,
    )
    ax.scatter(
        flagged["irradiance_wm2"],
        flagged["ac_power_kw"],
        s=7,
        alpha=0.7,
        color="#B03A2E",
        label=f"Flagged (n={len(flagged):,})",
        linewidths=0,
    )
    ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel("AC power (kW)")
    ax.set_title(f"{site_name} — power curve", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, markerscale=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _figure_daily_loss(daily: pd.DataFrame, site_name: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = daily[daily["lost_energy_kwh"] > 0].tail(60)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=140)
    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No days with detected loss",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#5A6260",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.bar(
            pd.to_datetime(plot_df["date"]),
            plot_df["lost_energy_kwh"],
            color="#A85428",
            width=0.8,
        )
        ax.set_ylabel("Lost energy (kWh)")
        fig.autofmt_xdate(rotation=45)
        _style(ax)
    ax.set_title(f"{site_name} — daily energy attributed to detected loss", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _figure_importance(importance: pd.DataFrame, site_name: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=140)
    ax.barh(top["feature"], top["importance"], color="#1F6F66")
    method = top["method"].iloc[0] if len(top) else "unknown"
    ax.set_xlabel(f"Importance ({method})")
    ax.set_title(f"{site_name} — feature importance", fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _figure_projection(projection: pd.DataFrame, site_name: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=140)
    ax.fill_between(
        projection["timestamp"], projection["projected_kw"], color="#1F6F66", alpha=0.18
    )
    ax.plot(projection["timestamp"], projection["projected_kw"], color="#1F6F66", linewidth=1.6)
    ax.set_ylabel("Projected AC power (kW)")
    ax.set_title(
        f"{site_name} — clear-sky ceiling (not a weather forecast)", fontsize=11, loc="left"
    )
    fig.autofmt_xdate(rotation=45)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_site(site_id: str, out_root: Path, *, figures: bool = True) -> dict:
    site = get_site(site_id)
    detected, model = _load_site_inputs(site_id)
    out_dir = out_root / site_id
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = power_curve(detected)
    summary = power_curve_summary(curve)
    daily = compute_daily_loss(detected)

    X, _ = get_X_y(detected, mode="weather_only")
    importance = feature_importance(model, X)

    start = pd.Timestamp(detected["timestamp"].max()).normalize() + pd.Timedelta(days=1)
    projection = clear_sky_projection(site, model, start=start)

    curve.to_csv(out_dir / "power_curve.csv", index=False)
    summary.to_csv(out_dir / "power_curve_summary.csv", index=False)
    daily.to_csv(out_dir / "daily_loss.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    projection.to_csv(out_dir / "clear_sky_projection.csv", index=False)

    if figures:
        _figure_power_curve(curve, site.name, out_dir / "power_curve.png")
        _figure_daily_loss(daily, site.name, out_dir / "daily_loss.png")
        _figure_importance(importance, site.name, out_dir / "feature_importance.png")
        _figure_projection(projection, site.name, out_dir / "clear_sky_projection.png")

    return {
        "site_id": site_id,
        "data_mode": site.data_mode,
        "intervals": len(detected),
        "daylight_intervals": len(curve),
        "flagged_intervals": int(curve["is_anomaly"].sum()),
        "total_lost_kwh": round(float(daily["lost_energy_kwh"].sum()), 1),
        "projected_clear_sky_kwh": round(float(projection["projected_kw"].sum() * 0.25), 1),
        "top_feature": importance["feature"].iloc[0] if len(importance) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site id; repeatable. Default: every site in the registry.",
    )
    parser.add_argument("--out", type=Path, default=Path(settings.report_dir) / "diagnostics")
    parser.add_argument(
        "--no-figures", action="store_true", help="Write CSVs only. Does not require matplotlib."
    )
    args = parser.parse_args()

    figures = not args.no_figures
    if figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            logger.error(
                'matplotlib is not installed. Run: pip install -e ".[experiments]" '
                "— or pass --no-figures to write CSVs only."
            )
            return 1
        matplotlib.use("Agg")

    manifest = load_manifest(Path(settings.model_dir) / MANIFEST_FILENAME)
    if manifest is None:
        logger.error("No artifact manifest found. Run `make build-artifacts` first.")
        return 1
    logger.info("Artifacts built at %s (commit %s)", manifest.built_at, manifest.git_commit[:8])

    site_ids = args.sites or list_site_ids()
    rows = []
    for site_id in site_ids:
        try:
            rows.append(run_site(site_id, args.out, figures=figures))
            logger.info("  %-18s ok", site_id)
        except FileNotFoundError as exc:
            logger.warning("  %-18s skipped: %s", site_id, exc)

    if not rows:
        logger.error("No sites produced diagnostics.")
        return 1

    index = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out / "index.csv", index=False)
    logger.info("Wrote diagnostics for %d site(s) -> %s", len(rows), args.out)
    print(index.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
