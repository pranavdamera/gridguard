"""
The canonical artifact build.

One command produces everything the deployed backend needs, deterministically:

    verify/download data -> preprocess -> train -> calibrate -> evaluate
    -> write artifacts -> write manifest

The backend never trains. It loads what this pipeline produced and reports the
manifest alongside every metric, so a number on the methodology page can always
be traced to the build that generated it.

Per site
--------
1. Load canonical telemetry from its data source (measured or simulated).
2. Split temporally. Never randomly — a random split on time series leaks the
   future into training and inflates every score.
3. Train lag-aware forecasting models for short-horizon prediction.
4. Train the weather-only expected-generation model on **healthy intervals
   only**, which is the model detection runs against.
5. Calibrate conformal bounds and the legacy sigma statistics on the same
   healthy intervals.
6. Evaluate forecasts on the held-out test split.
7. Run detection over the test split, group events, and — where ground-truth
   labels exist — score detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from gridguard.anomaly.detect import detect_anomalies, fit_calibrations
from gridguard.anomaly.evaluate import evaluate_detection
from gridguard.anomaly.events import group_anomaly_events
from gridguard.artifacts.manifest import SiteArtifact
from gridguard.config import settings
from gridguard.data.faults import build_evaluation_scenario
from gridguard.data.sources import load_site_data
from gridguard.features.engineer import get_X_y, train_test_split_by_date
from gridguard.models.baseline import ALL_MODELS, get_model
from gridguard.models.evaluate import compare_models, evaluate_model
from gridguard.models.physics import PhysicsBaseline, PhysicsResidualModel
from gridguard.sites.registry import Site, get_site

logger = logging.getLogger(__name__)

VAL_FRAC = 0.15  # chronological tail of training data, for early stopping


@dataclass
class SiteBuildResult:
    """In-memory result of building one site, plus its manifest entry."""

    site: Site
    artifact: SiteArtifact
    test_df: pd.DataFrame
    detected: pd.DataFrame
    events: pd.DataFrame


def site_artifact_dir(site_id: str, model_dir: Path | None = None) -> Path:
    """Per-site artifact directory, so sites never overwrite each other."""
    return Path(model_dir or settings.model_dir) / site_id


def build_site(
    site_id: str,
    *,
    model_dir: Path | None = None,
    data_dir: Path | None = None,
    model_names: list[str] | None = None,
    include_physics: bool = True,
    use_cache: bool = True,
    demo: bool = False,
    **load_kwargs,
) -> SiteBuildResult:
    """Build every artifact for one site and return its manifest entry."""
    site = get_site(site_id)
    out_dir = site_artifact_dir(site_id, model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Building %s (%s, %s)", site.site_id, site.name, site.data_mode)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if site.is_real:
        load_kwargs.setdefault("start", settings.real_data_start)
        load_kwargs.setdefault("end", settings.real_data_end)
    else:
        # The served fleet carries only the scripted demonstration event. The
        # full fault taxonomy is injected separately, into the test split, for
        # detector evaluation — mixing it into the served data would bury the
        # walkthrough under hundreds of simultaneous synthetic failures.
        load_kwargs.setdefault("inject_scenario", False)
    frame, provenance = load_site_data(
        site_id, cache_dir=data_dir, use_cache=use_cache, demo=demo, **load_kwargs
    )

    split_date = (
        settings.real_train_test_split_date if site.is_real else settings.train_test_split_date
    )
    train_df, test_df = train_test_split_by_date(frame, split_date)
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Temporal split at {split_date} left {len(train_df)} train / {len(test_df)} test "
            f"rows for '{site_id}'. Adjust the split date or widen the data window."
        )

    val_cutoff = int(len(train_df) * (1 - VAL_FRAC))
    fit_df = train_df.iloc[:val_cutoff].copy()
    val_df = train_df.iloc[val_cutoff:].copy()
    logger.info(
        "Split at %s -> fit %d | val %d | test %d rows",
        split_date,
        len(fit_df),
        len(val_df),
        len(test_df),
    )

    artifact = SiteArtifact(
        site_id=site.site_id,
        site_name=site.name,
        data_mode=site.data_mode,
        dataset=provenance.dataset,
        data_start=provenance.start,
        data_end=provenance.end,
        interval_minutes=provenance.interval_minutes or 15,
        row_count=len(frame),
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_test_split_date=split_date,
    )

    # ------------------------------------------------------------------
    # 2. Lag-aware forecasting models
    # ------------------------------------------------------------------
    names = model_names or ALL_MODELS
    X_fit, y_fit = get_X_y(fit_df, mode="lag_aware")
    X_val, y_val = get_X_y(val_df, mode="lag_aware")
    X_test, y_test = get_X_y(test_df, mode="lag_aware")

    trained: dict[str, object] = {}
    for name in names:
        logger.info("Training forecast model: %s", name)
        model = get_model(name)
        if name == "xgboost":
            # Early stopping validates on the chronological tail of training
            # data, never on the test split.
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_fit, y_fit)
        _save_pickle(model, out_dir / f"{name}.pkl")
        trained[name] = model

    artifact.forecast_models = list(trained)
    metrics_df = compare_models(trained, X_test, y_test)
    artifact.forecast_metrics = metrics_df.to_dict("records")
    artifact.best_forecast_model = str(metrics_df.iloc[0]["model"])

    # ------------------------------------------------------------------
    # 3. Weather-only anomaly model, trained on healthy intervals only
    # ------------------------------------------------------------------
    healthy_fit_df = _healthy_only(fit_df)
    logger.info(
        "Training weather-only anomaly model on %d healthy intervals (of %d).",
        len(healthy_fit_df),
        len(fit_df),
    )

    X_fit_w, y_fit_w = get_X_y(healthy_fit_df, mode="weather_only")
    X_val_w, y_val_w = get_X_y(_healthy_only(val_df), mode="weather_only")

    anomaly_model = get_model("xgboost")
    anomaly_model.fit(X_fit_w, y_fit_w, eval_set=[(X_val_w, y_val_w)], verbose=False)
    _save_pickle(anomaly_model, out_dir / "anomaly_detector.pkl")

    # ------------------------------------------------------------------
    # 4. Physics baseline and hybrid
    # ------------------------------------------------------------------
    if include_physics and site.has_mount_geometry:
        physics = PhysicsBaseline(site).fit(X_fit_w, y_fit_w)
        _save_pickle(physics, out_dir / "physics.pkl")

        hybrid = PhysicsResidualModel(site).fit(X_fit_w, y_fit_w)
        _save_pickle(hybrid, out_dir / "physics_hybrid.pkl")

        X_test_w, y_test_w = get_X_y(test_df, mode="weather_only")
        artifact.physics_metrics = {
            "physics": evaluate_model(physics, X_test_w, y_test_w),
            "physics_hybrid": evaluate_model(hybrid, X_test_w, y_test_w),
            "weather_only_xgboost": evaluate_model(anomaly_model, X_test_w, y_test_w),
            "assumptions": physics.explain(),
            "evaluation_note": (
                "Weather-only comparison: these models see no lagged power, so their scores "
                "are not comparable with the lag-aware forecast models above."
                + (
                    " On synthetic data the physics baseline inverts the same clear-sky model "
                    "that generated the series, so its accuracy here is near-tautological and "
                    "should not be read as evidence."
                    if not site.is_real
                    else ""
                )
            ),
        }
    elif include_physics:
        artifact.notes.append(
            "Physics baseline skipped: no tilt/azimuth in the registry for this site."
        )

    # ------------------------------------------------------------------
    # 5. Calibration
    # ------------------------------------------------------------------
    # Split conformal requires calibration residuals from data the model did
    # *not* train on. Calibrating on the fit split would use in-sample
    # residuals, which are biased small, producing bounds that are too tight and
    # coverage below the stated guarantee.
    #
    # The validation split is used instead: held out from model fitting, and
    # chronologically adjacent to the test period, which keeps calibration and
    # evaluation residuals as close to exchangeable as a temporal split allows.
    # It is divided again so that coverage is reported on intervals the
    # calibration itself never saw.
    healthy_val = _healthy_only(val_df)
    if len(healthy_val) < 200:
        raise ValueError(
            f"Only {len(healthy_val)} healthy validation intervals for '{site_id}' — "
            "too few to calibrate conformal bounds. Widen the data window."
        )

    cal_cutoff = int(len(healthy_val) * 0.6)
    calibration_df = healthy_val.iloc[:cal_cutoff].copy()
    coverage_df = healthy_val.iloc[cal_cutoff:].copy()

    calibration, _ = fit_calibrations(
        calibration_df, anomaly_model, alpha=settings.conformal_alpha, model_dir=out_dir
    )
    artifact.conformal_alpha = calibration.alpha
    artifact.conformal_calibration_rows = calibration.n_calibration
    artifact.detection_method = settings.anomaly_method

    # Coverage on held-out healthy intervals: the check that the guarantee
    # survived contact with data neither the model nor the calibration saw.
    if not coverage_df.empty:
        val_detected = detect_anomalies(
            coverage_df, anomaly_model, calibration=calibration, model_dir=out_dir
        )
        daylight = val_detected["irradiance_wm2"] > 50
        artifact.conformal_coverage = calibration.coverage_on(
            val_detected.loc[daylight, "residual_kw"].to_numpy(),
            val_detected.loc[daylight, "irradiance_wm2"].to_numpy(),
        )

    # ------------------------------------------------------------------
    # 6. Detection over the test split
    # ------------------------------------------------------------------
    detected = detect_anomalies(test_df, anomaly_model, calibration=calibration, model_dir=out_dir)
    events = group_anomaly_events(detected)
    if not events.empty:
        events["site_id"] = site.site_id

    # ------------------------------------------------------------------
    # 7. Detection scoring on an injected-fault copy of the test split
    # ------------------------------------------------------------------
    artifact.detection_metrics = _score_detection(
        site, test_df, anomaly_model, calibration, out_dir
    )

    # ------------------------------------------------------------------
    # 8. Persist frames the API serves
    # ------------------------------------------------------------------
    processed = Path(data_dir or settings.data_processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    detected.to_parquet(processed / f"detected_{site.site_id}.parquet", index=False)
    if not events.empty:
        events.to_parquet(processed / f"events_{site.site_id}.parquet", index=False)

    artifact.files = sorted(p.name for p in out_dir.glob("*") if p.is_file())
    return SiteBuildResult(
        site=site, artifact=artifact, test_df=test_df, detected=detected, events=events
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_detection(
    site: Site,
    test_df: pd.DataFrame,
    model,
    calibration,
    out_dir: Path,
) -> dict:
    """Score the detector against faults injected into the held-out test split.

    This is how GridGuard gets detection metrics for **measured** sites at all.
    Real PV telemetry carries no annotated failures, so instead of pretending it
    does, known faults are injected on top of the real baseline and the detector
    is scored against those labels. The generation, weather and noise are
    genuine; only the failures are simulated, and every reported figure is
    labelled accordingly.

    Faults are injected into the *test* split only, after training and
    calibration are complete, so the detector has never seen them.
    """
    from gridguard.data.faults import apply_faults

    baseline = test_df.copy()
    for column in ("is_injected_fault", "injected_fault_type", "is_generation_loss"):
        baseline = baseline.drop(columns=column, errors="ignore")
    baseline = baseline.drop(columns="ac_power_baseline_kw", errors="ignore")

    scenario = build_evaluation_scenario(baseline["timestamp"])
    if not scenario:
        return {
            "note": (
                "Test split is too short to host the full fault taxonomy; detection "
                "metrics were not computed."
            )
        }

    faulted = apply_faults(baseline, scenario, capacity_kw=site.capacity_kw)
    detected = detect_anomalies(faulted, model, calibration=calibration, model_dir=out_dir)
    metrics = evaluate_detection(detected, events=scenario)

    payload = metrics.to_dict()
    payload["evaluation_design"] = (
        f"{len(scenario)} labelled faults spanning the full taxonomy were injected into the "
        f"held-out test split of {'measured' if site.is_real else 'simulated'} telemetry, "
        "after training and calibration. The detector had never seen them. Generation and "
        "weather are "
        + (
            "genuine measured data; only the failures are simulated."
            if site.is_real
            else "simulated."
        )
    )
    return payload


def _healthy_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop injected-fault intervals so baselines learn healthy behaviour.

    Without this the expected-generation model absorbs the faults present in its
    own training data and quietly learns that degraded output is normal.
    """
    if "is_injected_fault" not in df.columns:
        return df
    return df[~df["is_injected_fault"].fillna(False)].copy()


def _save_pickle(obj: object, path: Path) -> None:
    """Persist a model, compressed.

    Tree ensembles pickle to tens of megabytes uncompressed, which makes the
    artifact set impractical to move between a build step and a deployment.
    joblib's compression typically reduces them by an order of magnitude at
    negligible load-time cost.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)
    logger.debug("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
