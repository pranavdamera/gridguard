"""Centralised settings, loaded from the environment or a .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Data ---------------------------------------------------------------
    #: Which fleet the API serves by default: "real" (measured PVDAQ telemetry)
    #: or "synthetic" (the simulated DMV demonstration fleet). Both are always
    #: available per-site; this only sets the default view.
    data_mode: str = "synthetic"

    #: Curated real-data window. Kept in settings so `make data-real` and the
    #: artifact manifest agree on exactly which slice is canonical.
    real_data_start: str = "2016-01-01"
    real_data_end: str = "2016-12-31"

    #: NSRDB is optional and unused by the shipped pipeline; see data/nsrdb.py.
    nrel_api_key: str = ""
    nrel_api_email: str = ""

    # --- Paths (relative to repo root, resolved at runtime) -----------------
    model_dir: Path = Path("artifacts/models")
    report_dir: Path = Path("artifacts/reports")

    #: Working cache and derived frames. Regenerable, and not version-controlled.
    data_processed_dir: Path = Path("data/processed")

    #: Curated measured datasets that ship with the repository, so a fresh clone
    #: can run the real-data pipeline without downloading anything. Small,
    #: reviewed, and version-controlled — see docs/data_sources.md.
    data_curated_dir: Path = Path("data/curated")

    # --- Anomaly detection --------------------------------------------------
    #: Miscoverage level for the conformal lower bound. 0.05 targets a 95%
    #: one-sided prediction interval on healthy intervals.
    conformal_alpha: float = 0.05

    #: Legacy sigma detector, retained as a benchmark alongside conformal.
    anomaly_threshold_sigma: float = 2.0

    #: Detector used by the API and pipeline: "conformal" or "sigma".
    anomaly_method: str = "conformal"

    # --- Modelling ----------------------------------------------------------
    #: Both fleets share a split date, since they now share a timeline.
    train_test_split_date: str = "2016-10-01"
    real_train_test_split_date: str = "2016-10-01"

    # --- Spatial ------------------------------------------------------------
    #: Radius within which sites are treated as weather neighbours.
    neighbor_radius_km: float = 60.0

    # --- API ----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    #: Comma-separated list of allowed browser origins. Never "*" in production;
    #: see docs/deployment.md.
    cors_allowed_origins: str = "http://localhost:3000"

    #: Allow Vercel preview deployments (https://<branch>-<project>.vercel.app)
    #: to call the API. Off by default because it widens the origin set.
    cors_allow_vercel_previews: bool = False

    @property
    def allowed_origin_list(self) -> list[str]:
        """CORS origins as a cleaned list."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


settings = Settings()
