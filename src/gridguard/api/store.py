"""
Artifact store: everything the API serves, loaded once at startup.

The backend never trains. It loads what ``scripts/build_artifacts.py``
produced — models, calibrations, detected frames, grouped events, dataset
provenance and the build manifest — and holds them in memory for the process
lifetime.

Loading is deliberately tolerant. A site whose artifacts are missing is skipped
with a warning rather than taking the whole service down, so a partially-built
deployment still serves what it has and ``/health`` reports honestly what is
missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from gridguard.anomaly.conformal import ConformalCalibration
from gridguard.artifacts.manifest import MANIFEST_FILENAME, ArtifactManifest, load_manifest
from gridguard.config import settings
from gridguard.data.provenance import DatasetProvenance, load_provenance
from gridguard.sites.registry import Site, list_sites

logger = logging.getLogger(__name__)


@dataclass
class SiteBundle:
    """Everything loaded for one site."""

    site: Site
    detected: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    provenance: DatasetProvenance | None = None
    anomaly_model: object | None = None
    forecast_models: dict[str, object] = field(default_factory=dict)
    calibration: ConformalCalibration | None = None

    @property
    def has_data(self) -> bool:
        return self.detected is not None and not self.detected.empty


class ArtifactStore:
    """Loads and holds the artifacts backing the API."""

    def __init__(
        self,
        model_dir: Path | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or settings.model_dir)
        self.data_dir = Path(data_dir or settings.data_processed_dir)
        self.manifest: ArtifactManifest | None = None
        self.bundles: dict[str, SiteBundle] = {}
        self.load_errors: list[str] = []

    # -- loading ------------------------------------------------------------

    def load(self) -> ArtifactStore:
        """Load the manifest and every site bundle it references."""
        self.manifest = load_manifest(self.model_dir / MANIFEST_FILENAME)
        if self.manifest is None:
            self.load_errors.append(
                f"No artifact manifest at {self.model_dir / MANIFEST_FILENAME}. "
                "Run `make build-artifacts`."
            )
            logger.warning(self.load_errors[-1])

        for site in list_sites():
            bundle = self._load_site(site)
            if bundle.has_data or bundle.anomaly_model is not None:
                self.bundles[site.site_id] = bundle

        logger.info(
            "Artifact store ready: %d site(s) loaded%s",
            len(self.bundles),
            f", {len(self.load_errors)} warning(s)" if self.load_errors else "",
        )
        return self

    def _load_site(self, site: Site) -> SiteBundle:
        bundle = SiteBundle(site=site)
        site_dir = self.model_dir / site.site_id

        detected_path = self.data_dir / f"detected_{site.site_id}.parquet"
        if detected_path.exists():
            bundle.detected = pd.read_parquet(detected_path)

        events_path = self.data_dir / f"events_{site.site_id}.parquet"
        if events_path.exists():
            bundle.events = pd.read_parquet(events_path)

        for suffix in ("", "_demo"):
            provenance_path = self.data_dir / f"telemetry_{site.site_id}{suffix}.provenance.json"
            records = load_provenance(provenance_path)
            if records:
                bundle.provenance = records[0]
                break

        bundle.anomaly_model = self._load_pickle(site_dir / "anomaly_detector.pkl")
        bundle.calibration = ConformalCalibration.load(site_dir / "conformal_calibration.json")

        # Forecast-model pickles are deliberately *not* loaded. Nothing the API
        # serves needs them: /metrics reads the manifest, and /forecast uses the
        # weather-only model. Loading a tree ensemble per site would add
        # hundreds of megabytes of resident memory for no served response.

        return bundle

    def _load_pickle(self, path: Path) -> object | None:
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:
            message = f"Could not load {path.name} for {path.parent.name}: {exc}"
            self.load_errors.append(message)
            logger.warning(message)
            return None

    # -- access -------------------------------------------------------------

    @property
    def site_ids(self) -> list[str]:
        return sorted(self.bundles)

    def bundle(self, site_id: str) -> SiteBundle | None:
        return self.bundles.get(site_id)

    def sites(self, data_mode: str | None = None) -> list[Site]:
        sites = [b.site for b in self.bundles.values()]
        if data_mode:
            sites = [s for s in sites if s.data_mode == data_mode]
        return sorted(sites, key=lambda s: s.site_id)

    def default_site_id(self, data_mode: str | None = None) -> str | None:
        """The site the UI lands on when none is specified.

        Prefers the demo site when the synthetic fleet is available, since the
        guided walkthrough starts there.
        """
        from gridguard.data.synthetic import DEMO_SITE_ID

        candidates = [s.site_id for s in self.sites(data_mode)]
        if not candidates:
            return None
        if DEMO_SITE_ID in candidates:
            return DEMO_SITE_ID
        return candidates[0]

    def all_events(self) -> pd.DataFrame:
        """Every site's events in one frame, most severe and most costly first."""
        frames = [
            b.events for b in self.bundles.values() if b.events is not None and not b.events.empty
        ]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)

        severity_rank = {"high": 0, "medium": 1, "low": 2}
        combined["_rank"] = (
            combined.get("severity", pd.Series(dtype=str)).map(severity_rank).fillna(3)
        )
        combined = combined.sort_values(["_rank", "total_lost_kwh"], ascending=[True, False]).drop(
            columns="_rank"
        )

        # Event ids are only unique within a site, so give the fleet view a
        # stable global handle the frontend can route on.
        combined["global_event_id"] = (
            combined["site_id"].astype(str) + ":" + combined["event_id"].astype(str)
        )
        return combined.reset_index(drop=True)

    def provenance_records(self) -> list[DatasetProvenance]:
        return [b.provenance for b in self.bundles.values() if b.provenance is not None]

    def health(self) -> dict:
        """Non-sensitive health summary for /health."""
        total_rows = sum(len(b.detected) for b in self.bundles.values() if b.detected is not None)
        return {
            "sites_loaded": len(self.bundles),
            "sites_with_models": sum(
                1 for b in self.bundles.values() if b.anomaly_model is not None
            ),
            "sites_with_calibration": sum(
                1 for b in self.bundles.values() if b.calibration is not None
            ),
            "data_rows": total_rows,
            "manifest_present": self.manifest is not None,
            "artifact_built_at": self.manifest.built_at if self.manifest else None,
            "artifact_commit": self.manifest.git_commit if self.manifest else None,
            "warnings": self.load_errors[:10],
        }
