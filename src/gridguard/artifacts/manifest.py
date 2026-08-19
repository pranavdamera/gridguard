"""
Artifact manifest: what was built, from what, and how well it scored.

Why this exists
---------------
A deployed backend must never retrain on startup — training is slow,
non-deterministic across library versions, and would make every restart a
silent experiment. So the backend loads prebuilt artifacts, and this manifest is
what makes those artifacts trustworthy: it records the exact commit, dataset
window, feature mode, hyperparameters, calibration configuration and evaluation
scores behind every file in the artifact directory.

If a number appears on the methodology page, it came from here. Nothing in the
web application reports a metric that is not backed by a manifest entry, which
is what stops hand-written figures drifting away from what the models actually
achieved.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"

#: Bumped when the artifact layout changes in a way the backend must notice.
ARTIFACT_SCHEMA_VERSION = "1.0"


def git_commit(short: bool = False) -> str:
    """Current git commit, or "unknown" outside a repository."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return False


@dataclass
class SiteArtifact:
    """Everything built for one site."""

    site_id: str
    site_name: str
    data_mode: str

    # dataset
    dataset: str = ""
    data_start: str = ""
    data_end: str = ""
    interval_minutes: int = 15
    row_count: int = 0
    train_rows: int = 0
    test_rows: int = 0
    train_test_split_date: str = ""

    # models
    forecast_models: list[str] = field(default_factory=list)
    best_forecast_model: str = ""
    anomaly_model: str = "xgboost_weather_only"
    anomaly_feature_mode: str = "weather_only"

    # calibration
    conformal_alpha: float | None = None
    conformal_calibration_rows: int = 0
    conformal_coverage: dict = field(default_factory=dict)
    detection_method: str = "conformal"

    # evaluation
    forecast_metrics: list[dict] = field(default_factory=list)
    physics_metrics: dict = field(default_factory=dict)
    detection_metrics: dict = field(default_factory=dict)

    # files
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArtifactManifest:
    """Top-level manifest for one artifact build."""

    artifact_version: str = ARTIFACT_SCHEMA_VERSION
    built_at: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    python_version: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)
    sites: list[SiteArtifact] = field(default_factory=list)
    build_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.built_at:
            self.built_at = datetime.now(UTC).isoformat(timespec="seconds")
        if not self.git_commit:
            self.git_commit = git_commit()
            self.git_dirty = _git_is_dirty()
        if not self.python_version:
            self.python_version = platform.python_version()
        if not self.package_versions:
            self.package_versions = _collect_versions()

    def site(self, site_id: str) -> SiteArtifact | None:
        return next((s for s in self.sites if s.site_id == site_id), None)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["sites"] = [s.to_dict() for s in self.sites]
        return payload

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        logger.info("Wrote artifact manifest -> %s", path)
        return path

    @classmethod
    def from_dict(cls, raw: dict) -> ArtifactManifest:
        sites = [
            SiteArtifact(
                **{k: v for k, v in entry.items() if k in SiteArtifact.__dataclass_fields__}
            )
            for entry in raw.get("sites", [])
        ]
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and k != "sites"}
        return cls(**known, sites=sites)


def load_manifest(path: Path) -> ArtifactManifest | None:
    """Load a manifest, or None when it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return ArtifactManifest.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not parse artifact manifest at %s: %s", path, exc)
        return None


def _collect_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour affects the artifacts."""
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "pvlib", "scipy"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:
            continue
    return versions
