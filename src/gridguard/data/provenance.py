"""
Dataset provenance records.

Every curated dataset GridGuard ships or builds is accompanied by a provenance
record answering, without ambiguity: where did these numbers come from, what do
they measure, over what period, and what did GridGuard do to them?

This exists because GridGuard mixes measured telemetry with simulated
demonstration data. Provenance is what keeps that mixture honest — the ``/data``
page in the web application is rendered directly from these records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROVENANCE_FILENAME = "provenance.json"


@dataclass
class DatasetProvenance:
    """Where one site's dataset came from and what was done to it.

    Fields are deliberately explicit rather than a free-form blob so the web
    application can render them as a table without special-casing.
    """

    # --- identity -----------------------------------------------------------
    site_id: str
    data_mode: str  # "real" | "synthetic"
    dataset: str  # e.g. "NREL PVDAQ (OEDI data lake)" | "GridGuard DMV simulator"

    # --- physical system ----------------------------------------------------
    system_identifier: str = ""  # upstream system/system_id, empty for synthetic
    site_name: str = ""
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None
    capacity_kw: float | None = None
    capacity_basis: str = ""  # "DC nameplate", "AC nameplate", "illustrative estimate"
    tilt_deg: float | None = None
    azimuth_deg: float | None = None

    # --- measurement --------------------------------------------------------
    interval_minutes: int | None = None
    native_interval_minutes: int | None = None
    start: str = ""  # ISO 8601, local standard time
    end: str = ""
    timezone_note: str = ""
    #: Fixed UTC offset of ``start``/``end`` and of the frame's ``timestamp``
    #: column, in hours. ``timezone_note`` explains it for a reader; this is
    #: what code uses. The schema shim cannot derive UTC event times without
    #: it, so it is recorded rather than re-inferred at every call site.
    utc_offset_hours: int | None = None
    row_count: int | None = None

    # --- weather ------------------------------------------------------------
    weather_source: str = ""
    irradiance_kind: str = ""  # "plane-of-array" | "global horizontal" | "modelled"
    irradiance_channel: str = ""  # upstream column name, for traceability
    irradiance_scale_factor: float | None = None
    irradiance_scale_basis: str = ""

    # --- legal / reference --------------------------------------------------
    source_url: str = ""
    license_note: str = ""

    # --- processing ---------------------------------------------------------
    retrieved_at: str = ""
    processing_notes: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.retrieved_at:
            self.retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
        if self.data_mode not in ("real", "synthetic"):
            raise ValueError(f"data_mode must be 'real' or 'synthetic', got {self.data_mode!r}")

    @property
    def is_real(self) -> bool:
        return self.data_mode == "real"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> DatasetProvenance:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def save_provenance(records: list[DatasetProvenance], path: Path) -> Path:
    """Write provenance records to a JSON file, newest write wins."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "datasets": [r.to_dict() for r in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_provenance(path: Path) -> list[DatasetProvenance]:
    """Read provenance records back. Returns [] when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [DatasetProvenance.from_dict(d) for d in raw.get("datasets", [])]
