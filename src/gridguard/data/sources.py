"""
Data source abstraction.

One interface, two implementations, dispatched by the site's ``data_mode``:

    DataSource
    ├── RealPVDataSource      measured PVDAQ telemetry via the OEDI data lake
    └── SyntheticDMVSource    simulated DMV campus fleet

Both return a frame satisfying the canonical schema plus a
:class:`~gridguard.data.provenance.DatasetProvenance` record, so everything
downstream — features, models, anomaly detection, the API — is agnostic to
where the numbers came from while the provenance distinction stays explicit and
visible all the way to the UI.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gridguard.config import settings
from gridguard.data import oedi
from gridguard.data.provenance import DatasetProvenance
from gridguard.data.schema import (
    clip_physical_ranges,
    drop_unusable_rows,
    resample_to_interval,
    validate_canonical,
)
from gridguard.data.synthetic import UTC_OFFSET_HOURS, generate_site_telemetry
from gridguard.sites.registry import Site, get_site

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Loads canonical telemetry for one site."""

    @abstractmethod
    def load(self, site: Site, **kwargs) -> tuple[pd.DataFrame, DatasetProvenance]:
        """Return canonical telemetry and its provenance for ``site``."""


# ---------------------------------------------------------------------------
# Real measured telemetry
# ---------------------------------------------------------------------------


class RealPVDataSource(DataSource):
    """Measured PV telemetry from NREL PVDAQ via the OEDI data lake.

    Weather comes from instruments at the same site as the array — the PVDAQ
    systems GridGuard uses publish co-located irradiance, ambient temperature
    and wind alongside generation. That is better than reanalysis or satellite
    weather for this purpose (it is what the array actually experienced) and it
    means the real-data path needs no API credentials at all.

    Extending the fleet to a system without on-site instruments would require a
    separate weather source; none is shipped, because none of the curated
    systems needs one.
    """

    def __init__(self, interval_minutes: int = 15) -> None:
        self.interval_minutes = interval_minutes

    def load(
        self,
        site: Site,
        *,
        start: date | str,
        end: date | str,
        **kwargs,
    ) -> tuple[pd.DataFrame, DatasetProvenance]:
        if not site.is_real:
            raise ValueError(
                f"RealPVDataSource requires a real site; '{site.site_id}' is synthetic."
            )
        if site.source_system_id is None:
            raise ValueError(
                f"Site '{site.site_id}' has data_mode='real' but no source_system_id in the registry."
            )

        start_date = _as_date(start)
        end_date = _as_date(end)

        # The metrics table is the authority on what each metric_id means and
        # what units it is published in — resolve channels before fetching so
        # only the needed ones are pulled across the wire.
        metrics = oedi.fetch_metrics(site.source_system_id)
        channels = oedi.resolve_channels(metrics)
        logger.info("Resolved channels for %s: %s", site.site_id, channels.as_note())

        long_frame = oedi.fetch_range(
            site.source_system_id,
            start_date,
            end_date,
            metric_ids={c.metric_id for c in channels.all_channels()},
        )

        utc_offset = oedi.detect_utc_offset_hours(long_frame)
        if utc_offset is None:
            utc_offset = UTC_OFFSET_HOURS
            offset_note = (
                f"UTC offset not observable in the published data; assumed "
                f"UTC{UTC_OFFSET_HOURS:+d} (local standard time)."
            )
        else:
            offset_note = (
                f"Local standard time at a fixed UTC{utc_offset:+d} offset, derived from the "
                f"published measured_on and utc_measured_on columns. The offset is constant "
                f"across the range, confirming no daylight-saving shift."
            )

        wide = oedi.pivot_channels(long_frame, channels)
        native_rows = len(wide)
        native_interval = _infer_native_interval_minutes(wide["timestamp"])

        frame = wide.copy()
        frame["site_id"] = site.site_id
        frame["data_mode"] = "real"
        for column in ("temperature_c", "wind_speed_ms"):
            if column not in frame.columns:
                frame[column] = np.nan

        frame = validate_canonical(frame, strict=False)
        frame = clip_physical_ranges(frame, capacity_kw=site.capacity_kw)
        frame = resample_to_interval(frame, self.interval_minutes)

        gaps = _describe_gaps(frame)
        frame = drop_unusable_rows(frame)

        is_poa = channels.irradiance_kind == "plane-of-array"
        _, clearsky_note = oedi.validate_irradiance_against_clearsky(
            frame["timestamp"],
            frame["irradiance_wm2"],
            latitude=site.latitude,
            longitude=site.longitude,
            altitude=site.elevation_m,
            utc_offset_hours=utc_offset,
            tilt_deg=site.tilt_deg if is_poa else None,
            azimuth_deg=site.azimuth_deg if is_poa else None,
        )

        provenance = DatasetProvenance(
            site_id=site.site_id,
            data_mode="real",
            dataset="NREL PVDAQ (Open Energy Data Initiative data lake)",
            system_identifier=f"PVDAQ system_id={site.source_system_id}",
            site_name=site.name,
            latitude=site.latitude,
            longitude=site.longitude,
            elevation_m=site.elevation_m,
            capacity_kw=site.capacity_kw,
            capacity_basis=site.capacity_basis,
            tilt_deg=site.tilt_deg,
            azimuth_deg=site.azimuth_deg,
            interval_minutes=self.interval_minutes,
            native_interval_minutes=native_interval,
            start=str(frame["timestamp"].min()),
            end=str(frame["timestamp"].max()),
            timezone_note=offset_note,
            utc_offset_hours=utc_offset,
            row_count=len(frame),
            weather_source=(
                "Instruments co-located with the array (published in the same PVDAQ record)"
            ),
            irradiance_kind=channels.irradiance_kind,
            irradiance_channel=channels.irradiance.describe(),
            irradiance_scale_factor=channels.irradiance.scale,
            irradiance_scale_basis=(
                f"Published in {channels.irradiance.units} with calc_scale="
                f"{channels.irradiance.scale:g}, calc_offset={channels.irradiance.offset:g}. "
                f"{clearsky_note}"
            ),
            source_url=oedi.PVDAQ_LANDING_PAGE,
            license_note=oedi.LICENSE_NOTE,
            processing_notes=[
                f"Channels selected from the system metrics table by declared units: "
                f"{channels.as_note()}",
                f"Resampled from {native_interval}-minute to "
                f"{self.interval_minutes}-minute intervals by mean.",
                "Negative irradiance and power (instrument offsets) clamped to zero.",
                f"Assembled {native_rows} native samples into {len(frame)} usable intervals.",
                gaps,
            ],
            known_limitations=[
                "No fault labels. Measured telemetry carries no annotated failures, so this "
                "dataset supports forecast evaluation only — detection metrics come from "
                "injected scenarios instead.",
                "Where several calibrated irradiance sensors exist, the lowest metric_id is "
                "used; alternates are not cross-checked, so a drifting sensor would go "
                "unnoticed.",
                "Irradiance channels published only in raw millivolts are rejected rather "
                "than converted, since the dataset does not publish the instrument "
                "calibration constant.",
            ],
        )
        return frame, provenance


# ---------------------------------------------------------------------------
# Synthetic DMV fleet
# ---------------------------------------------------------------------------


class SyntheticDMVSource(DataSource):
    """Simulated DMV campus fleet with injected, labelled faults."""

    def load(
        self,
        site: Site,
        *,
        start: str | None = None,
        end: str | None = None,
        seed: int = 42,
        inject_scenario: bool = True,
        demo: bool = False,
        **kwargs,
    ) -> tuple[pd.DataFrame, DatasetProvenance]:
        from gridguard.data.synthetic import DEFAULT_END, DEFAULT_START

        return generate_site_telemetry(
            site,
            start=str(start or DEFAULT_START),
            end=str(end or DEFAULT_END),
            seed=seed,
            inject_scenario=inject_scenario,
            demo=demo,
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def get_source(site: Site, **kwargs) -> DataSource:
    """Return the source implementation appropriate to a site's data mode."""
    if site.is_real:
        return RealPVDataSource(**kwargs)
    return SyntheticDMVSource()


def load_site_data(
    site_id: str,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, DatasetProvenance]:
    """Load one site's canonical telemetry.

    Resolution order:

    1. **Curated** (``data/curated/``) — small measured datasets committed to the
       repository. A fresh clone can therefore run the real-data pipeline with
       no network access at all; ``make data-real`` regenerates them from the
       OEDI data lake when they need refreshing.
    2. **Cache** (``data/processed/``) — whatever a previous run downloaded or
       generated. Not version-controlled.
    3. **Source** — download from the data lake, or generate the simulation.

    Pass ``use_cache=False`` to skip 1 and 2 and force a refresh.
    """
    site = get_site(site_id)
    cache_dir = Path(cache_dir or settings.data_processed_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_demo" if kwargs.get("demo") else ""
    frame_path = cache_dir / f"telemetry_{site_id}{suffix}.parquet"
    prov_path = cache_dir / f"telemetry_{site_id}{suffix}.provenance.json"

    if use_cache:
        # Curated datasets are only meaningful for measured sites; the
        # simulation is regenerated deterministically and never shipped.
        candidates = []
        if site.is_real:
            curated = Path(settings.data_curated_dir)
            candidates.append(
                (
                    curated / f"telemetry_{site_id}.parquet",
                    curated / f"telemetry_{site_id}.provenance.json",
                    "curated",
                )
            )
        candidates.append((frame_path, prov_path, "cached"))

        for data_path, provenance_path, kind in candidates:
            if not (data_path.exists() and provenance_path.exists()):
                continue
            from gridguard.data.provenance import load_provenance

            records = load_provenance(provenance_path)
            if records:
                logger.info("Loaded %s telemetry for %s from %s", kind, site_id, data_path)
                return pd.read_parquet(data_path), records[0]

    source = get_source(site)
    frame, provenance = source.load(site, **kwargs)

    from gridguard.data.provenance import save_provenance

    frame.to_parquet(frame_path, index=False)
    save_provenance([provenance], prov_path)
    logger.info("Cached telemetry for %s -> %s (%d rows)", site_id, frame_path, len(frame))
    return frame, provenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_date(value: date | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _infer_native_interval_minutes(timestamps: pd.Series) -> int:
    deltas = pd.to_datetime(timestamps).sort_values().diff().dropna()
    if deltas.empty:
        return 0
    return int(deltas.mode().iloc[0].total_seconds() // 60)


def _describe_gaps(frame: pd.DataFrame) -> str:
    """One-line summary of missing coverage, for the provenance record."""
    missing = int(frame["ac_power_kw"].isna().sum())
    if not missing:
        return "No unrecoverable gaps after resampling."
    pct = 100 * missing / max(len(frame), 1)
    return (
        f"{missing} intervals ({pct:.2f}%) remained missing after short-gap interpolation "
        f"and were dropped rather than imputed."
    )
