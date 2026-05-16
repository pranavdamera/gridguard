"""Tests for site registry and site-aware synthetic generation."""

from pathlib import Path

import pandas as pd
import pytest

from gridguard.sites.registry import get_site, list_site_ids, load_sites, Site


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def test_load_sites_returns_dict():
    sites = load_sites()
    assert isinstance(sites, dict)
    assert len(sites) >= 7  # at least the 7 DMV sites


def test_all_required_dmv_sites_present():
    ids = list_site_ids()
    expected = {
        "gmu_fairfax",
        "nova_annandale",
        "nova_alexandria",
        "nova_loudoun",
        "nova_manassas",
        "nova_woodbridge",
        "dc_community",
    }
    assert expected.issubset(set(ids)), f"Missing sites: {expected - set(ids)}"


def test_site_fields_populated():
    site = get_site("gmu_fairfax")
    assert isinstance(site, Site)
    assert site.name
    assert 38.0 < site.latitude < 40.0  # Northern Virginia latitude range
    assert -78.5 < site.longitude < -76.5  # Northern Virginia longitude range
    assert site.capacity_kw > 0


def test_get_site_unknown_raises_key_error():
    with pytest.raises(KeyError, match="Unknown site_id"):
        get_site("does_not_exist")


def test_site_is_frozen():
    site = get_site("nova_annandale")
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        site.capacity_kw = 999.0


def test_list_site_ids_sorted():
    ids = list_site_ids()
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Site-aware synthetic generation
# ---------------------------------------------------------------------------


def test_load_raw_data_with_site_id(tmp_path):
    from gridguard.ingestion.download import load_raw_data

    df = load_raw_data(source="synthetic", output_dir=tmp_path, site_id="gmu_fairfax")
    assert "site_id" in df.columns
    assert (df["site_id"] == "gmu_fairfax").all()
    assert len(df) > 0


def test_load_raw_data_without_site_id(tmp_path):
    """Existing behaviour: no site_id column when none provided."""
    from gridguard.ingestion.download import load_raw_data

    df = load_raw_data(source="synthetic", output_dir=tmp_path)
    assert "site_id" not in df.columns


def test_per_site_cache_is_separate(tmp_path):
    """Different site IDs produce separate cache files."""
    from gridguard.ingestion.download import load_raw_data

    load_raw_data(source="synthetic", output_dir=tmp_path, site_id="gmu_fairfax")
    load_raw_data(source="synthetic", output_dir=tmp_path, site_id="nova_alexandria")

    assert (tmp_path / "raw_gmu_fairfax.parquet").exists()
    assert (tmp_path / "raw_nova_alexandria.parquet").exists()
    # Default cache should not exist
    assert not (tmp_path / "raw.parquet").exists()


def test_different_latitudes_produce_different_irradiance():
    """Latitude affects sun geometry: compared over a full year, lower latitudes
    receive higher mean irradiance because the sun is more directly overhead."""
    from gridguard.ingestion.download import _generate_synthetic

    # Use a full year so seasonal averaging works correctly.
    # In summer, higher latitudes have longer days (which can offset lower sun angle),
    # but over a full year the lower-latitude site receives more total irradiance.
    df_low = _generate_synthetic(start="2022-01-01", end="2022-12-31", latitude=35.0, seed=0)
    df_high = _generate_synthetic(start="2022-01-01", end="2022-12-31", latitude=55.0, seed=0)

    # Clear-sky peak (90th percentile) should be higher at lower latitude
    p90_low = df_low["irradiance_wm2"].quantile(0.90)
    p90_high = df_high["irradiance_wm2"].quantile(0.90)
    assert p90_low > p90_high, (
        f"lat=35 90th-pct irradiance ({p90_low:.1f}) should exceed lat=55 ({p90_high:.1f})"
    )


def test_capacity_scales_ac_power():
    """Doubling capacity_kw should roughly double peak AC power."""
    from gridguard.ingestion.download import _generate_synthetic

    df_small = _generate_synthetic(start="2022-07-01", end="2022-07-31", system_capacity_kw=10.0)
    df_large = _generate_synthetic(start="2022-07-01", end="2022-07-31", system_capacity_kw=20.0)

    ratio = df_large["ac_power_kw"].max() / df_small["ac_power_kw"].max()
    assert 1.8 < ratio < 2.2, f"Expected ~2x capacity ratio, got {ratio:.2f}"
