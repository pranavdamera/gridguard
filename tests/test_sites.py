"""Tests for site registry and site-aware synthetic generation."""

import pandas as pd
import pytest

from gridguard.data.schema import CANONICAL_COLUMNS
from gridguard.data.synthetic import generate_site_telemetry
from gridguard.sites.registry import (
    REAL_DATA_NOTE,
    SYNTHETIC_DISCLAIMER,
    Site,
    get_site,
    list_site_ids,
    list_sites,
    load_sites,
    sites_as_records,
)
from tests._helpers import make_frame

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def test_load_sites_returns_dict():
    sites = load_sites()
    assert isinstance(sites, dict)
    assert len(sites) >= 7


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
# Data-mode separation
# ---------------------------------------------------------------------------


def test_registry_carries_both_data_modes():
    modes = {s.data_mode for s in list_sites()}
    assert modes == {"real", "synthetic"}


def test_real_sites_have_source_system_and_geometry():
    """A real site must be traceable upstream and modellable by physics."""
    for site in list_sites(data_mode="real"):
        assert site.source_system_id is not None, f"{site.site_id} has no source_system_id"
        assert site.has_mount_geometry, f"{site.site_id} is missing tilt/azimuth"
        assert site.capacity_basis, f"{site.site_id} does not state its capacity basis"


def test_synthetic_sites_carry_a_visible_disclaimer():
    for site in list_sites(data_mode="synthetic"):
        assert site.disclaimer == SYNTHETIC_DISCLAIMER
        assert "not measured operational data" in site.disclaimer.lower()


def test_real_sites_are_labelled_as_measured():
    for site in list_sites(data_mode="real"):
        assert site.disclaimer == REAL_DATA_NOTE
        assert "measured" in site.disclaimer.lower()


def test_records_expose_data_mode_and_disclaimer():
    """Anything rendered in a UI must be able to state its provenance."""
    for record in sites_as_records():
        assert record["data_mode"] in ("real", "synthetic")
        assert record["disclaimer"]


def test_unknown_data_mode_in_csv_is_rejected(tmp_path):
    csv = tmp_path / "sites.csv"
    csv.write_text(
        "site_id,name,region,data_mode,latitude,longitude,capacity_kw\n"
        "bad,Bad Site,DMV,imaginary,38.0,-77.0,10.0\n"
    )
    with pytest.raises(ValueError, match="data_mode"):
        load_sites(csv)


# ---------------------------------------------------------------------------
# Site-aware synthetic generation
# ---------------------------------------------------------------------------


def _site(**overrides) -> Site:
    """A synthetic site with overridable physical parameters."""
    base = dict(
        site_id="test_site",
        name="Test Site",
        region="DMV",
        data_mode="synthetic",
        latitude=38.8,
        longitude=-77.3,
        capacity_kw=100.0,
        elevation_m=50.0,
        tilt_deg=25.0,
        azimuth_deg=180.0,
    )
    base.update(overrides)
    return Site(**base)


def test_generated_frame_satisfies_canonical_schema():
    frame = make_frame(start="2022-07-01", end="2022-07-31")
    for column in CANONICAL_COLUMNS:
        assert column in frame.columns, f"missing canonical column {column}"
    assert (frame["data_mode"] == "synthetic").all()
    assert frame["timestamp"].is_monotonic_increasing


def test_site_id_is_always_present():
    """Canonical frames are always attributable to a site."""
    frame = make_frame(start="2022-07-01", end="2022-07-15")
    assert (frame["site_id"] == "gmu_fairfax").all()


def test_lower_latitude_receives_more_irradiance():
    """Sun geometry: over a full year a lower-latitude array sees a higher peak."""
    low, _ = generate_site_telemetry(
        _site(latitude=30.0),
        start="2022-01-01",
        end="2022-12-31",
        seed=0,
        inject_scenario=False,
    )
    high, _ = generate_site_telemetry(
        _site(latitude=55.0),
        start="2022-01-01",
        end="2022-12-31",
        seed=0,
        inject_scenario=False,
    )
    p90_low = low["irradiance_wm2"].quantile(0.90)
    p90_high = high["irradiance_wm2"].quantile(0.90)
    assert (
        p90_low > p90_high
    ), f"lat=30 90th-pct irradiance ({p90_low:.1f}) should exceed lat=55 ({p90_high:.1f})"


def test_capacity_scales_ac_power():
    """Doubling nameplate roughly doubles peak AC power."""
    small, _ = generate_site_telemetry(
        _site(capacity_kw=10.0),
        start="2022-07-01",
        end="2022-07-31",
        seed=5,
        inject_scenario=False,
    )
    large, _ = generate_site_telemetry(
        _site(capacity_kw=20.0),
        start="2022-07-01",
        end="2022-07-31",
        seed=5,
        inject_scenario=False,
    )
    ratio = large["ac_power_kw"].max() / small["ac_power_kw"].max()
    assert 1.8 < ratio < 2.2, f"Expected ~2x capacity ratio, got {ratio:.2f}"


def test_generation_never_exceeds_nameplate():
    frame = make_frame(start="2022-06-01", end="2022-08-31")
    site = get_site("gmu_fairfax")
    assert frame["ac_power_kw"].max() <= site.capacity_kw + 1e-6


def test_generation_is_zero_at_night():
    frame = make_frame(start="2022-06-01", end="2022-06-30")
    night = frame[frame["irradiance_wm2"] <= 0]
    assert not night.empty
    assert (night["ac_power_kw"] == 0).all()


def test_generation_is_deterministic_for_a_seed():
    """The demo must reproduce exactly across runs."""
    first = make_frame(start="2022-07-01", end="2022-07-10", seed=123)
    second = make_frame(start="2022-07-01", end="2022-07-10", seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_different_sites_get_independent_weather():
    """Each site draws its own cloud sequence rather than sharing one."""
    gmu = make_frame(start="2022-07-01", end="2022-07-31", site_id="gmu_fairfax")
    nova = make_frame(start="2022-07-01", end="2022-07-31", site_id="nova_loudoun")
    correlation = gmu["irradiance_wm2"].corr(nova["irradiance_wm2"])
    assert correlation < 0.99, "Sites should not share an identical weather sequence"


def test_provenance_marks_synthetic_and_lists_limitations():
    _, provenance = generate_site_telemetry(
        get_site("gmu_fairfax"),
        start="2022-07-01",
        end="2022-07-31",
        inject_scenario=False,
    )
    assert provenance.data_mode == "synthetic"
    assert not provenance.is_real
    assert provenance.known_limitations
    assert any("not measured" in note.lower() for note in provenance.known_limitations)


def test_site_seed_is_stable_across_processes():
    """Determinism must survive Python's per-process string hash salt.

    The site seed is derived from the site id. Using the builtin ``hash()`` here
    would silently produce different "deterministic" data on every run, since
    Python salts string hashing per process unless PYTHONHASHSEED is pinned.
    """
    import subprocess
    import sys

    script = (
        "from tests._helpers import make_frame;"
        "print('%.6f' % make_frame(start='2022-07-01', end='2022-07-05').ac_power_kw.sum())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"Generator is not reproducible across processes: {runs}"
