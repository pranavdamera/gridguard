"""Tests for geodesic distance and neighbour-aware anomaly attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridguard.sites.registry import Site, get_site, list_sites
from gridguard.spatial.context import (
    AnomalyScope,
    classify_anomaly_scope,
    compute_fleet_spatial_context,
    normalised_residual,
)
from gridguard.spatial.geo import (
    bounding_box,
    haversine_km,
    nearest_neighbors,
    neighbor_graph,
    pairwise_distance_matrix,
)

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def test_haversine_zero_for_same_point():
    assert haversine_km(38.8, -77.0, 38.8, -77.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_matches_known_distance():
    """DC to Gaithersburg MD is about 29 km; allow 1 km for the spherical model."""
    distance = haversine_km(38.9072, -77.0369, 39.1319, -77.2141)
    assert distance == pytest.approx(29.3, abs=1.0)


def test_haversine_one_degree_latitude_is_about_111km():
    assert haversine_km(38.0, -77.0, 39.0, -77.0) == pytest.approx(111.2, abs=0.5)


def test_haversine_is_symmetric():
    a = haversine_km(38.8, -77.3, 39.1, -77.2)
    b = haversine_km(39.1, -77.2, 38.8, -77.3)
    assert a == pytest.approx(b)


def test_haversine_vectorises():
    lats = np.array([38.8, 39.0])
    lons = np.array([-77.3, -77.2])
    result = haversine_km(lats, lons, 38.9, -77.0)
    assert result.shape == (2,)
    assert (result > 0).all()


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    sites = list_sites()
    ids, matrix = pairwise_distance_matrix(sites)
    assert len(ids) == len(sites)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_nist_arrays_are_on_one_campus():
    """Sanity check on the registry coordinates: under 1 km apart."""
    for other in ("nist_canopy", "nist_roof"):
        site_a, site_b = get_site("nist_ground"), get_site(other)
        assert (
            haversine_km(site_a.latitude, site_a.longitude, site_b.latitude, site_b.longitude) < 1.0
        )


def test_neighbor_graph_excludes_self_and_respects_radius():
    sites = list_sites()
    graph = neighbor_graph(sites, radius_km=15.0)
    for site_id, neighbors in graph.items():
        assert site_id not in [n for n, _ in neighbors]
        assert all(distance <= 15.0 for _, distance in neighbors)


def test_neighbor_graph_is_sorted_nearest_first():
    graph = neighbor_graph(list_sites(), radius_km=100.0)
    for neighbors in graph.values():
        distances = [d for _, d in neighbors]
        assert distances == sorted(distances)


def test_nearest_neighbors_returns_k():
    result = nearest_neighbors(list_sites(), "gmu_fairfax", k=3)
    assert len(result) == 3
    assert "gmu_fairfax" not in [site_id for site_id, _ in result]


def test_nearest_neighbors_unknown_site_raises():
    with pytest.raises(KeyError):
        nearest_neighbors(list_sites(), "nowhere", k=2)


def test_bounding_box_contains_every_site():
    sites = list_sites()
    box = bounding_box(sites)
    for site in sites:
        assert box["min_latitude"] <= site.latitude <= box["max_latitude"]
        assert box["min_longitude"] <= site.longitude <= box["max_longitude"]


def test_bounding_box_on_empty_fleet_raises():
    with pytest.raises(ValueError):
        bounding_box([])


# ---------------------------------------------------------------------------
# Residual normalisation
# ---------------------------------------------------------------------------


def _frame(actual: float, predicted: float, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-06-15 09:00", periods=n, freq="15min"),
            "ac_power_kw": [actual] * n,
            "predicted_kw": [predicted] * n,
        }
    )


def test_normalised_residual_scales_by_capacity():
    """The same absolute shortfall is worse for a smaller array."""
    frame = _frame(actual=80.0, predicted=100.0)
    small = normalised_residual(frame, capacity_kw=100.0)
    large = normalised_residual(frame, capacity_kw=500.0)
    assert small == pytest.approx(-0.20)
    assert large == pytest.approx(-0.04)
    assert small < large


def test_normalised_residual_handles_empty_and_zero_capacity():
    assert normalised_residual(pd.DataFrame(), capacity_kw=100.0) == 0.0
    assert normalised_residual(_frame(80.0, 100.0), capacity_kw=0.0) == 0.0


# ---------------------------------------------------------------------------
# Scope classification
# ---------------------------------------------------------------------------


def test_site_specific_when_neighbours_are_healthy():
    context = classify_anomaly_scope(
        site_id="gmu_fairfax",
        site_residual=-0.30,
        neighbor_residuals={"a": 0.00, "b": -0.01, "c": 0.01},
        neighbor_distances={"a": 5.0, "b": 10.0, "c": 15.0},
        radius_km=60.0,
    )
    assert context.scope is AnomalyScope.SITE_SPECIFIC
    assert context.confidence == "high"
    assert context.neighbors_affected == 0
    assert context.explanation


def test_regional_when_neighbours_drop_together():
    context = classify_anomaly_scope(
        site_id="gmu_fairfax",
        site_residual=-0.18,
        neighbor_residuals={"a": -0.16, "b": -0.20, "c": -0.15},
        neighbor_distances={"a": 5.0, "b": 10.0, "c": 15.0},
        radius_km=60.0,
    )
    assert context.scope is AnomalyScope.REGIONAL
    assert context.neighbors_affected == 3
    assert context.regional_anomaly_score == pytest.approx(1.0)


def test_too_few_neighbours_yields_indeterminate():
    """With one neighbour, no spatial claim is made rather than a guess."""
    context = classify_anomaly_scope(
        site_id="lonely",
        site_residual=-0.40,
        neighbor_residuals={"a": 0.0},
        neighbor_distances={"a": 5.0},
        radius_km=60.0,
    )
    assert context.scope is AnomalyScope.INDETERMINATE
    assert context.confidence == "low"
    assert "too few" in context.explanation.lower()


def test_no_neighbours_yields_indeterminate():
    context = classify_anomaly_scope("alone", -0.5, {}, {}, radius_km=60.0)
    assert context.scope is AnomalyScope.INDETERMINATE
    assert context.neighbor_count == 0


def test_healthy_site_among_healthy_neighbours_is_not_site_specific():
    context = classify_anomaly_scope(
        site_id="fine",
        site_residual=0.01,
        neighbor_residuals={"a": 0.0, "b": 0.01, "c": -0.01},
        neighbor_distances={"a": 5.0, "b": 10.0, "c": 15.0},
        radius_km=60.0,
    )
    assert context.scope is not AnomalyScope.SITE_SPECIFIC


def test_distance_weighting_favours_nearer_neighbours():
    """A near neighbour should move the weighted residual more than a far one."""
    context = classify_anomaly_scope(
        site_id="x",
        site_residual=-0.10,
        neighbor_residuals={"near": -0.30, "far": 0.00, "far2": 0.00},
        neighbor_distances={"near": 1.0, "far": 50.0, "far2": 55.0},
        radius_km=60.0,
    )
    # Unweighted mean is -0.10; inverse-distance weighting pulls it toward the
    # near neighbour's -0.30.
    assert context.distance_weighted_residual < context.neighbor_residual_mean


def test_evidence_is_always_reported():
    """Every verdict exposes the numbers behind it so a human can disagree."""
    context = classify_anomaly_scope(
        site_id="x",
        site_residual=-0.30,
        neighbor_residuals={"a": 0.0, "b": 0.0},
        neighbor_distances={"a": 5.0, "b": 10.0},
        radius_km=60.0,
    )
    payload = context.to_dict()
    for key in (
        "site_normalised_residual",
        "neighbor_residual_median",
        "distance_weighted_residual",
        "neighbors_affected",
        "neighbors_considered",
        "explanation",
    ):
        assert key in payload


# ---------------------------------------------------------------------------
# Fleet-level context
# ---------------------------------------------------------------------------


def test_fleet_context_never_mixes_data_modes():
    """A simulated site's residuals say nothing about a measured site's weather.

    The synthetic generator draws each site's weather independently, so allowing
    a synthetic neighbour to inform a real site's verdict would manufacture
    agreement out of noise.
    """
    frames = {
        "nist_ground": _frame(50.0, 200.0),  # measured, badly underperforming
        "nist_canopy": _frame(200.0, 200.0),
        "nist_roof": _frame(60.0, 60.0),
        "gmu_fairfax": _frame(10.0, 200.0),  # simulated, also underperforming
        "nova_annandale": _frame(80.0, 80.0),
    }
    contexts = compute_fleet_spatial_context(frames, radius_km=100.0)

    real_neighbors = set(contexts["nist_ground"].neighbors_considered)
    assert real_neighbors <= {"nist_canopy", "nist_roof"}

    synthetic_neighbors = set(contexts["gmu_fairfax"].neighbors_considered)
    assert not (synthetic_neighbors & {"nist_ground", "nist_canopy", "nist_roof"})


def test_fleet_context_identifies_the_single_bad_site():
    frames = {
        "nist_ground": _frame(60.0, 250.0),  # deep shortfall
        "nist_canopy": _frame(230.0, 235.0),
        "nist_roof": _frame(70.0, 71.0),
    }
    contexts = compute_fleet_spatial_context(frames, radius_km=100.0)
    assert contexts["nist_ground"].scope is AnomalyScope.SITE_SPECIFIC
    assert contexts["nist_canopy"].scope is not AnomalyScope.SITE_SPECIFIC


def test_fleet_context_ignores_unknown_sites():
    contexts = compute_fleet_spatial_context(
        {"not_in_registry": _frame(10.0, 20.0)}, radius_km=60.0
    )
    assert contexts == {}


def test_fleet_context_accepts_explicit_site_list():
    sites = [
        Site(
            site_id="a",
            name="A",
            region="X",
            data_mode="synthetic",
            latitude=38.0,
            longitude=-77.0,
            capacity_kw=100.0,
        ),
        Site(
            site_id="b",
            name="B",
            region="X",
            data_mode="synthetic",
            latitude=38.05,
            longitude=-77.0,
            capacity_kw=100.0,
        ),
        Site(
            site_id="c",
            name="C",
            region="X",
            data_mode="synthetic",
            latitude=38.1,
            longitude=-77.0,
            capacity_kw=100.0,
        ),
    ]
    frames = {
        "a": _frame(20.0, 100.0),
        "b": _frame(99.0, 100.0),
        "c": _frame(98.0, 100.0),
    }
    contexts = compute_fleet_spatial_context(frames, radius_km=50.0, sites=sites)
    assert contexts["a"].scope is AnomalyScope.SITE_SPECIFIC
