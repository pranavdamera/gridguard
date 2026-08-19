"""
Geodesic distance and fleet neighbourhood structure.

Distances use the haversine formula on a spherical Earth. At the scale of a
regional PV fleet — tens of kilometres — the difference between a spherical and
an ellipsoidal (WGS-84) model is well under 0.5%, far smaller than the
uncertainty in whether two arrays share weather at all. Haversine is used
because it is exact for the model it assumes, dependency-free, and easy to
verify against published distances.
"""

from __future__ import annotations

import numpy as np

from gridguard.sites.registry import Site

#: Mean Earth radius (IUGG), kilometres.
EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle distance in kilometres. Scalars or broadcastable arrays.

    >>> round(haversine_km(38.8977, -77.0365, 38.8895, -77.0353), 1)  # WH -> WM
    0.9
    """
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    # arcsin form is numerically better than arccos for small distances.
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(distance) if np.isscalar(lat1) and np.isscalar(lat2) else distance


def pairwise_distance_matrix(sites: list[Site]) -> tuple[list[str], np.ndarray]:
    """Return ``(site_ids, distance_matrix_km)`` for a fleet.

    The matrix is symmetric with a zero diagonal, ordered to match ``site_ids``.
    """
    site_ids = [s.site_id for s in sites]
    lats = np.array([s.latitude for s in sites], dtype=float)
    lons = np.array([s.longitude for s in sites], dtype=float)
    matrix = haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    return site_ids, np.asarray(matrix)


def neighbor_graph(sites: list[Site], radius_km: float) -> dict[str, list[tuple[str, float]]]:
    """Map each site to its neighbours within ``radius_km``.

    Each value is a list of ``(neighbour_site_id, distance_km)`` sorted nearest
    first. A site is never its own neighbour.
    """
    site_ids, matrix = pairwise_distance_matrix(sites)
    graph: dict[str, list[tuple[str, float]]] = {}
    for i, site_id in enumerate(site_ids):
        neighbors = [
            (site_ids[j], float(matrix[i, j]))
            for j in range(len(site_ids))
            if j != i and matrix[i, j] <= radius_km
        ]
        graph[site_id] = sorted(neighbors, key=lambda pair: pair[1])
    return graph


def nearest_neighbors(
    sites: list[Site],
    site_id: str,
    k: int = 3,
) -> list[tuple[str, float]]:
    """The ``k`` nearest sites to ``site_id`` as ``(site_id, distance_km)``."""
    site_ids, matrix = pairwise_distance_matrix(sites)
    if site_id not in site_ids:
        raise KeyError(f"Unknown site_id '{site_id}'. Known: {site_ids}")
    index = site_ids.index(site_id)
    ordered = [
        (site_ids[j], float(matrix[index, j])) for j in np.argsort(matrix[index]) if j != index
    ]
    return ordered[:k]


def bounding_box(sites: list[Site], pad_deg: float = 0.05) -> dict[str, float]:
    """Padded lat/lon bounds for a fleet, for fitting a map viewport."""
    if not sites:
        raise ValueError("Cannot compute a bounding box for an empty fleet.")
    lats = [s.latitude for s in sites]
    lons = [s.longitude for s in sites]
    return {
        "min_latitude": min(lats) - pad_deg,
        "max_latitude": max(lats) + pad_deg,
        "min_longitude": min(lons) - pad_deg,
        "max_longitude": max(lons) + pad_deg,
        "center_latitude": (min(lats) + max(lats)) / 2,
        "center_longitude": (min(lons) + max(lons)) / 2,
    }
