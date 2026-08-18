"""Geospatial analytics: distance, neighbourhoods, and regional attribution."""

from gridguard.spatial.context import (
    AnomalyScope,
    SpatialContext,
    classify_anomaly_scope,
    compute_fleet_spatial_context,
)
from gridguard.spatial.geo import (
    bounding_box,
    haversine_km,
    nearest_neighbors,
    neighbor_graph,
    pairwise_distance_matrix,
)

__all__ = [
    "AnomalyScope",
    "SpatialContext",
    "bounding_box",
    "classify_anomaly_scope",
    "compute_fleet_spatial_context",
    "haversine_km",
    "nearest_neighbors",
    "neighbor_graph",
    "pairwise_distance_matrix",
]
