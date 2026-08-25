"""
Assets: the things inside a site that can fail.

Why this exists
---------------
Until now a site was a flat record — one capacity, one tilt, one power channel.
That is enough to say "this site is underperforming", which is *detection*.
It is not enough to say "inverter 2 tripped while inverter 1 is fine", which is
*attribution*, and attribution is the question the project is moving toward.

So a site becomes a container of assets arranged in a tree:

    site
      └── array            (a physical sub-array, has tilt/azimuth/capacity)
            └── inverter   (converts that array's DC to AC)
      └── meter            (measures what the site exports)
      └── pyranometer      (measures irradiance in the array plane)
      └── weather_station  (ambient temperature, wind)

Two rules keep the tree honest and are enforced in tests:

1. **Capacity reconciles.** The sum of a site's array capacities equals the
   capacity the site registry has always reported. Introducing structure must
   not quietly change any number the project already published.
2. **Sensors are assets too.** A frozen pyranometer is a fault *of an asset*,
   not an ambient property of the site. Modelling sensors as assets is what
   lets sensor faults be attributed rather than merely detected.

The synthetic sites' internal structure is declared, not measured, exactly as
their generation is. The three NIST sites' structure comes from the published
PVDAQ system metadata already recorded in the registry notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from gridguard.domain.ids import asset_id, validate_id


class AssetKind(StrEnum):
    """What a monitored component is."""

    ARRAY = "array"
    INVERTER = "inverter"
    STRING = "string"
    METER = "meter"
    PYRANOMETER = "pyranometer"
    WEATHER_STATION = "weather_station"


#: Kinds that generate power, and therefore have a capacity.
GENERATING_KINDS: frozenset[AssetKind] = frozenset({AssetKind.ARRAY, AssetKind.STRING})

#: Kinds that measure rather than generate. A fault in one of these corrupts the
#: *observation* without changing what the site actually produced — the
#: distinction the detector has to get right to avoid false alarms.
SENSING_KINDS: frozenset[AssetKind] = frozenset(
    {AssetKind.METER, AssetKind.PYRANOMETER, AssetKind.WEATHER_STATION}
)


@dataclass(frozen=True)
class Asset:
    """One monitored component of a site."""

    asset_id: str
    site_id: str
    kind: AssetKind
    name: str = ""
    parent_id: str | None = None

    #: Nameplate capacity in kW. Set for generating assets, None otherwise.
    capacity_kw: float | None = None

    #: Mount geometry, for generating assets that have one.
    tilt_deg: float | None = None
    azimuth_deg: float | None = None

    #: The canonical telemetry column this asset is the source of, when it is
    #: the one the shipped single-channel data comes from. Assets that no
    #: shipped column maps to leave this None — the structure is declared ahead
    #: of the telemetry that will eventually populate it.
    telemetry_column: str | None = None

    #: Free text. For synthetic sites this states that the structure is
    #: illustrative, in the same terms the site registry uses.
    notes: str = field(default="")

    def __post_init__(self) -> None:
        validate_id(self.asset_id, kind="asset_id")
        validate_id(self.site_id, kind="site_id")
        if self.is_generating and self.capacity_kw is None:
            raise ValueError(f"{self.asset_id}: {self.kind} assets must declare capacity_kw")
        if not self.is_generating and self.capacity_kw is not None:
            raise ValueError(
                f"{self.asset_id}: {self.kind} assets do not generate and must not "
                "declare capacity_kw"
            )

    @property
    def is_generating(self) -> bool:
        return self.kind in GENERATING_KINDS

    @property
    def is_sensing(self) -> bool:
        return self.kind in SENSING_KINDS

    @property
    def has_mount_geometry(self) -> bool:
        return self.tilt_deg is not None and self.azimuth_deg is not None


@dataclass(frozen=True)
class AssetTree:
    """Every asset belonging to one site, with its parent/child structure."""

    site_id: str
    assets: tuple[Asset, ...]

    def __post_init__(self) -> None:
        ids = [a.asset_id for a in self.assets]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"{self.site_id}: duplicate asset ids {sorted(duplicates)}")

        known = set(ids)
        for asset in self.assets:
            if asset.site_id != self.site_id:
                raise ValueError(f"{asset.asset_id} belongs to {asset.site_id}, not {self.site_id}")
            if asset.parent_id is not None and asset.parent_id not in known:
                raise ValueError(
                    f"{asset.asset_id} names parent {asset.parent_id!r}, which is not an "
                    f"asset of {self.site_id}"
                )
            if asset.parent_id == asset.asset_id:
                raise ValueError(f"{asset.asset_id} is its own parent")

    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self):
        return iter(self.assets)

    def get(self, asset_id_: str) -> Asset:
        for asset in self.assets:
            if asset.asset_id == asset_id_:
                return asset
        raise KeyError(f"{asset_id_!r} is not an asset of {self.site_id}")

    def of_kind(self, kind: AssetKind) -> tuple[Asset, ...]:
        return tuple(a for a in self.assets if a.kind == kind)

    def children_of(self, asset_id_: str) -> tuple[Asset, ...]:
        return tuple(a for a in self.assets if a.parent_id == asset_id_)

    @property
    def total_capacity_kw(self) -> float:
        """Sum of array capacities.

        Only arrays are summed, not strings: strings are children of arrays, so
        counting both would double-count the same silicon.
        """
        return sum(a.capacity_kw or 0.0 for a in self.of_kind(AssetKind.ARRAY))

    @property
    def primary_array(self) -> Asset:
        """The largest array, used when a site-level figure needs one asset.

        This is what the schema shim attributes legacy single-channel telemetry
        to: those rows were always a whole-site measurement, and the largest
        array is the least misleading place to hang them.
        """
        arrays = self.of_kind(AssetKind.ARRAY)
        if not arrays:
            raise ValueError(f"{self.site_id} has no array assets")
        return max(arrays, key=lambda a: a.capacity_kw or 0.0)


def build_asset_id(site_id: str, kind: AssetKind | str, ordinal: int = 1) -> str:
    """Public wrapper over :func:`gridguard.domain.ids.asset_id`."""
    return asset_id(site_id, str(kind), ordinal)
