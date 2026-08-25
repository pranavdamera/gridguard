"""
Fault categories and hypotheses.

Categories
----------
The shipped taxonomy in :mod:`gridguard.data.faults` has eight classes split
along one axis: whether generation is actually lost. That split is the right one
for a detector, whose job is to fire on real loss and stay silent otherwise.

It is the wrong split for attribution. "Nothing is being lost" lumps a frozen
pyranometer together with a dead radio link and with normal inverter clipping —
three conditions with completely different causes, different operator responses,
and different consequences for whether the rest of the system can be trusted. So
a second, orthogonal axis is introduced here: what kind of thing went wrong.

    equipment       The plant itself is producing less. Real energy is lost.
    sensor          The plant is fine; the instrument reading it is not.
    communication   The plant may be fine; we cannot see it.
    environmental   External and physical — shading, soiling, weather.

The existing eight classes keep their exact string values, so committed
artifacts, stored parquet and API responses stay wire-compatible. Category is
additive metadata, not a renaming.

Hypotheses
----------
:class:`FaultHypothesis` lands now, ahead of the engine that will produce it, so
that the shape of an attribution result is fixed before anything depends on it.
Two properties of that shape matter and are enforced here:

* A hypothesis names an **asset**, not just a site. Attribution that cannot say
  which component is suspect is detection wearing a different label.
* Confidence carries its **evidence**. A bare number is not interpretable, and
  the project's stated rule is that confidence must come from a stated rule or a
  calibration process rather than being asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FaultCategory(StrEnum):
    """What kind of thing failed."""

    EQUIPMENT = "equipment"
    SENSOR = "sensor"
    COMMUNICATION = "communication"
    ENVIRONMENTAL = "environmental"


#: Category for each shipped fault class, keyed by its ``FaultType`` string
#: value. Kept as strings rather than importing ``FaultType`` so this module
#: stays free of a dependency on the injection machinery — the categories are
#: part of the domain vocabulary, the injector is one consumer of it.
FAULT_CATEGORIES: dict[str, FaultCategory] = {
    "complete_outage": FaultCategory.EQUIPMENT,
    "partial_outage": FaultCategory.EQUIPMENT,
    "persistent_derate": FaultCategory.EQUIPMENT,
    "gradual_degradation": FaultCategory.EQUIPMENT,
    "clipping": FaultCategory.EQUIPMENT,
    "shading": FaultCategory.ENVIRONMENTAL,
    "sensor_dropout": FaultCategory.SENSOR,
    "comm_dropout": FaultCategory.COMMUNICATION,
}


def category_for(fault_type: str) -> FaultCategory:
    """Category of a fault class, by its string value."""
    try:
        return FAULT_CATEGORIES[str(fault_type)]
    except KeyError:
        raise KeyError(
            f"No category declared for fault type {fault_type!r}. Every fault class "
            "must be categorised — add it to FAULT_CATEGORIES."
        ) from None


@dataclass(frozen=True)
class Evidence:
    """One observation supporting or opposing a hypothesis.

    ``supports`` is signed so that evidence *against* a hypothesis is
    representable. An attribution engine that can only accumulate support will
    happily rank a class that the data actively contradicts.
    """

    name: str
    detail: str
    supports: bool
    #: Strength of this single piece of evidence, 0..1. Where an engine derives
    #: this from measured true/false-positive rates, it should say so in
    #: ``detail`` rather than leaving the number unexplained.
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"evidence weight must be in [0, 1], got {self.weight}")


@dataclass(frozen=True)
class FaultHypothesis:
    """A candidate explanation for an observed deviation."""

    fault_type: str
    category: FaultCategory
    asset_id: str
    site_id: str

    #: Posterior probability that this class explains the observation. Must come
    #: from a stated rule or a calibration process — never assigned by hand.
    confidence: float

    #: How ``confidence`` was arrived at, e.g. the name of the calibration run
    #: behind the per-signature rates. Required: an unexplained number is not
    #: an interpretable one.
    confidence_basis: str

    evidence: tuple[Evidence, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.confidence_basis.strip():
            raise ValueError(
                f"{self.fault_type}: confidence_basis is required. A confidence value "
                "with no stated basis is exactly the thing this project forbids."
            )

    @property
    def supporting(self) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.supports)

    @property
    def opposing(self) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if not e.supports)

    def explain(self) -> str:
        """A human-readable account of why this hypothesis is ranked where it is."""
        lines = [
            f"{self.fault_type} ({self.category}) at {self.asset_id}: "
            f"confidence {self.confidence:.2f} [{self.confidence_basis}]"
        ]
        for evidence in self.evidence:
            mark = "+" if evidence.supports else "-"
            lines.append(f"  {mark} {evidence.name} (w={evidence.weight:.2f}): {evidence.detail}")
        return "\n".join(lines)
