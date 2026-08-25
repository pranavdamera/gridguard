"""
Identifier derivation.

Asset identifiers are derived, not assigned. A site's third inverter has the
same id on every machine, in every process, and across every rebuild, because
the id is a function of what the thing *is* rather than of when it happened to
be created.

Why a digest and not ``hash()``
-------------------------------
Python salts string hashing per process unless ``PYTHONHASHSEED`` is pinned.
This project has already been bitten by that once: the site simulator seeded
itself from ``hash(site_id)`` and produced different "deterministic" data on
every run. ``blake2b`` has no such behaviour, so anything derived here is stable
across processes, machines and Python versions.
"""

from __future__ import annotations

import re
from hashlib import blake2b

#: Identifiers are lowercase, alphanumeric, and separated by underscores.
#: Enforced rather than assumed, because these values end up in file paths,
#: URL segments and protobuf fields.
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

#: Bytes of digest used when deriving a stable numeric seed.
_SEED_DIGEST_BYTES = 8


class InvalidIdError(ValueError):
    """Raised when an identifier does not satisfy :data:`ID_PATTERN`."""


def validate_id(value: str, *, kind: str = "id") -> str:
    """Return ``value`` if it is a well-formed identifier, else raise."""
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise InvalidIdError(
            f"{kind} {value!r} is not a valid identifier: expected lowercase "
            "alphanumeric segments separated by single underscores."
        )
    return value


def asset_id(site_id: str, kind: str, ordinal: int = 1) -> str:
    """Derive an asset identifier from what the asset is.

    ``asset_id("nist_roof", "inverter", 2) -> "nist_roof_inverter_2"``

    Deterministic and human-readable, which matters more here than compactness:
    these appear in logs, CSVs and error messages that people read.
    """
    validate_id(site_id, kind="site_id")
    validate_id(kind, kind="asset kind")
    if ordinal < 1:
        raise ValueError(f"ordinal must be >= 1, got {ordinal}")
    return f"{site_id}_{kind}_{ordinal}"


def stable_seed(*parts: str, bits: int = 32) -> int:
    """A reproducible integer seed derived from string parts.

    Use anywhere a random stream must be reproducible across processes.
    """
    if not parts:
        raise ValueError("stable_seed needs at least one part")
    digest = blake2b("\x00".join(parts).encode("utf-8"), digest_size=_SEED_DIGEST_BYTES)
    return int.from_bytes(digest.digest(), "big") % (2**bits)
