"""
Conformal prediction bounds for expected solar generation.

What this gives you that a sigma threshold does not
---------------------------------------------------
The legacy detector flags an interval when its residual falls below
``-k * sigma`` of the training residuals. That is interpretable but the
threshold is heuristic: choosing ``k = 2`` implies a false-alarm rate only if
residuals happen to be Gaussian, and PV residuals emphatically are not — they
are heavy-tailed and sharply skewed by cloud transients.

Split conformal prediction replaces the assumption with a guarantee. Given
calibration residuals that are exchangeable with future residuals, the lower
bound ``q`` computed here satisfies

    P(actual - predicted >= q) >= 1 - alpha

with **no distributional assumption at all** — only exchangeability. Setting
``alpha = 0.05`` therefore means: on healthy intervals, at most ~5% should fall
below the bound. That number is a design parameter you set, not a property you
hope for.

What the guarantee does *not* say
---------------------------------
Three limitations matter in practice, and GridGuard states them wherever it
reports a bound:

1. **It is marginal, not conditional.** Coverage holds on average over the
   calibration distribution. It does not promise 95% coverage separately at
   8 a.m. and at solar noon. GridGuard addresses this with Mondrian
   (per-hour-bucket) calibration below, which restores the guarantee *within*
   each bucket.
2. **Exchangeability is an assumption about the world.** Panel degradation,
   sensor drift, and seasonal shift all break it. A bound calibrated on summer
   data is not valid for winter. This is why calibration is recomputed per
   artifact build and the window is recorded in the manifest.
3. **A breach is not a fault.** It says the interval is outside the calibrated
   range of healthy behaviour. Attribution to a cause is a separate question —
   see :mod:`gridguard.spatial.context`.

Open question: validity under degraded telemetry
------------------------------------------------
Limitation 2 above is about to get sharper. As GridGuard moves to a distributed
edge architecture, calibration and scoring residuals stop being exchangeable in
a *second*, independent way: samples go missing, arrive late, or arrive
corrupted, and which samples that happens to is not random with respect to the
conditions the site is in — a communications link under load, or a sensor
failing in heat, correlates with exactly the intervals a detector cares about.

Nothing in this module compensates for that, deliberately. Coverage under each
degradation regime is to be **measured and reported as a curve**, not assumed to
carry over and not quietly patched around. Widening the interval until the
guarantee appears to hold again would destroy the finding.

The experiment is specified in ``experiments/README.md``. Anyone changing
calibration behaviour here should read it first: degradation of interval
validity under missing, delayed and corrupted telemetry is a result this project
intends to produce, not a defect to be engineered away.

References
----------
Angelopoulos & Bates (2023), "Conformal Prediction: A Gentle Introduction",
Foundations and Trends in Machine Learning. arXiv:2107.07511.

Vovk et al. (2005), "Algorithmic Learning in a Random World" — Mondrian
conformal prediction, chapter 4.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Fallback bucket key used when an interval's bucket has no calibration data.
GLOBAL_BUCKET = "__global__"


# ---------------------------------------------------------------------------
# Core split-conformal quantile
# ---------------------------------------------------------------------------


def compute_conformal_bound(
    calibration_residuals: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Lower split-conformal bound at miscoverage level ``alpha``.

    Uses the standard finite-sample construction: with ``n`` calibration
    residuals, the bound is the ``k``-th smallest where ``k = floor(alpha *
    (n + 1))``. This guarantees ``P(residual >= bound) >= 1 - alpha`` for an
    exchangeable future residual.

    Args:
        calibration_residuals: Residuals (actual - predicted) from the
            calibration split. Must come from the same model and feature set
            used at evaluation time, and must never include evaluation data.
        alpha: Miscoverage rate in (0, 1). ``0.05`` targets 95% coverage.

    Returns:
        The bound. Returns ``-inf`` when ``n`` is too small to support the
        requested ``alpha`` (``floor(alpha*(n+1)) < 1``), which correctly means
        "no finite bound is justified by this little data" rather than
        fabricating a tight one.
    """
    residuals = np.asarray(calibration_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("calibration_residuals must be non-empty")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = residuals.size
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        logger.debug(
            "alpha=%.3f with n=%d admits no finite conformal bound (need n >= %d).",
            alpha,
            n,
            int(np.ceil(1 / alpha) - 1),
        )
        return float("-inf")

    return float(np.sort(residuals)[k - 1])


def compute_conformal_interval(
    calibration_residuals: np.ndarray,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Two-sided conformal interval at total miscoverage ``alpha``.

    Splits ``alpha`` evenly between the tails, so the interval covers at least
    ``1 - alpha`` of future residuals. The upper bound is reported for context —
    generation *above* expectation is not a fault, but it is diagnostic (it
    usually means the weather input, not the array, is off).
    """
    residuals = np.asarray(calibration_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("calibration_residuals must be non-empty")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    lower = compute_conformal_bound(residuals, alpha=alpha / 2)
    upper = -compute_conformal_bound(-residuals, alpha=alpha / 2)
    return lower, upper


def flag_conformal_anomalies(
    residuals: np.ndarray,
    calibration_residuals: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Boolean mask of residuals falling below the conformal lower bound."""
    bound = compute_conformal_bound(calibration_residuals, alpha=alpha)
    return np.asarray(residuals, dtype=float) < bound


# ---------------------------------------------------------------------------
# Mondrian (per-bucket) calibration
# ---------------------------------------------------------------------------


def irradiance_bucket(irradiance_wm2: float) -> str:
    """Bucket an interval by irradiance regime.

    Residual spread scales strongly with available irradiance: an absolute
    error of 20 kW is enormous at dawn and unremarkable at solar noon. Bucketing
    by irradiance rather than clock hour makes the conditioning physical, so it
    transfers across seasons and latitudes instead of encoding one site's
    daylight pattern.
    """
    if irradiance_wm2 < 50:
        return "night"
    if irradiance_wm2 < 250:
        return "low"
    if irradiance_wm2 < 550:
        return "medium"
    if irradiance_wm2 < 850:
        return "high"
    return "peak"


#: Bucket order, for display.
BUCKET_ORDER = ["night", "low", "medium", "high", "peak"]

#: Minimum calibration points required before a bucket gets its own bound.
MIN_BUCKET_SAMPLES = 100


@dataclass
class ConformalCalibration:
    """Per-bucket conformal bounds, persisted as a model artifact.

    This is the calibration half of the detector. It is fitted on healthy
    intervals from the training split only — never on evaluation data — so that
    a degraded test period cannot widen the very bounds meant to detect it.
    """

    alpha: float
    lower_bounds: dict[str, float] = field(default_factory=dict)
    upper_bounds: dict[str, float] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)
    calibration_start: str = ""
    calibration_end: str = ""
    n_calibration: int = 0

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        residuals: np.ndarray,
        irradiance_wm2: np.ndarray,
        alpha: float = 0.05,
        timestamps: pd.Series | None = None,
    ) -> ConformalCalibration:
        """Calibrate bounds per irradiance bucket, plus a global fallback.

        Buckets with fewer than :data:`MIN_BUCKET_SAMPLES` calibration points
        fall back to the global bound: a bound computed from 12 samples is not
        a guarantee, it is noise wearing the costume of one.
        """
        residuals = np.asarray(residuals, dtype=float)
        irradiance = np.asarray(irradiance_wm2, dtype=float)
        if residuals.shape != irradiance.shape:
            raise ValueError("residuals and irradiance_wm2 must have the same shape")

        finite = np.isfinite(residuals) & np.isfinite(irradiance)
        residuals, irradiance = residuals[finite], irradiance[finite]
        if residuals.size == 0:
            raise ValueError("No finite calibration residuals supplied.")

        calibration = cls(alpha=alpha, n_calibration=int(residuals.size))

        global_lower, global_upper = compute_conformal_interval(residuals, alpha=alpha)
        calibration.lower_bounds[GLOBAL_BUCKET] = global_lower
        calibration.upper_bounds[GLOBAL_BUCKET] = global_upper
        calibration.sample_counts[GLOBAL_BUCKET] = int(residuals.size)

        buckets = np.array([irradiance_bucket(v) for v in irradiance])
        for name in BUCKET_ORDER:
            mask = buckets == name
            count = int(mask.sum())
            calibration.sample_counts[name] = count
            if count < MIN_BUCKET_SAMPLES:
                logger.debug(
                    "Bucket '%s' has only %d calibration points (<%d); using the global bound.",
                    name,
                    count,
                    MIN_BUCKET_SAMPLES,
                )
                continue
            lower, upper = compute_conformal_interval(residuals[mask], alpha=alpha)
            calibration.lower_bounds[name] = lower
            calibration.upper_bounds[name] = upper

        if timestamps is not None and len(timestamps):
            stamps = pd.to_datetime(pd.Series(timestamps))
            calibration.calibration_start = str(stamps.min())
            calibration.calibration_end = str(stamps.max())

        logger.info(
            "Conformal calibration fitted at alpha=%.3f on %d intervals; " "bucket bounds: %s",
            alpha,
            residuals.size,
            {k: round(v, 2) for k, v in calibration.lower_bounds.items() if k != GLOBAL_BUCKET},
        )
        return calibration

    # -- application --------------------------------------------------------

    def lower_bound_for(self, irradiance_wm2: float) -> float:
        bucket = irradiance_bucket(irradiance_wm2)
        return self.lower_bounds.get(bucket, self.lower_bounds.get(GLOBAL_BUCKET, float("-inf")))

    def upper_bound_for(self, irradiance_wm2: float) -> float:
        bucket = irradiance_bucket(irradiance_wm2)
        return self.upper_bounds.get(bucket, self.upper_bounds.get(GLOBAL_BUCKET, float("inf")))

    def lower_bounds_for_array(self, irradiance_wm2: np.ndarray) -> np.ndarray:
        return np.array([self.lower_bound_for(v) for v in np.asarray(irradiance_wm2, dtype=float)])

    def upper_bounds_for_array(self, irradiance_wm2: np.ndarray) -> np.ndarray:
        return np.array([self.upper_bound_for(v) for v in np.asarray(irradiance_wm2, dtype=float)])

    def coverage_on(self, residuals: np.ndarray, irradiance_wm2: np.ndarray) -> dict[str, float]:
        """Empirical coverage of the lower bound, overall and per bucket.

        Run on a held-out healthy split, this is the check that the guarantee
        actually held out of sample.
        """
        residuals = np.asarray(residuals, dtype=float)
        irradiance = np.asarray(irradiance_wm2, dtype=float)
        bounds = self.lower_bounds_for_array(irradiance)
        covered = residuals >= bounds

        report = {"overall": float(np.mean(covered)) if covered.size else float("nan")}
        buckets = np.array([irradiance_bucket(v) for v in irradiance])
        for name in BUCKET_ORDER:
            mask = buckets == name
            if mask.sum():
                report[name] = float(np.mean(covered[mask]))
        return report

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "sample_counts": self.sample_counts,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "n_calibration": self.n_calibration,
            "bucket_definition": "irradiance_wm2: night<50, low<250, medium<550, high<850, peak>=850",
            "guarantee": (
                f"Under exchangeability, at least {(1 - self.alpha) * 100:.0f}% of healthy "
                "intervals fall at or above the lower bound within each calibrated bucket."
            ),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        logger.info("Saved conformal calibration -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> ConformalCalibration | None:
        path = Path(path)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return cls(
            alpha=float(raw["alpha"]),
            lower_bounds={k: float(v) for k, v in raw.get("lower_bounds", {}).items()},
            upper_bounds={k: float(v) for k, v in raw.get("upper_bounds", {}).items()},
            sample_counts={k: int(v) for k, v in raw.get("sample_counts", {}).items()},
            calibration_start=raw.get("calibration_start", ""),
            calibration_end=raw.get("calibration_end", ""),
            n_calibration=int(raw.get("n_calibration", 0)),
        )
