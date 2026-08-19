"""Shared fixtures.

Fixtures are deliberately small and generated in-process: the test suite must
run in CI without network access, so nothing here touches the OEDI data lake.
Tests that exercise the real-data path do so against recorded fixtures in
``tests/fixtures/`` instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridguard.data.synthetic import generate_site_telemetry
from gridguard.sites.registry import get_site


@pytest.fixture(scope="session")
def demo_site():
    return get_site("gmu_fairfax")


@pytest.fixture(scope="session")
def real_site():
    return get_site("nist_ground")


@pytest.fixture(scope="session")
def synthetic_frame(demo_site) -> pd.DataFrame:
    """Six months of clean synthetic telemetry, no injected faults."""
    frame, _ = generate_site_telemetry(
        demo_site,
        start="2022-01-01",
        end="2022-06-30",
        seed=77,
        inject_scenario=False,
    )
    return frame


@pytest.fixture(scope="session")
def faulted_frame(demo_site) -> pd.DataFrame:
    """A year of synthetic telemetry with the full labelled fault taxonomy."""
    frame, _ = generate_site_telemetry(
        demo_site,
        start="2022-01-01",
        end="2022-12-31",
        seed=13,
        inject_scenario=True,
    )
    return frame


@pytest.fixture()
def fitted_weather_model(synthetic_frame):
    """A cheap weather-only model, for detector and calibration tests."""
    from sklearn.ensemble import RandomForestRegressor

    from gridguard.features.engineer import get_X_y

    X, y = get_X_y(synthetic_frame, mode="weather_only")
    model = RandomForestRegressor(n_estimators=25, max_depth=8, random_state=0, n_jobs=1)
    model.fit(X, y)
    return model
