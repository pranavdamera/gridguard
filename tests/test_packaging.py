"""
Packaging invariants.

These tests exist because of a specific class of failure. The Dockerfile used to
restate the dependency list by hand, and it drifted: it omitted ``pvlib`` and
``scipy`` while installing ``lightgbm``, which nothing imports. Since
``gridguard.api.main`` imports ``gridguard.data.synthetic`` at module level,
which imports ``pvlib`` at module level, the image could not start the API at
all — every container run died on import.

Nothing in the test suite caught it, because the suite runs against a normally
installed environment where the dependencies are present regardless of what the
image declares. So the invariant is asserted directly: whatever the API actually
imports must be something the package actually declares.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

#: Entry points a deployment actually executes.
ENTRY_POINTS = ["gridguard.api.main", "gridguard.pipeline"]


def _declared_dependencies() -> set[str]:
    """Normalised distribution names from ``[project.dependencies]``."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    return {
        re.split(r"[<>=!~\[]", spec)[0].strip().lower().replace("_", "-")
        for spec in project["dependencies"]
    }


def _module_path(module: str) -> Path | None:
    for candidate in (
        SRC / (module.replace(".", "/") + ".py"),
        SRC / module.replace(".", "/") / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def _import_graph(entry_points: list[str]) -> tuple[set[str], set[str]]:
    """Walk first-party imports from ``entry_points``.

    Returns the reachable ``gridguard`` modules and the top-level third-party
    packages they import. Only absolute, module-level-resolvable imports are
    followed, which is enough: a dependency imported lazily inside a function
    still has to be installed for that function to work.
    """
    reachable: set[str] = set()
    third_party: set[str] = set()
    stack = list(entry_points)

    while stack:
        module = stack.pop()
        if module in reachable:
            continue
        reachable.add(module)

        path = _module_path(module)
        if path is None:
            continue

        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            else:
                continue

            for name in names:
                if name.startswith("gridguard"):
                    stack.append(name)
                else:
                    third_party.add(name.split(".")[0])

    return reachable, third_party - set(sys.stdlib_module_names)


def _distributions_for(top_level: str) -> list[str]:
    """Which installed distributions provide the given top-level package."""
    return [
        name.lower().replace("_", "-")
        for name in packages_distributions().get(top_level, [top_level])
    ]


def test_api_import_graph_is_fully_declared():
    """Every third-party package the deployed code imports is a declared dependency."""
    declared = _declared_dependencies()
    _, third_party = _import_graph(ENTRY_POINTS)

    assert third_party, "import graph walk found nothing — the walker is broken, not the deps"

    undeclared = {
        top: _distributions_for(top)
        for top in sorted(third_party)
        if not any(dist in declared for dist in _distributions_for(top))
    }
    assert not undeclared, (
        "These packages are imported by deployed code but not declared in "
        f"pyproject.toml [project.dependencies]: {undeclared}. "
        "An install that honours the declared set will fail at import."
    )


def test_pvlib_and_scipy_are_reachable_from_the_api():
    """Guards the exact regression: both were dropped from the image's list.

    If this ever stops holding it means the API's import graph genuinely
    changed. That is fine — but it should be a deliberate edit here, not a
    silent one that quietly makes the omission survivable again.
    """
    _, third_party = _import_graph(["gridguard.api.main"])
    assert {"pvlib", "scipy"} <= third_party


def test_dockerfile_does_not_hand_maintain_a_dependency_list():
    """The image must derive its dependencies from pyproject, not restate them."""
    content = DOCKERFILE.read_text()

    assert "pyproject.toml" in content, "Dockerfile must read pyproject.toml"

    # A pinned requirement literal in a pip install line is the shape of the
    # bug: a second, hand-edited copy of the dependency set.
    pinned = re.findall(r'"([A-Za-z][A-Za-z0-9_.-]*(?:\[[^\]]*\])?[<>=!~]=[^"]*)"', content)
    assert not pinned, (
        f"Dockerfile pins requirements inline: {pinned}. "
        "Install from pyproject.toml instead, so the two cannot drift apart."
    )


@pytest.mark.parametrize("unused", ["lightgbm"])
def test_removed_dependencies_stay_removed(unused: str):
    """``lightgbm`` was installed by the image and imported by nothing."""
    _, third_party = _import_graph(ENTRY_POINTS)
    assert unused not in third_party
    assert unused not in _declared_dependencies()


def _optional_dependencies(extra: str) -> set[str]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    return {
        re.split(r"[<>=!~\[]", spec)[0].strip().lower().replace("_", "-")
        for spec in project["optional-dependencies"][extra]
    }


def test_collector_imports_are_covered_by_its_extra():
    """The collector is deployed code with its own image, so it gets the same guard.

    Its dependencies live in the `collector` extra rather than the core set,
    because the API reads prebuilt artifacts and needs neither an MQTT client
    nor a database driver. That split only holds if what the collector imports
    is actually declared somewhere.
    """
    declared = _declared_dependencies() | _optional_dependencies("collector")
    _, third_party = _import_graph(
        ["gridguard.collector.collector", "gridguard.collector.store", "gridguard.collector.mqtt"]
    )

    assert third_party, "import graph walk found nothing"
    undeclared = {
        top: _distributions_for(top)
        for top in sorted(third_party)
        if not any(dist in declared for dist in _distributions_for(top))
    }
    assert not undeclared, (
        f"imported by the collector but declared nowhere: {undeclared}. "
        "Add them to [project.optional-dependencies] collector."
    )


def test_the_api_image_does_not_need_the_collector_dependencies():
    """The whole point of the split: keep psycopg and paho out of the API image.

    Both are imported lazily inside functions, so reaching them from the API's
    module-level graph would mean an import moved to the top of a file and
    quietly widened what every API container has to install.
    """
    _, api_third_party = _import_graph(["gridguard.api.main"])
    assert "psycopg" not in api_third_party
    assert "paho" not in api_third_party
