"""
Shared fixtures for the rade_qnet test suite.

Fixtures here are intentionally limited to *locating* things -- the source
tree, the test tree, the module list.  Behavioural fixtures (synthetic specs,
sources and bundles) belong in ``rade_qnet.testkit.fixtures`` so that model
authors outside this repository can use them too, and are re-exported from the
relevant sub-package ``conftest.py`` as each phase lands.

The suite is run from the repository root::

    .venv/bin/python -m pytest tests/rade_qnet -q

Running it that way puts the repository root on ``sys.path``, which is what
makes the ``src.rade_qnet`` import path resolve.  The package itself uses
relative imports internally, so it is equally importable as ``rade_qnet`` from an
installed distribution; the tests simply follow the repository's convention.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.rade_qnet.core.provenance.logging import ROOT_LOGGER_NAME

# Resolved once at import time.  ``parents`` indexes from this file outwards:
# [0] is tests/rade_qnet, [1] is tests, [2] is the repository root.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_framework_logging() -> Iterator[None]:
    """
    Undo any logging configuration a test installs.

    ``configure_logging`` is deliberately global and sticky: it sets a level,
    attaches a handler, and sets ``propagate = False`` so an application with
    its own root handler does not see every framework line twice.  All of that
    is right for an application and wrong for a test suite, because the state
    outlives the test that created it.

    The failure it causes is nastier than it sounds.  ``propagate = False``
    stops ``caplog`` from seeing framework log records at all, so a test
    asserting that something was logged passes in isolation and fails only
    when it runs after a test that configured logging.  Restoring the logger
    here keeps that ordering dependency out of the suite entirely.

    Yields
    ------
    None
        The test runs with the framework logger restored afterwards.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    level, propagate = logger.level, logger.propagate
    handlers = list(logger.handlers)
    try:
        yield
    finally:
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers


@pytest.fixture(scope="session")
def repository_root() -> Path:
    """
    Return the absolute path to the repository root.

    Returns
    -------
    Path
        Directory containing ``src`` and ``tests``.
    """
    return _REPOSITORY_ROOT


@pytest.fixture(scope="session")
def package_root() -> Path:
    """
    Return the absolute path to the ``rade_qnet`` source package.

    Returns
    -------
    Path
        The ``src/rade_qnet`` directory.
    """
    return _REPOSITORY_ROOT / "src" / "rade_qnet"


@pytest.fixture(scope="session")
def test_root() -> Path:
    """
    Return the absolute path to the ``rade_qnet`` test package.

    Returns
    -------
    Path
        The ``tests/rade_qnet`` directory.
    """
    return _REPOSITORY_ROOT / "tests" / "rade_qnet"


@pytest.fixture
def run_directory(tmp_path: Path) -> Iterator[Path]:
    """
    Provide an empty directory standing in for a run's working directory.

    Pipelines write bundles, reports and manifests beneath a run directory.
    Tests are given a fresh temporary one per test so that no test can observe
    another's artifacts, and so a failing test leaves its output behind for
    inspection rather than overwriting a shared location.

    Parameters
    ----------
    tmp_path
        Pytest's per-test temporary directory.

    Yields
    ------
    Path
        An existing, empty directory.
    """
    directory = tmp_path / "run"
    directory.mkdir()
    yield directory
