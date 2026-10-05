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

#: The captured baseline the parity tests compare against.
_GOLDEN_ROOT = _REPOSITORY_ROOT / "tests" / "fixtures" / "rade_qnet" / "golden"

#: Artifacts whose presence stands for the whole capture.
#:
#: One from each half of it, chosen to catch the two ways the fixture goes
#: missing.  ``manifest.json`` is absent when nothing was ever captured.
#: ``elementary_pnl.npy`` is absent when only the text files arrived -- which
#: happens when the tree crossed a markdown-only proxy, since the arrays
#: cannot travel that way and the JSON beside them can.
#:
#: A presence check, not an integrity check: the manifest's digests are what
#: prove a fixture is the one it claims to be.
_GOLDEN_ARTIFACTS = (
    _GOLDEN_ROOT / "hybrid_gnn_rnn" / "manifest.json",
    _GOLDEN_ROOT / "hybrid_gnn_rnn" / "input" / "elementary_pnl.npy",
)


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
def requires_golden() -> None:
    """
    Skip the requesting tests when the captured baseline is not present.

    Requested with ``@pytest.mark.usefixtures("requires_golden")`` on a test
    class, rather than by a module-level marker. The distinction matters:
    the classes that replay the baseline sit in files alongside classes that
    do not, and in one case the ratio is two tests to nineteen. Skipping a
    whole module to protect two tests would quietly stop running seventeen
    that work perfectly well without any fixture at all.

    Why skip rather than fail. The capture is a one-off step that a fresh
    clone has not run, and the arrays cannot cross a markdown-only proxy, so
    their absence is an ordinary condition rather than a defect. A suite that
    reports red for a missing optional input teaches people to ignore red.

    What is lost when this skips is worth stating plainly: these are the
    tests that compare this implementation's numbers against the original's.
    Everything else still runs, so the suite proves the framework behaves --
    it just stops proving it produces the same figures as the baseline.

    This also gates the tests that check the capture is *complete*, which
    reads like a contradiction and is not. The two answer different
    questions, and the order matters. This fixture asks whether a capture is
    present at all; the completeness tests then ask whether the one that is
    present has every artifact it should. Gating them means a half-copied
    tree -- the text files without the arrays -- is reported as "no fixture"
    rather than as a dozen failures, while a genuine capture missing one
    file still fails loudly, which is what those tests are for.

    Raises
    ------
    Skipped
        If any artifact in :data:`_GOLDEN_ARTIFACTS` is missing, naming the
        first one and how to produce it.
    """
    for artifact in _GOLDEN_ARTIFACTS:
        if not artifact.is_file():
            pytest.skip(
                f"golden fixture not captured: {artifact} is missing. "
                f"Capture it with examples/rade_qnet/phase0_capture_baseline.py, "
                f"and note that a partial copy is worse than none -- the text "
                f"files alone stop these tests skipping without letting them pass"
            )


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
