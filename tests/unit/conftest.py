"""
Pytest configuration for unit tests.

- Registers optional-suite markers (q_learning, machine_learning).
- Forces matplotlib to Agg backend so unit tests never open plot windows.
"""

import pytest


def pytest_configure(config):
    """Register custom markers and disable matplotlib display in unit tests."""
    # Unit tests must not open plot windows; use non-interactive backend
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass

    config.addinivalue_line(
        "markers",
        "q_learning: optional q_learning suite (API may lag; skip with -m 'not q_learning').",
    )
    config.addinivalue_line(
        "markers",
        "machine_learning: optional machine_learning suite (requires TensorFlow; skip with -m 'not machine_learning').",
    )
