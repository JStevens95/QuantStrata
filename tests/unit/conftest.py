"""
Pytest configuration for unit tests.

Registers optional-suite markers so you can skip q_learning and machine_learning:

    pytest tests/unit/ -m "not q_learning and not machine_learning"
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "q_learning: optional q_learning suite (API may lag; skip with -m 'not q_learning').",
    )
    config.addinivalue_line(
        "markers",
        "machine_learning: optional machine_learning suite (requires TensorFlow; skip with -m 'not machine_learning').",
    )
