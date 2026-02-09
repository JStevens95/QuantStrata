"""
Mark all tests in this directory as machine_learning so they can be skipped with -m 'not machine_learning'.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Add machine_learning marker to all tests in this directory."""
    for item in items:
        item.add_marker(pytest.mark.machine_learning)
