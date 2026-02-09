"""
Mark all tests in this directory as q_learning so they can be skipped with -m 'not q_learning'.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Add q_learning marker to all tests in this directory."""
    for item in items:
        item.add_marker(pytest.mark.q_learning)
