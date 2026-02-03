"""
Pytest configuration for m_learning tests.

Marks and fixtures for ML unit tests. TensorFlow is required for most tests;
if TF fails to load (e.g. ABI mismatch), collection may abort. Run with
--ignore=src/m_learning for TF-free tests, or ensure a compatible TF is installed.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_tensorflow: mark test as requiring TensorFlow (default: all m_learning tests)",
    )
