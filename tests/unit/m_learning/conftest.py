"""
Pytest configuration for m_learning tests.

Requires Python 3.12+ and TensorFlow >= 2.20. Run with the same interpreter
and environment where TensorFlow is installed (e.g. venv with requirements.txt).
If the wrong Python is used (e.g. 3.9), TensorFlow can abort on import.
"""

import sys

import pytest

# Require Python 3.12+ before any m_learning imports (which pull in TensorFlow)
if sys.version_info < (3, 12):
    pytest.exit(
        "m_learning tests require Python 3.12+. "
        f"Current: {sys.version_info.major}.{sys.version_info.minor}. "
        "Use the correct interpreter (e.g. python3.12 -m pytest).",
        returncode=pytest.ExitCode.USAGE,
    )


def pytest_configure(config):
    """Register custom markers and verify TensorFlow is importable."""
    config.addinivalue_line(
        "markers",
        "requires_tensorflow: mark test as requiring TensorFlow (default: all m_learning tests)",
    )
    # Fail fast if TensorFlow is not available (avoid abort later when tests import m_learning)
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        pytest.exit(
            "m_learning tests require TensorFlow (e.g. pip install 'tensorflow>=2.20'). "
            f"Import failed: {e}",
            returncode=pytest.ExitCode.USAGE,
        )
