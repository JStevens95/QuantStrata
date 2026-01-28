"""
Performance Optimization Module.

This module provides high-performance backends for numerical computations:
- Numba JIT-compiled Monte Carlo path generation
- Numba JIT-compiled payoff evaluation
- Numba JIT-compiled tridiagonal solvers

The module automatically detects available backends and provides
a unified API for backend selection.

Usage
-----
>>> from src.core.performance import get_backend, Backend
>>> backend = get_backend("auto")  # Returns best available
>>> print(f"Using backend: {backend}")

Author: QuantStrata Team
"""

from src.core.performance.backend import (
    Backend,
    BackendConfig,
    get_backend,
    set_default_backend,
    numba_available,
    get_numba_version,
)

__all__ = [
    "Backend",
    "BackendConfig",
    "get_backend",
    "set_default_backend",
    "numba_available",
    "get_numba_version",
]
