"""
Backend Configuration and Detection.

This module provides:
- Backend detection (NumPy, Numba)
- Configuration management
- Unified backend selection API

Design Philosophy
-----------------
1. NumPy is always available (baseline, no external dependencies)
2. Numba is optional but provides significant speedups
3. Backend selection is explicit and configurable
4. Graceful fallback when preferred backend unavailable

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional
import warnings


# =============================================================================
# Backend Enumeration
# =============================================================================

class Backend(str, Enum):
    """
    Available computational backends.
    
    Attributes
    ----------
    NUMPY:
        Pure NumPy implementation (always available).
        - Pro: No external dependencies, debuggable
        - Con: Slower for large-scale computations
        
    NUMBA:
        Numba JIT-compiled implementation.
        - Pro: 10-100x speedup for numerical loops
        - Con: Requires Numba installation, compilation overhead
        
    AUTO:
        Automatically select best available backend.
        Priority: NUMBA > NUMPY
    """
    NUMPY = "numpy"
    NUMBA = "numba"
    AUTO = "auto"


# Type alias for backend specification
BackendStr = Literal["numpy", "numba", "auto"]


# =============================================================================
# Numba Detection
# =============================================================================

_NUMBA_AVAILABLE: Optional[bool] = None
_NUMBA_VERSION: Optional[str] = None


def numba_available() -> bool:
    """
    Check if Numba is available for import.
    
    Returns
    -------
    bool
        True if Numba is installed and importable.
        
    Notes
    -----
    Result is cached after first call for efficiency.
    """
    global _NUMBA_AVAILABLE, _NUMBA_VERSION
    
    if _NUMBA_AVAILABLE is None:
        try:
            import numba
            _NUMBA_AVAILABLE = True
            _NUMBA_VERSION = numba.__version__
        except ImportError:
            _NUMBA_AVAILABLE = False
            _NUMBA_VERSION = None
            
    return _NUMBA_AVAILABLE


def get_numba_version() -> Optional[str]:
    """
    Get the installed Numba version.
    
    Returns
    -------
    Optional[str]
        Version string if Numba is installed, None otherwise.
    """
    numba_available()  # Ensure detection has run
    return _NUMBA_VERSION


# =============================================================================
# Backend Configuration
# =============================================================================

@dataclass
class BackendConfig:
    """
    Global backend configuration.
    
    Attributes
    ----------
    default_backend:
        Default backend for new operations.
    warn_on_fallback:
        If True, warn when falling back from preferred backend.
    parallel:
        Enable parallel execution where supported.
    cache:
        Enable Numba function caching (reduces recompilation).
    fastmath:
        Enable fast math optimizations (may reduce precision).
    """
    default_backend: BackendStr = "auto"
    warn_on_fallback: bool = True
    parallel: bool = True
    cache: bool = True
    fastmath: bool = False


# Global configuration instance
_CONFIG = BackendConfig()


def get_config() -> BackendConfig:
    """Get the global backend configuration."""
    return _CONFIG


def set_config(config: BackendConfig) -> None:
    """Set the global backend configuration."""
    global _CONFIG
    _CONFIG = config


# =============================================================================
# Backend Selection
# =============================================================================

def get_backend(preferred: BackendStr = "auto") -> Backend:
    """
    Select the computational backend to use.
    
    Parameters
    ----------
    preferred:
        Preferred backend:
        - "numpy": Force NumPy backend
        - "numba": Prefer Numba (falls back to NumPy if unavailable)
        - "auto": Automatically select best available
        
    Returns
    -------
    Backend
        The selected backend (always NUMPY or NUMBA).
        
    Examples
    --------
    >>> backend = get_backend("auto")
    >>> if backend == Backend.NUMBA:
    ...     print("Using Numba for high performance")
    ... else:
    ...     print("Using NumPy (Numba not available)")
    """
    config = get_config()
    
    # Use global default if auto
    if preferred == "auto":
        preferred = config.default_backend
        
    # Still auto? Select best available
    if preferred == "auto":
        if numba_available():
            return Backend.NUMBA
        return Backend.NUMPY
        
    # Explicit NumPy request
    if preferred == "numpy":
        return Backend.NUMPY
        
    # Explicit Numba request
    if preferred == "numba":
        if numba_available():
            return Backend.NUMBA
        else:
            if config.warn_on_fallback:
                warnings.warn(
                    "Numba requested but not available. Falling back to NumPy. "
                    "Install Numba for better performance: pip install numba",
                    UserWarning,
                    stacklevel=2,
                )
            return Backend.NUMPY
            
    # Unknown backend
    raise ValueError(f"Unknown backend: {preferred!r}. Use 'numpy', 'numba', or 'auto'.")


def set_default_backend(backend: BackendStr) -> None:
    """
    Set the default backend for all operations.
    
    Parameters
    ----------
    backend:
        Backend to use as default ("numpy", "numba", or "auto").
        
    Examples
    --------
    >>> set_default_backend("numba")  # Prefer Numba everywhere
    >>> set_default_backend("numpy")  # Force NumPy everywhere
    >>> set_default_backend("auto")   # Auto-select (default)
    """
    config = get_config()
    if backend not in ("numpy", "numba", "auto"):
        raise ValueError(f"Unknown backend: {backend!r}")
    config.default_backend = backend


# =============================================================================
# Backend Information
# =============================================================================

def get_backend_info() -> dict:
    """
    Get information about available backends.
    
    Returns
    -------
    dict
        Dictionary with backend availability and versions.
        
    Examples
    --------
    >>> info = get_backend_info()
    >>> print(info)
    {'numpy': {'available': True, 'version': '1.24.0'},
     'numba': {'available': True, 'version': '0.58.0'},
     'default': 'auto'}
    """
    import numpy as np
    
    return {
        "numpy": {
            "available": True,
            "version": np.__version__,
        },
        "numba": {
            "available": numba_available(),
            "version": get_numba_version(),
        },
        "default": get_config().default_backend,
        "selected": get_backend("auto").value,
    }
