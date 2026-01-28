"""
Tests for backend configuration and detection.

Author: QuantStrata Team
"""
import pytest
import numpy as np

from src.core.performance.backend import (
    Backend,
    BackendConfig,
    get_backend,
    set_default_backend,
    numba_available,
    get_numba_version,
    get_backend_info,
    get_config,
)


class TestBackendDetection:
    """Tests for backend detection."""
    
    def test_numba_available_returns_bool(self):
        """numba_available() should return boolean."""
        result = numba_available()
        assert isinstance(result, bool)
    
    def test_numba_version_consistent(self):
        """Version should be string if available, None otherwise."""
        version = get_numba_version()
        if numba_available():
            assert isinstance(version, str)
            assert len(version) > 0
        else:
            assert version is None
    
    def test_backend_info_structure(self):
        """get_backend_info() should return proper structure."""
        info = get_backend_info()
        
        assert "numpy" in info
        assert "numba" in info
        assert "default" in info
        assert "selected" in info
        
        assert info["numpy"]["available"] is True
        assert isinstance(info["numpy"]["version"], str)


class TestBackendSelection:
    """Tests for backend selection."""
    
    def test_numpy_backend_always_available(self):
        """NumPy backend should always be selectable."""
        backend = get_backend("numpy")
        assert backend == Backend.NUMPY
    
    def test_auto_backend_returns_valid(self):
        """Auto should return either NUMPY or NUMBA."""
        backend = get_backend("auto")
        assert backend in (Backend.NUMPY, Backend.NUMBA)
    
    def test_numba_backend_fallback(self):
        """Numba request should fall back to NumPy if unavailable."""
        backend = get_backend("numba")
        if numba_available():
            assert backend == Backend.NUMBA
        else:
            assert backend == Backend.NUMPY
    
    def test_invalid_backend_raises(self):
        """Invalid backend should raise ValueError."""
        with pytest.raises(ValueError):
            get_backend("invalid_backend")


class TestBackendConfig:
    """Tests for backend configuration."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        config = get_config()
        assert config.default_backend in ("numpy", "numba", "auto")
        assert isinstance(config.parallel, bool)
        assert isinstance(config.cache, bool)
    
    def test_set_default_backend(self):
        """Should be able to change default backend."""
        original = get_config().default_backend
        
        try:
            set_default_backend("numpy")
            assert get_config().default_backend == "numpy"
            
            set_default_backend("auto")
            assert get_config().default_backend == "auto"
        finally:
            # Restore original
            set_default_backend(original)
    
    def test_invalid_default_raises(self):
        """Invalid default backend should raise."""
        with pytest.raises(ValueError):
            set_default_backend("invalid")


class TestBackendEnum:
    """Tests for Backend enum."""
    
    def test_enum_values(self):
        """Enum should have expected values."""
        assert Backend.NUMPY.value == "numpy"
        assert Backend.NUMBA.value == "numba"
        assert Backend.AUTO.value == "auto"
    
    def test_enum_comparison(self):
        """Enum should be comparable to strings."""
        assert Backend.NUMPY == Backend.NUMPY
        assert Backend.NUMPY != Backend.NUMBA
