"""Unit tests for rade_ml.validation.exceptions."""
import pytest

from src.rade_ml.validation.exceptions import (
    CacheLoaderError,
    UnsupportedFileTypeError,
    FileLoadError,
    FileSaveError,
    MissingKeyFields,
    UndefinedModelArchitecture,
    UndefinedVariableType,
    UndefinedTransformerType,
    HybridModelNotAvailable,
    UndefinedLayerType,
    UndefinedReductionType,
    UndefinedComputationMethod,
)


class TestExceptionHierarchy:
    def test_cache_loader_base(self):
        assert issubclass(UnsupportedFileTypeError, CacheLoaderError)
        assert issubclass(FileLoadError, CacheLoaderError)
        assert issubclass(FileSaveError, CacheLoaderError)

    @pytest.mark.parametrize("exc_cls", [
        MissingKeyFields,
        UndefinedModelArchitecture,
        UndefinedVariableType,
        UndefinedTransformerType,
        HybridModelNotAvailable,
        UndefinedLayerType,
        UndefinedReductionType,
        UndefinedComputationMethod,
    ])
    def test_all_exceptions_are_exception_subclass(self, exc_cls):
        assert issubclass(exc_cls, Exception)

    @pytest.mark.parametrize("exc_cls", [
        MissingKeyFields,
        UndefinedModelArchitecture,
        UndefinedTransformerType,
        UndefinedLayerType,
    ])
    def test_exceptions_carry_message(self, exc_cls):
        e = exc_cls("test message")
        assert "test message" in str(e)
