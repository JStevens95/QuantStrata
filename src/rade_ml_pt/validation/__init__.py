"""
Validation: input checks and custom exceptions.
"""
from src.rade_ml_pt.validation.base import validate_dict_keys
from src.rade_ml_pt.validation.exceptions import (
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

__all__ = [
    "validate_dict_keys",
    "CacheLoaderError",
    "UnsupportedFileTypeError",
    "FileLoadError",
    "FileSaveError",
    "MissingKeyFields",
    "UndefinedModelArchitecture",
    "UndefinedVariableType",
    "UndefinedTransformerType",
    "HybridModelNotAvailable",
    "UndefinedLayerType",
    "UndefinedReductionType",
    "UndefinedComputationMethod",
]
