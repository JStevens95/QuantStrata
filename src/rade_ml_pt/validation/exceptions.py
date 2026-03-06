class CacheLoaderError(Exception):
    """Base exception for cache loader errors."""


class UnsupportedFileTypeError(CacheLoaderError):
    """Raised when attempting to handle an unsupported file type."""


class FileLoadError(CacheLoaderError):
    """Raised for file reading or parsing errors."""


class FileSaveError(CacheLoaderError):
    """Raised for file saving errors."""


class MissingKeyFields(Exception):
    """Raise exception for missing key fields error."""

    pass


class UndefinedModelArchitecture(Exception):
    """Raise exception for undefined model architecture."""

    pass


class UndefinedVariableType(Exception):
    """Raise exception for undefined variable type."""

    pass


class UndefinedTransformerType(Exception):
    """Raise exception for undefined transformer type."""

    pass


class HybridModelNotAvailable(Exception):
    """Raise exception for hybrid model not available."""

    pass


class UndefinedLayerType(Exception):
    """Raise exception for undefined layer type."""

    pass


class UndefinedReductionType(Exception):
    """Raise exception for undefined dimensionality reduction type."""

    pass


class UndefinedComputationMethod(Exception):
    """Raise exception for undefined computation method."""

    pass
