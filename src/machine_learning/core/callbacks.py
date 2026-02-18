"""
Backward-compatibility shim — callbacks have moved to training/callbacks.py.

All symbols are re-exported so existing ``from src.machine_learning.core.callbacks
import ...`` imports continue to work.  New code should import from
``src.machine_learning.training.callbacks`` directly.

.. deprecated::
    This shim will be removed in a future version.  Update imports to::

        from src.machine_learning.training.callbacks import (
            MetricsLogger,
            PricingErrorCallback,
            get_standard_callbacks,
        )
"""
import warnings

warnings.warn(
    "Importing callbacks from 'src.machine_learning.core.callbacks' is deprecated. "
    "Use 'src.machine_learning.training.callbacks' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.machine_learning.training.callbacks import (  # noqa: F401
    MetricsLogger,
    PricingErrorCallback,
    get_standard_callbacks,
)

__all__ = [
    "MetricsLogger",
    "PricingErrorCallback",
    "get_standard_callbacks",
]
