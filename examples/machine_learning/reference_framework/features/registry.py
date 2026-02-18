"""
Feature registry: maps feature/transform names to compute functions.

Enables pluggable transforms—training and inference look up the same
transform by id (e.g. "zscore", "onehot") and apply it consistently.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Type alias: transform takes (data, **kwargs) -> transformed_data
TransformFn = Callable[..., Any]


class FeatureRegistry:
    """
    Registry of transform functions by identifier.

    Use case: schema says transform_id="zscore" → registry.get("zscore")
    returns the Standardiser or zscore function. Both training and inference
    use the same registry, ensuring identical feature computation.
    """

    def __init__(self) -> None:
        self._transforms: Dict[str, TransformFn] = {}
        self._default_registered: bool = False

    def register(self, transform_id: str, fn: TransformFn) -> None:
        """
        Register a transform function.

        Parameters
        ----------
        transform_id : str
            Identifier (e.g. "zscore", "minmax", "onehot").
        fn : callable
            Function with signature (data, **kwargs) -> transformed.
        """
        if transform_id in self._transforms:
            logger.warning("Overwriting transform '%s'", transform_id)
        self._transforms[transform_id] = fn

    def get(self, transform_id: str) -> Optional[TransformFn]:
        """
        Retrieve transform by id.

        Returns
        -------
        callable or None
        """
        return self._transforms.get(transform_id)

    def get_or_raise(self, transform_id: str) -> TransformFn:
        """
        Retrieve transform; raise KeyError if not found.
        """
        fn = self._transforms.get(transform_id)
        if fn is None:
            raise KeyError(f"Transform '{transform_id}' not registered")
        return fn

    def register_defaults(self) -> None:
        """
        Register built-in transforms: none, zscore, minmax.
        Standardiser instances are created per-call via factory.
        """
        if self._default_registered:
            return

        def none_fn(data: Any, **kwargs: Any) -> Any:
            """Identity transform."""
            return data

        # Lazy import to avoid circular deps
        from reference_framework.features.transforms.standardiser import Standardiser

        def zscore_fn(data: Any, stats: Optional[Dict] = None, fit: bool = False, **kwargs: Any) -> Any:
            """
            Z-score transform. If stats provided, use them; else fit on data.
            """
            s = Standardiser(method="zscore")
            if stats is not None:
                s.from_dict(stats)
                return s.transform(data)
            if fit:
                s.fit(data)
                return s.transform(data), s.to_dict()
            raise ValueError("zscore requires stats= or fit=True")

        def minmax_fn(data: Any, stats: Optional[Dict] = None, fit: bool = False, **kwargs: Any) -> Any:
            """Min-max transform."""
            s = Standardiser(method="minmax")
            if stats is not None:
                s.from_dict(stats)
                return s.transform(data)
            if fit:
                s.fit(data)
                return s.transform(data), s.to_dict()
            raise ValueError("minmax requires stats= or fit=True")

        self.register("none", none_fn)
        self.register("zscore", zscore_fn)
        self.register("minmax", minmax_fn)
        self._default_registered = True

    def list_registered(self) -> List[str]:
        """Return list of registered transform ids."""
        return list(self._transforms.keys())


# Module-level singleton for convenience
_REGISTRY: Optional[FeatureRegistry] = None


def get_registry() -> FeatureRegistry:
    """Get or create the global feature registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FeatureRegistry()
        _REGISTRY.register_defaults()
    return _REGISTRY
