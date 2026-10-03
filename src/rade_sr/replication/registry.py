"""
Risk factor registry — decorator-based asset class dispatch.

Maps ``asset_class: str → RiskFactorBuilder`` instance.  Registration
happens via ``@RiskFactorRegistry.register`` at import time.

Importing ``builders.py`` is the registration act.
The pipeline never needs to know which asset classes exist — it calls
``registry.get(asset_class)`` and dispatches blindly.

Adding a new asset class = one new decorated class in builders.py.
Nothing else changes.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Type

from src.rade_sr.core.protocols import RiskFactorBuilder
from src.rade_sr.core.exceptions import UnknownAssetClassError

logger = logging.getLogger(__name__)


class RiskFactorRegistry:
    """Maps asset_class string to RiskFactorBuilder instance.

    Usage
    -----
    ::

        @RiskFactorRegistry.register
        class FXRiskFactorBuilder:
            asset_class = "fx"
            def build(self, factor_id, factor_config) -> Any: ...

    Then elsewhere::

        builder = RiskFactorRegistry.get("fx")
        asset = builder.build("EURUSD", config)
    """

    _registry: Dict[str, RiskFactorBuilder] = {}

    @classmethod
    def register(cls, builder_cls: Type) -> Type:
        """Class decorator that registers a RiskFactorBuilder implementation.

        The class must define ``asset_class`` as a class-level string attribute.
        An instance is created and stored in the registry.

        Parameters
        ----------
        builder_cls : type
            The builder class to register.

        Returns
        -------
        type
            The same class (unmodified), allowing normal instantiation.

        Raises
        ------
        TypeError
            If the class doesn't satisfy the RiskFactorBuilder protocol.
        AttributeError
            If the class doesn't define ``asset_class``.
        """
        instance = builder_cls()
        if not isinstance(instance, RiskFactorBuilder):
            raise TypeError(
                f"{builder_cls.__name__} does not satisfy RiskFactorBuilder protocol."
            )
        key = getattr(builder_cls, "asset_class", None)
        if not key:
            raise AttributeError(
                f"{builder_cls.__name__} must define `asset_class` as a class attribute."
            )
        cls._registry[key] = instance
        logger.debug("Registered RiskFactorBuilder: %s -> %s", key, builder_cls.__name__)
        return builder_cls

    @classmethod
    def get(cls, asset_class: str) -> RiskFactorBuilder:
        """Look up the builder for a given asset class.

        Raises
        ------
        UnknownAssetClassError
            If no builder is registered for the requested asset class.
        """
        try:
            return cls._registry[asset_class]
        except KeyError:
            raise UnknownAssetClassError(
                f"No builder registered for asset_class={asset_class!r}. "
                f"Registered: {list(cls._registry)}"
            )

    @classmethod
    def registered(cls) -> List[str]:
        """List all registered asset class keys."""
        return list(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (primarily for testing)."""
        cls._registry.clear()
