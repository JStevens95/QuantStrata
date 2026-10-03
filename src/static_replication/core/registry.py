"""
RiskFactorRegistry - instance scoped registry mapping asset_class --> RiskFactorBuilder.

Design:
    - One instance per pipeline run. No class-level shared state.
    - Thread-safe via threading
    - Fresh instance per test run guarantees isolation between tests.

Usage:
    registry = RiskFactorRegistry()
    registry.register(FxRiskFactorBuilder())
    registry.register(IrRiskFactorBuilder())

    builder = register.get("FX")
    asset = builde.build("FX_SPOT.USD.GBP", factor_config)
"""
from __future__ import annotations

import logging
import threading

from typing import Dict, List

from src.static_replication.core.exceptions import RegistryError, UnknownAssetClassError

# define module level logging.
logger = logging.getLogger(__name__)


class RiskFactorRegistry:
    """
    Instance-scoped registry: asset_class -> RiskFactorBuilder.

    Thread-safe - safe to call register() from multiple threads simultaneously is used for IO-bound asset loading stages.
    """

    def __init__(self) -> None:

        # initiate required variables.
        self._builders: Dict[str, object] = {}
        self._lock = threading.Lock()

    def register(self, builder: object) -> None:
        """
        Register a builder instance.

        :param builder:
        :return:
        """
        asset_class = getattr(builder, "asset_class", None)
        if not asset_class:
            raise RegistryError(f"Builder {builder!r} has no asset_class attribute.")
        with self._lock:
            if asset_class in self._builders:
                logger.warning("Overwriting existing builder for asset_class %r", asset_class)
            self._builders[asset_class] = builder
            logger.debug("Registeered builder for asset_class %r", asset_class)

    def get(self, asset_class: str) -> object:
        """Return the builder registered for asset_class."""
        builder = self._builders.get(asset_class)
        if builder is None:
            raise UnknownAssetClassError(f"No builder registered for asset class {asset_class!r}")
        return builder

    def all(self) -> Dict[str, object]:
        """Return a snapshot copy of all registered builders keyed by asset_class."""
        with self._lock:
            return dict(self._builders)

    def registered_classes(self) -> List[str]:
        """Return sorted list of all registered asset class keys."""
        return sorted(self._builders.keys())

    def reset(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._builders.clear()
            logger.debug("RiskFactorRegistry cleared.")

    def __len__(self) -> int:
        return len(self._builders)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        return f"RiskFactorRegistry(registered={self.registered_classes()})"
