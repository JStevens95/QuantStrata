"""
Default plugin registry.

``default_registry()`` returns a :class:`Registry` pre-loaded with the built-in FX and
Rates plugins. To add an asset class: build its subpackage under ``assets/<class>/``
and register one line here (or call ``registry.register(...)`` at the call site).
"""
from __future__ import annotations

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.assets.fx import FX_PLUGIN
from src.rade_static_replication.assets.rates import RATES_PLUGIN


def default_registry() -> Registry:
    """A registry with the built-in asset classes installed."""
    return Registry().register(FX_PLUGIN).register(RATES_PLUGIN)
