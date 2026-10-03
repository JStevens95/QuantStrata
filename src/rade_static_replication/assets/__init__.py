"""
Asset-class plugins — the extension seam.

Each asset class is an isolated subpackage (``fx/``, ``rates/``) bundling a builder,
generator, and pricer behind the protocols in :mod:`assets.base`. :func:`default_registry`
wires the built-ins.
"""
from src.rade_static_replication.assets.base import (
    AssetClassPlugin,
    ElementaryGenerator,
    Pricer,
    Registry,
    RiskFactorBuilder,
)
from src.rade_static_replication.assets.registry import default_registry

__all__ = [
    "AssetClassPlugin", "Registry", "default_registry",
    "RiskFactorBuilder", "ElementaryGenerator", "Pricer",
]
