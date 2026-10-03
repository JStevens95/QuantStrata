"""
FX asset-class plugin.

Edit ``builder.py`` (market data), ``instruments.py``/``generator.py`` (replicating
basis), or ``pricer.py`` (models) in isolation. The bundle is exported as
:data:`FX_PLUGIN`.
"""
from src.rade_static_replication.assets.base import AssetClassPlugin
from src.rade_static_replication.assets.fx.builder import FXRiskFactorBuilder
from src.rade_static_replication.assets.fx.generator import FXElementaryGenerator
from src.rade_static_replication.assets.fx.pricer import FXPricer

FX_PLUGIN = AssetClassPlugin(
    name="fx",
    builder=FXRiskFactorBuilder(),
    generator=FXElementaryGenerator(),
    pricer=FXPricer(),
)

__all__ = ["FX_PLUGIN", "FXRiskFactorBuilder", "FXElementaryGenerator", "FXPricer"]
