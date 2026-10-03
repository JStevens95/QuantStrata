"""
Rates asset-class plugin.

Edit ``builder.py`` (curve/cube), ``instruments.py``/``generator.py`` (swaption grid),
or ``pricer.py`` (Bachelier) in isolation. Exported as :data:`RATES_PLUGIN`.
"""
from src.rade_static_replication.assets.base import AssetClassPlugin
from src.rade_static_replication.assets.rates.builder import RatesRiskFactorBuilder
from src.rade_static_replication.assets.rates.generator import RatesElementaryGenerator
from src.rade_static_replication.assets.rates.pricer import RatesPricer

RATES_PLUGIN = AssetClassPlugin(
    name="rates",
    builder=RatesRiskFactorBuilder(),
    generator=RatesElementaryGenerator(),
    pricer=RatesPricer(),
)

__all__ = ["RATES_PLUGIN", "RatesRiskFactorBuilder", "RatesElementaryGenerator", "RatesPricer"]
