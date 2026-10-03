"""
Typed market-data layer.

Immutable, self-validating objects with explicit conventions — the single source of
truth for "what the market looked like" per risk factor. No nested dicts.

Layout mirrors ``assets/``:

* ``base`` / ``snapshot`` / ``scenarios`` — the shared spine + abstract bases;
* ``common/`` — the one cross-asset instrument (``DiscountCurve``);
* ``fx/`` and ``rates/`` — each asset class's own instruments + snapshot + scenario set
  (FX 2-D lognormal surface vs rates 3-D normal cube — different shapes/conventions);
* ``shocks`` — relative/absolute shock resolution.
"""
from src.rade_static_replication.marketdata.common import DiscountCurve
from src.rade_static_replication.marketdata.fx import FXScenarioSet, FXSnapshot, Spot, VolSurface
from src.rade_static_replication.marketdata.rates import (
    RatesScenarioSet,
    RatesSnapshot,
    VolCube,
)
from src.rade_static_replication.marketdata.scenarios import ScenarioSet
from src.rade_static_replication.marketdata.shocks import ShockConvention, apply_shock
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot

__all__ = [
    "DiscountCurve", "VolSurface", "VolCube", "Spot",
    "MarketSnapshot", "FXSnapshot", "RatesSnapshot",
    "ScenarioSet", "FXScenarioSet", "RatesScenarioSet",
    "ShockConvention", "apply_shock",
]
