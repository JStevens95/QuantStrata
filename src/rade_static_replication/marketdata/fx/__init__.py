"""FX market-data objects: instruments (Spot, VolSurface) + snapshot + scenario set."""
from src.rade_static_replication.marketdata.fx.instruments import Spot, VolSurface
from src.rade_static_replication.marketdata.fx.scenarios import FXScenarioSet
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot

__all__ = ["Spot", "VolSurface", "FXSnapshot", "FXScenarioSet"]
