"""trade - elementary trade generation for FX and IR risk factors."""
from src.static_replication.trade.generator import FxTradeGenerator, IrTradeGenerator, TradeGenerator

__all__ = [
    "FxTradeGenerator", "IrTradeGenerator", "TradeGenerator",
]
