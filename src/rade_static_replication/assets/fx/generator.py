"""FX elementary-trade generator — options/forwards on a forward × moneyness grid."""
from __future__ import annotations

from typing import Dict, List

from src.rade_static_replication.assets.fx.instruments import DEFAULT_GRID
from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot


class FXElementaryGenerator:
    """Place vanilla/digital options and forwards on a (forward × moneyness) grid.

    For each expiry we anchor strikes to the outright forward (``strike = moneyness ×
    forward``) so the grid is self-consistent across rate environments. A single
    ATM-forward is emitted per expiry.
    """

    def generate(self, rf: RiskFactorData, grid: Dict) -> List[ElementaryTrade]:
        snapshot: FXSnapshot = rf.snapshot  # type: ignore[assignment]
        moneyness_grid = grid.get("moneyness", DEFAULT_GRID["moneyness"])
        expiry_grid = grid.get("expiries", DEFAULT_GRID["expiries"])
        payoff_types = grid.get("instruments", DEFAULT_GRID["instruments"])
        factor_id = rf.factor_id

        trades: List[ElementaryTrade] = []
        for expiry in expiry_grid:
            forward = snapshot.forward(float(expiry))
            for moneyness in moneyness_grid:
                strike = forward * float(moneyness)
                params = {"strike": strike, "expiry": float(expiry), "moneyness": float(moneyness)}
                for payoff in payoff_types:
                    # The forward only makes sense at-the-money-forward; skip off-ATM forwards.
                    if payoff == "forward" and abs(float(moneyness) - 1.0) > 1e-9:
                        continue
                    trade_id = f"{factor_id}|{payoff.upper()}|K={strike:.6g}|T={float(expiry):.4g}"
                    trades.append(ElementaryTrade(
                        trade_id=trade_id, factor_id=factor_id, asset_class="fx",
                        payoff_type=payoff, parameters=dict(params), notional=1.0,
                    ))
        return trades
