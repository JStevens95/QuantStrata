"""Rates elementary-trade generator — payer/receiver swaptions on an expiry × tenor × strike grid."""
from __future__ import annotations

from typing import Dict, List

from src.rade_static_replication.assets.rates.instruments import DEFAULT_GRID
from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot
from src.rade_static_replication.pricing.kernels._math import compute_forward_swap_rate


class RatesElementaryGenerator:
    """Emit swaptions around the ATM forward swap rate for each (expiry, swap-tenor).

    Strikes are placed at basis-point offsets from the ATM forward swap rate so the grid
    straddles the money for every (expiry, tenor) point regardless of curve level.
    """

    def generate(self, rf: RiskFactorData, grid: Dict) -> List[ElementaryTrade]:
        snapshot: RatesSnapshot = rf.snapshot  # type: ignore[assignment]
        curve_tenors = snapshot.discount_curve.tenors
        curve_zeros = snapshot.discount_curve.zero_rates
        expiry_grid = grid.get("expiries", DEFAULT_GRID["expiries"])
        swap_tenor_grid = grid.get("swap_tenors", DEFAULT_GRID["swap_tenors"])
        strike_offsets_bp = grid.get("strike_offsets_bp", DEFAULT_GRID["strike_offsets_bp"])
        payoff_types = grid.get("instruments", DEFAULT_GRID["instruments"])
        pay_frequency = grid.get("freq", DEFAULT_GRID["freq"])
        factor_id = rf.factor_id

        trades: List[ElementaryTrade] = []
        for expiry in expiry_grid:
            for swap_tenor in swap_tenor_grid:
                atm_rate = compute_forward_swap_rate(
                    curve_tenors, curve_zeros, float(expiry), float(swap_tenor), float(pay_frequency),
                )
                for offset_bp in strike_offsets_bp:
                    strike = atm_rate + float(offset_bp) * 1e-4  # bp -> absolute rate
                    for payoff in payoff_types:
                        trade_id = (
                            f"{factor_id}|{payoff.upper()}"
                            f"|E={float(expiry):.4g}|Tn={float(swap_tenor):.4g}|K={strike:.6g}"
                        )
                        trades.append(ElementaryTrade(
                            trade_id=trade_id, factor_id=factor_id, asset_class="rates", payoff_type=payoff,
                            parameters={
                                "expiry": float(expiry), "swap_tenor": float(swap_tenor),
                                "strike": strike, "freq": float(pay_frequency),
                            },
                            notional=1.0,
                        ))
        return trades
