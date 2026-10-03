"""
FX instrument definitions - elementary trades for FX risk factor replication.

Instrument types:
    - FxVanillaOption - European vanilla call/put (EC/EP)
    - FxDigitalOption - European digital/binary call/put (DC/DP)
    - FxQuantoOption - European quanto call/put (QC/QP)

Each instrument:
    - carries all pricing inputs as typed fields (no Dict[str, Any] parameters)
    - implements to_trade_id() using the canonical convention, from make_trade_id()
    - implements to_elementary_trade() to produce the pipeline handoff object.
    - delegates pricing to @njit kernels in pricing/kernels/fx.py

generates_fx_elementary_trades() creates the full trade universe for one FX risk factor from strike and tenor grids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.static_replication.instruments.base import InstrumentSpec, StrikeGrid, TenorGrid
from src.static_replication.core.types import ElementaryTrade, make_trade_id


# --------------------------------------------------------------------------------------
# FX Vanilla Option
# --------------------------------------------------------------------------------------

@dataclass
class FxVanillaOption(InstrumentSpec):
    """
    European FX vanilla call or put (Garman-Kohlhagen pricing).

    payoff_type: 'EC' or 'EP'
    """

    # define parameters
    spot: float = 0.0
    strike: float = 0.0
    maturity: float = 0.0
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    vol: float = 0.0
    notional: float = 1.0
    is_call: bool = field(default=True)

    def to_pricer_params(self) -> Dict[str, Any]:
        """Return the flat numeric parameter dict consumed by the FX vanilla pricer."""
        return {
            "s": self.spot,
            "k": self.strike,
            "t": self.maturity,
            "r_dom": self.domestic_rate,
            "r_for": self.foreign_rate,
            "sigma": self.vol,
            "notional": self.notional,
            "is_call": self.is_call,
        }

    def to_trade_id(self) -> str:
        """Return the FX vanilla trade id."""
        return make_trade_id(self.factor_id, self.payoff_type, self.spot, self.strike, self.maturity)

    def to_elementary_trade(self) -> ElementaryTrade:
        """Convert this option spec into an ``ElementaryTrade`` for the pipeline."""
        return ElementaryTrade(
            trade_id=self.to_trade_id(),
            asset_class=self.asset_class,
            payoff_type=self.payoff_type,
            factor_id=self.factor_id,
            notional=self.notional,
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            domestic_rate=self.domestic_rate,
            foreign_rate=self.foreign_rate,
            vol=self.vol,
        )


# --------------------------------------------------------------------------------------
# FX Digital Option
# --------------------------------------------------------------------------------------

@dataclass
class FxDigitalOption(InstrumentSpec):
    """
    European FX digital (cash-or-nothing) call or put.

    payoff_type: 'DC' or 'DP'
    Pays notional if S_T > K (call) or S_T < K (put) at expiry.
    """

    # define parameters
    spot: float = 0.0
    strike: float = 0.0
    maturity: float = 0.0
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    vol: float = 0.0
    notional: float = 1.0
    is_call: bool = field(default=True)

    def to_pricer_params(self) -> Dict[str, Any]:
        """Return the flat numeric parameter dict consumed by the FX digital pricer."""
        return {
            "s": self.spot,
            "k": self.strike,
            "t": self.maturity,
            "r_dom": self.domestic_rate,
            "r_for": self.foreign_rate,
            "sigma": self.vol,
            "notional": self.notional,
            "is_call": self.is_call,
        }

    def to_trade_id(self) -> str:
        """Return the FX digital trade id."""
        return make_trade_id(self.factor_id, self.payoff_type, self.spot, self.strike, self.maturity)

    def to_elementary_trade(self) -> ElementaryTrade:
        """Convert this option spec into an ``ElementaryTrade`` for the pipeline."""
        return ElementaryTrade(
            trade_id=self.to_trade_id(),
            asset_class=self.asset_class,
            payoff_type=self.payoff_type,
            factor_id=self.factor_id,
            notional=self.notional,
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            domestic_rate=self.domestic_rate,
            foreign_rate=self.foreign_rate,
            vol=self.vol,
        )


# --------------------------------------------------------------------------------------
# FX Quanto Option
# --------------------------------------------------------------------------------------

@dataclass
class FxQuantoOption(InstrumentSpec):
    """
    European FX quanto call or put.

    payoff_type: 'QC' or 'QP'
    quanto_factor encodes the pre-computed quanto adjustment to the foreign rate.
    """

    # define parameters
    spot: float = 0.0
    strike: float = 0.0
    maturity: float = 0.0
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    vol: float = 0.0
    notional: float = 1.0
    quanto_factor: float = 1.0
    is_call: bool = field(default=True)

    def to_pricer_params(self) -> Dict[str, Any]:
        """Return the flat numeric parameter dict consumed by the FX quanto pricer."""
        return {
            "s": self.spot,
            "k": self.strike,
            "t": self.maturity,
            "r_dom": self.domestic_rate,
            "r_for": self.foreign_rate,
            "sigma": self.vol,
            "quanto_factor": self.quanto_factor,
            "notional": self.notional,
            "is_call": self.is_call,
        }

    def to_trade_id(self) -> str:
        """Return the FX quanto trade id."""
        return make_trade_id(self.factor_id, self.payoff_type, self.spot, self.strike, self.maturity)

    def to_elementary_trade(self) -> ElementaryTrade:
        """Convert this option spec into an ``ElementaryTrade`` for the pipeline."""
        return ElementaryTrade(
            trade_id=self.to_trade_id(),
            asset_class=self.asset_class,
            payoff_type=self.payoff_type,
            factor_id=self.factor_id,
            notional=self.notional,
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            domestic_rate=self.domestic_rate,
            foreign_rate=self.foreign_rate,
            vol=self.vol,
        )


# --------------------------------------------------------------------------------------
# Trade Universe Builder
# --------------------------------------------------------------------------------------

def generate_fx_elementary_trades(
        factor_id: str, spot: float, domestic_rate: float, foreign_rate: float, vol: float, strike_grid: StrikeGrid,
        tenor_grid: TenorGrid, payoff_type: List[str] = None, notional: float = 1.0, vol_surface: Optional[Any] = None
) -> List[ElementaryTrade]:
    """
    Generate the full elementary trade universe for one FX risk factor.

    Creates one trade per (payoff_type, tenor, strike) combination.
    Each trade's ``vol`` is populated by querying ``vol_surface`` at the trade's (tenor, strike/spot moneyness) if a
    surface is provided; otherwise the scaler ``vol`` fallback us used for all trades.
    """
    if payoff_type is None:
        payoff_type = ["EC", "EP", "DC", "DP", "QC", "QP"]

    call_types = {"EC", "QC", "DC"}
    trades: List[ElementaryTrade] = []

    for payoff in payoff_type:
        is_call = payoff in call_types

        for tenor in tenor_grid.values:
            for raw_strike in strike_grid.values:
                # resolve strike from convention
                if strike_grid.conventions == "relative":
                    strike = spot * raw_strike
                elif strike_grid.conventions == "delta":
                    # placeholder for delta space.
                    strike = raw_strike
                else:
                    strike = raw_strike

                # per-trade vol: interpolate from vol_surface if available.
                if vol_surface is not None:
                    trade_vol = vol_surface.vol_at_strike(tenor=tenor, strike=strike, spot=spot)
                else:
                    trade_vol = vol

                # build fx elementary instrument per payoff type
                if payoff in ("EC", "EP"):
                    instr = FxVanillaOption(
                        asset_class="FX", payoff_type=payoff, factor_id=factor_id, product_type="FX_OPTION",
                        product_subtype="VANILLA", spot=spot, strike=strike, maturity=tenor,
                        domestic_rate=domestic_rate, foreign_rate=foreign_rate, vol=trade_vol,
                        notional=notional, is_call=is_call
                    )
                elif payoff in ("DC", "DP"):
                    instr = FxDigitalOption(
                        asset_class="FX", payoff_type=payoff, factor_id=factor_id, product_type="FX_OPTION",
                        product_subtype="DIGITAL", spot=spot, strike=strike, maturity=tenor,
                        domestic_rate=domestic_rate, foreign_rate=foreign_rate, vol=trade_vol,
                        notional=notional, is_call=is_call
                    )
                elif payoff in ("QC", "QP"):
                    instr = FxQuantoOption(
                        asset_class="FX", payoff_type=payoff, factor_id=factor_id, product_type="FX_OPTION",
                        product_subtype="DIGITAL", spot=spot, strike=strike, maturity=tenor,
                        domestic_rate=domestic_rate, foreign_rate=foreign_rate, vol=trade_vol,
                        notional=notional, is_call=is_call
                    )
                else:
                    continue

                # append instruments to elementary trades.
                trades.append(instr.to_elementary_trade())
    return trades
