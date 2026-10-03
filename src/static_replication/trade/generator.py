"""
Trade generator - builds List[ElementaryTrade] from loaded assets and trade config.

FxTradeGenerator - FX vanilla / digital / quanto options
IrTradeGenerator - IR swaps (PS, RS) and swpations (PSO/RSO, bachelier normal vol)
TradeGenerator - delegates to both generators for portfolio.
"""
from __future__ import annotations

import logging
import numpy as np

from typing import Any, Dict, List, Optional

from src.static_replication.core.types import ElementaryTrade
from src.static_replication.core.exceptions import TradeGenerationError
from src.static_replication.instruments.base import StrikeGrid, TenorGrid
from src.static_replication.instruments.fx import generate_fx_elementary_trades
from src.static_replication.instruments.rates import generate_ir_elementary_trades

# define module level logging.
logger = logging.getLogger(__name__)


class FxTradeGenerator:
    """Generate elementary FX trades from loaded FxAsset instances."""

    def generate(
            self, assets: Dict[str, Any], trade_config: Dict[str, Any]
    ) -> Dict[str, List[ElementaryTrade]]:
        """Generate elementary FX trades from loaded FxAsset instances."""

        # extract trade config variables.
        payoff_types = trade_config.get("payoff_types", ["EC", "EP"])
        strikes_values = trade_config.get("strikes_values", [])
        tenor_values = trade_config.get("tenor_values", [])
        strike_convention = trade_config.get("strike_convention", "relative")
        notional = float(trade_config.get("notional", 1.0))

        if not strikes_values:
            raise TradeGenerationError("trade_config missing 'strikes_values' parameter")
        if not tenor_values:
            raise TradeGenerationError("trade_config missing 'tenor_values' parameter")

        strike_grid = StrikeGrid(values=strikes_values, convention=strike_convention)
        tenor_grid = TenorGrid(values=tenor_values)
        result: Dict[str, List[ElementaryTrade]] = {}

        for factor_id, asset in assets.items():
            if not factor_id.startswith("FX"):
                continue

            try:
                spot = float(asset.spot) if asset.spot is not None else 0.0
                dom_ir = asset.dependencies.get("domestic_ir")
                for_ir = asset.dependencies.get("foreign_ir")
                domestic_rate = float(dom_ir.spot) if dom_ir and dom_ir.spot else 0.0
                foreign_rate = float(for_ir.spot) if for_ir.spot else 0.0

                # use vol surface if available to resolve vol per (tenor, strike).
                vol_surface = getattr(asset, "vol_surface", None)

                # generate fx elementary trades.
                trades = generate_fx_elementary_trades(
                    factor_id=factor_id, spot=spot, domestic_rate=domestic_rate, foreign_rate=foreign_rate,
                    vol_surface=vol_surface, strike_grid=strike_grid, tenor_grid=tenor_grid, payoff_type=payoff_types,
                    notional=notional
                )
                result[factor_id] = trades
                logger.info("FxTradeGenerator: generated %d trades for %s", len(trades), factor_id)
            except Exception as exc:
                raise TradeGenerationError(f"Trade generation failed for {factor_id}:{exc}") from exc
        return result


class IrTradeGenerator:
    """Generate elementary IR trades from loaded IrAsset instances."""

    def generate(self, assets: Dict[str, Any], trade_config: Dict[str, Any]) -> Dict[str, List[ElementaryTrade]]:
        """Generate elementary IR trades from loaded IrAsset instances."""

        # extract trade config variables.
        payoff_types = trade_config.get("payoff_types", ["PSO", "RSO"])
        strikes_values = trade_config.get("strikes_values", [0.0])
        tenor_values = trade_config.get("tenor_values", [])
        strike_convention = trade_config.get("strike_convention", "relative")
        swap_tenors: Optional[List[float]] = trade_config.get("swap_tenors", None)
        freq = float(trade_config.get("freq", 0.5))
        notional = float(trade_config.get("notional", 1.0))

        if not tenor_values:
            raise TradeGenerationError("trade_config missing 'tenor_values' parameter")

        strike_grid = StrikeGrid(values=strikes_values, convention=strike_convention)
        tenor_grid = TenorGrid(values=tenor_values)
        result: Dict[str, List[ElementaryTrade]] = {}

        for factor_id, asset in assets.items():
            if not factor_id.startswith("IR"):
                continue

            try:
                spot_rate = float(asset.spot) if asset.spot is not None else 0.0
                curve_tenors: np.ndarray = np.array([], dtype=float)
                curve_rates: np.ndarray = np.array([], dtype=float)
                if hasattr(asset, "curve_series") and assets.curve_series is not None:
                    curve_tenors = asset.curve_series.tenors.copy()
                    curve_rates = asset.curve_series.rates[-1].copy()

                vol = 0.0
                if hasattr(asset, "vol_surface") and assets.vol_surface is not None:
                    vol = asset.vol_surface

                trades = generate_ir_elementary_trades(
                    factor_id=factor_id, spot=spot_rate, curve_tenors=curve_tenors, curve_rates=curve_rates,
                    vol=vol, strike_grid=strike_grid, expiry_grid=tenor_grid, swap_tenors=swap_tenors,
                    payoff_type=payoff_types, freq=freq, notional=notional
                )
                result[factor_id] = trades
                logger.info("IrTradeGenerator: generated %d trades for %s", len(trades), factor_id)
            except Exception as exc:
                raise TradeGenerationError(f"IR trade generation failed for {factor_id}:{exc}") from exc
        return result


class TradeGenerator:
    """
    Combined trade generator for portfolio with both FX and IR risk factors.
    Delegates FX factors to FxTradeGenerator and IR factors to IrTradeGenerator.
    """

    def __init__(self) -> None:

        # initiate variables.
        self._fx_gen = FxTradeGenerator()
        self._ir_gen = IrTradeGenerator()

    def generate(self, assets: Dict[str, Any], trade_config: Dict[str, Any]) -> Dict[str, List[ElementaryTrade]]:
        """Generate elementary FX trades from loaded FxAsset instances."""

        # fetch asset class specific configurations.
        fx_config = trade_config.get("fx_config", trade_config)
        ir_config = trade_config.get("ir_config", trade_config)

        #
        result: Dict[str, List[ElementaryTrade]] = {}

        try:
            result.update(self._fx_gen.generate(assets, fx_config))
        except TradeGenerationError as exc:
            logger.warning("FXTradeGenerator: failed to generate FX trades for %s", assets)

        try:
            result.update(self._ir_gen.generate(assets, ir_config))
        except TradeGenerationError as exc:
            logger.warning("IRTradeGenerator: failed to generate IR trades for %s", assets)

        total = sum(len(v) for v in result.values())
        logger.info("TradeGenerator: total %d trades across %d factors", total, len(result))
        return result
