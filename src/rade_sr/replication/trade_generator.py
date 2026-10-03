"""
Trade generator stage — generates elementary trades from all loaded assets.

In the portfolio-first architecture, the generator receives the complete
set of loaded assets and produces the full trade universe in one pass.
Deduplication ensures no trade appears twice even if the same factor_id
were to be processed multiple times.

Dispatch:
    asset_class == "fx"    → generate_fx_elementary_trades()
    asset_class == "rates" → generate_ir_elementary_trades()

Strike/tenor grids are user-defined relative values resolved against
each asset's reference level (spot for FX, par swap rate for IR).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from src.rade_sr.core.types import ElementaryTrade
from src.rade_sr.core.exceptions import TradeGenerationError
from src.rade_sr.instruments.base import InstrumentSpec, StrikeGrid, TenorGrid
from src.rade_sr.instruments.fx import generate_fx_elementary_trades
from src.rade_sr.instruments.rates import generate_ir_elementary_trades

logger = logging.getLogger(__name__)


def _specs_to_trades(
    specs: List[InstrumentSpec],
    notional: float = 1.0,
) -> List[ElementaryTrade]:
    """Convert InstrumentSpec objects to ElementaryTrade objects.

    This bridges the instruments layer (InstrumentSpec with pricing
    methods) to the pipeline layer (ElementaryTrade with serialisable
    parameters).
    """
    return [
        ElementaryTrade(
            trade_id=spec.to_trade_id(),
            asset_class=spec.asset_class,
            payoff_type=spec.payoff_type,
            factor_id=spec.factor_id,
            parameters=spec.to_pricer_params(),
            notional=notional,
        )
        for spec in specs
    ]


class RiskFactorAwareTradeGenerator:
    """Generates elementary trades by interrogating each asset's market data.

    Iterates over ALL loaded assets in the portfolio (not per-cluster),
    generates trades for each, then deduplicates by trade_id.

    Dispatches to ``generate_fx_elementary_trades()`` or
    ``generate_ir_elementary_trades()`` based on ``asset.asset_class``.
    """

    def generate(
        self,
        assets: Dict[str, Any],
        trade_config: Dict[str, Any],
    ) -> Dict[str, List[ElementaryTrade]]:
        """Generate and deduplicate elementary trades for the entire portfolio.

        Parameters
        ----------
        assets : dict[str, Any]
            ``factor_id → loaded Asset``.  All assets in the portfolio.
        trade_config : dict
            Trade generation parameters. Expected keys:

            - ``notional`` (float): Trade notional (default 1.0).
            - ``maturities`` (list[float]): Tenor grid in year fractions.
            - ``fx`` (dict): FX-specific overrides.
              - ``strike_pcts`` (list[float]): Strike as % of spot
                (e.g. ``[0.80, 0.90, 1.00, 1.10, 1.20]``).
              - ``include_forwards`` (bool)
              - ``include_vanillas`` (bool)
              - ``include_digitals`` (bool)
            - ``rates`` (dict): IR-specific overrides.
              - ``strike_offsets_bp`` (list[float]): Strike offsets in bp
                from par rate (e.g. ``[-200, -100, 0, 100, 200]``).
              - ``include_swaps`` (bool)
              - ``include_swaptions`` (bool)
              - ``include_caps_floors`` (bool)
              - ``vol_type`` (str): ``"normal"`` or ``"lognormal"``.

        Returns
        -------
        dict[str, list[ElementaryTrade]]
            ``factor_id → [trades]``.  Mirrors the assets dict.
        """
        result: Dict[str, List[ElementaryTrade]] = {}
        total_raw = 0
        total_dedup = 0

        for factor_id, asset in assets.items():
            try:
                factor_trades = self._generate_for_factor(
                    factor_id, asset, trade_config,
                )
                deduped = self._deduplicate(factor_trades)
                result[factor_id] = deduped
                total_raw += len(factor_trades)
                total_dedup += len(deduped)
                logger.debug(
                    "Generated %d trades for factor %s (%d after dedup)",
                    len(factor_trades), factor_id, len(deduped),
                )
            except Exception as exc:
                raise TradeGenerationError(
                    f"Trade generation failed for factor {factor_id!r}: {exc}"
                ) from exc

        logger.info(
            "Portfolio trade generation: %d trades generated, %d after dedup",
            total_raw, total_dedup,
        )
        return result

    def _generate_for_factor(
        self,
        factor_id: str,
        asset: Any,
        trade_config: Dict[str, Any],
    ) -> List[ElementaryTrade]:
        """Generate trades for one risk factor.

        Resolves user-defined relative grids against the asset's reference
        level, then dispatches to the appropriate instrument generator.
        """
        asset_class = getattr(asset, "asset_class", "unknown")
        notional = trade_config.get("notional", 1.0)
        maturities = trade_config.get("maturities", [0.25, 0.5, 1.0, 2.0, 5.0])

        tenor_grid = TenorGrid(values=maturities)

        if asset_class == "fx":
            return self._generate_fx(factor_id, asset, trade_config, tenor_grid, notional)
        elif asset_class == "rates":
            return self._generate_ir(factor_id, asset, trade_config, tenor_grid, notional)
        else:
            raise TradeGenerationError(
                f"No trade generator for asset class {asset_class!r}"
            )

    def _generate_fx(
        self,
        factor_id: str,
        asset: Any,
        trade_config: Dict[str, Any],
        tenor_grid: TenorGrid,
        notional: float,
    ) -> List[ElementaryTrade]:
        """Generate FX elementary trades.

        Resolves strike_pcts (percentages of spot) into absolute strikes.
        """
        fx_cfg = trade_config.get("fx", {})
        spot = getattr(asset, "spot", 1.0) or 1.0
        strike_pcts = fx_cfg.get("strike_pcts", [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
        absolute_strikes = [pct * spot for pct in strike_pcts]

        strike_grid = StrikeGrid(
            values=absolute_strikes,
            convention="absolute",
            reference_level=spot,
        )

        specs = generate_fx_elementary_trades(
            factor_id=factor_id,
            tenor_grid=tenor_grid,
            strike_grid=strike_grid,
            include_forwards=fx_cfg.get("include_forwards", True),
            include_vanillas=fx_cfg.get("include_vanillas", True),
            include_digitals=fx_cfg.get("include_digitals", True),
        )

        return _specs_to_trades(specs, notional)

    def _generate_ir(
        self,
        factor_id: str,
        asset: Any,
        trade_config: Dict[str, Any],
        tenor_grid: TenorGrid,
        notional: float,
    ) -> List[ElementaryTrade]:
        """Generate IR elementary trades.

        Resolves strike_offsets_bp (basis point offsets from par rate)
        into absolute strike rates.
        """
        ir_cfg = trade_config.get("rates", {})
        par_rate = getattr(asset, "spot", 0.04) or 0.04
        offsets_bp = ir_cfg.get("strike_offsets_bp", [-200, -100, -50, 0, 50, 100, 200])
        absolute_strikes = [par_rate + bp / 10_000 for bp in offsets_bp]

        strike_grid = StrikeGrid(
            values=absolute_strikes,
            convention="absolute",
            reference_level=par_rate,
        )

        specs = generate_ir_elementary_trades(
            factor_id=factor_id,
            tenor_grid=tenor_grid,
            strike_grid=strike_grid,
            include_swaps=ir_cfg.get("include_swaps", True),
            include_swaptions=ir_cfg.get("include_swaptions", True),
            include_caps_floors=ir_cfg.get("include_caps_floors", True),
            vol_type=ir_cfg.get("vol_type", "normal"),
        )

        return _specs_to_trades(specs, notional)

    @staticmethod
    def _deduplicate(trades: List[ElementaryTrade]) -> List[ElementaryTrade]:
        """Remove duplicate trades by trade_id, preserving first occurrence order."""
        seen: Set[str] = set()
        unique: List[ElementaryTrade] = []
        for t in trades:
            if t.trade_id not in seen:
                seen.add(t.trade_id)
                unique.append(t)
        return unique
