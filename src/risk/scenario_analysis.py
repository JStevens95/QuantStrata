from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from src.portfolio.core import Portfolio
from src.pricers.portfolio import PortfolioPricer
from src.marketdata.scenarios.base import MarketView, ScenarioPack, ScenarioShock  # adjust path if needed


@dataclass(frozen=True, slots=True)
class StressResult:
    """
    Output of pricing a portfolio across a base market and a set of stressed markets.

    Attributes
    ----------
    scenario_names:
        Scenario names in evaluation order. Index 0 is always "BASE".
    pv:
        Portfolio PV per scenario (same ordering as scenario_names).
    pnl:
        Portfolio PnL per scenario computed as pv - pv_base.
    pv_by_scenario:
        Convenience mapping from scenario name -> PV.
    pnl_by_scenario:
        Convenience mapping from scenario name -> PnL.
    """
    scenario_names: List[str]
    pv: np.ndarray
    pnl: np.ndarray
    pv_by_scenario: Dict[str, float]
    pnl_by_scenario: Dict[str, float]


def run_portfolio_scenarios(
    portfolio: Portfolio,
    base_market: MarketView,
    portfolio_pricer: PortfolioPricer,
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
    *,
    base_name: str = "BASE",
) -> StressResult:
    """
    Price a portfolio under the base market and a set of scenario shocks.

    Parameters
    ----------
    portfolio:
        Portfolio to price.
    base_market:
        Base MarketView (can be a concrete Market or any MarketView wrapper).
    portfolio_pricer:
        PortfolioPricer configured with a pricer registry.
    scenarios:
        Either:
          - ScenarioPack: mapping of name -> ScenarioShock, or
          - Sequence[ScenarioShock]: ordered list of ScenarioShock objects.
    base_name:
        Label for the base scenario row.

    Returns
    -------
    StressResult
        PV/PnL arrays and dicts, aligned to the scenario ordering.
    """
    scenario_names, shocked_markets = _materialize_markets(
        base_market=base_market,
        scenarios=scenarios,
        base_name=base_name,
    )

    pv = np.empty(len(shocked_markets), dtype=float)

    for i, market_view in enumerate(shocked_markets):
        # Price under each scenario market view. PortfolioPricer should accept MarketView.
        pv[i] = float(portfolio_pricer.price(portfolio, market_view).totals.pv)

    pv_base = float(pv[0])
    pnl = pv - pv_base

    pv_by = {name: float(val) for name, val in zip(scenario_names, pv)}
    pnl_by = {name: float(val) for name, val in zip(scenario_names, pnl)}

    return StressResult(
        scenario_names=scenario_names,
        pv=pv,
        pnl=pnl,
        pv_by_scenario=pv_by,
        pnl_by_scenario=pnl_by,
    )


def _materialize_markets(
    base_market: MarketView,
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
    base_name: str,
) -> Tuple[List[str], List[MarketView]]:
    """
    Convert scenarios into an ordered list of (scenario_names, market_views),
    always including a base row first.

    Notes
    -----
    - If `scenarios` is a ScenarioPack, we preserve the mapping iteration order.
      (In Python 3.7+ dict order is guaranteed insertion order.)
    - If `scenarios` is a list, we preserve list order and use shock.name for labels.
    """
    scenario_names: List[str] = [base_name]
    market_views: List[MarketView] = [base_market]

    if isinstance(scenarios, ScenarioPack):
        # Apply all with explicit names from the pack.
        shocked: Mapping[str, MarketView] = scenarios.apply_all(base_market)
        for scenario_name, market_view in shocked.items():
            scenario_names.append(str(scenario_name))
            market_views.append(market_view)
        return scenario_names, market_views

    # Otherwise: sequence of ScenarioShock (ordered)
    for shock in scenarios:
        shock_name = getattr(shock, "name", shock.__class__.__name__)
        scenario_names.append(str(shock_name))
        market_views.append(shock.apply(base_market))

    return scenario_names, market_views