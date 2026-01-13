from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from src.marketdata.scenarios.interfaces import MarketView, ScenarioPack, ScenarioShock
from src.portfolio.core import Portfolio
from src.portfolio.portfolio import PortfolioPricer


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """
    Output of pricing a portfolio across a base market and a set of scenario markets.

    Attributes
    ----------
    scenario_names:
        Scenario names in evaluation order. Index 0 is always the base scenario (default "BASE").
    pv:
        Portfolio PV per scenario (aligned with scenario_names).
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
) -> ScenarioResult:
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
    ScenarioRunResult
        PV/PnL arrays and dicts, aligned to scenario ordering.
    """
    scenario_names, scenario_market_views, _ = build_scenario_markets(
        base_market=base_market,
        scenarios=scenarios,
        base_name=base_name,
    )

    pv = np.empty(len(scenario_market_views), dtype=float)

    # Price base + each scenario (MarketView-compatible)
    for i, market_view in enumerate(scenario_market_views):
        pv[i] = float(portfolio_pricer.price(portfolio, market_view).totals.pv)

    pv_base = float(pv[0])
    pnl = pv - pv_base

    pv_by = {name: float(val) for name, val in zip(scenario_names, pv)}
    pnl_by = {name: float(val) for name, val in zip(scenario_names, pnl)}

    return ScenarioResult(
        scenario_names=scenario_names,
        pv=pv,
        pnl=pnl,
        pv_by_scenario=pv_by,
        pnl_by_scenario=pnl_by,
    )


def build_scenario_markets(
    *,
    base_market: MarketView,
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
    base_name: str = "BASE",
) -> Tuple[List[str], List[MarketView], Dict[str, ScenarioShock]]:
    """
    Build scenario names + MarketViews and a mapping of name -> ScenarioShock.

    Notes
    -----
    - Always includes a base row first (base_name, base_market).
    - For ScenarioPack, uses the pack's mapping keys as the scenario names.
    - For sequences, uses shock.name if present, else the class name.

    Returns
    -------
    (scenario_names, scenario_market_views, shock_by_name)
    """
    if not isinstance(base_name, str) or not base_name:
        raise ValueError("base_name must be a non-empty string.")

    scenario_names: List[str] = [base_name]
    scenario_market_views: List[MarketView] = [base_market]
    shock_by_name: Dict[str, ScenarioShock] = {}

    if isinstance(scenarios, ScenarioPack):
        for scenario_name, shock in scenarios.scenarios.items():
            name = str(scenario_name)
            if name == base_name:
                raise ValueError(f"ScenarioPack contains a scenario named '{base_name}'. This conflicts with base_name.")
            if name in shock_by_name:
                raise ValueError(f"Duplicate scenario name '{name}' detected in ScenarioPack.")
            scenario_names.append(name)
            scenario_market_views.append(shock.apply(base_market))
            shock_by_name[name] = shock
        return scenario_names, scenario_market_views, shock_by_name

    for shock in scenarios:
        name = str(getattr(shock, "name", shock.__class__.__name__))
        if name == base_name:
            raise ValueError(f"Scenario sequence contains a shock named '{base_name}'. This conflicts with base_name.")
        if name in shock_by_name:
            raise ValueError(f"Duplicate scenario name '{name}' detected in scenario sequence.")
        scenario_names.append(name)
        scenario_market_views.append(shock.apply(base_market))
        shock_by_name[name] = shock

    return scenario_names, scenario_market_views, shock_by_name