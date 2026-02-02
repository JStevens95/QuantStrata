"""Monte Carlo VaR: simulate factor shocks and revalue portfolio."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from src.marketdata.scenarios.interfaces import MarketView, ScenarioShock
from src.marketdata.scenarios.shocks import CompositeShock
from src.risk.sensitivities.result import SensitivityKey
from src.risk.var.config import VarConfig, VarResult


class FactorModel(Protocol):
    """Protocol for factor models used in MC VaR."""

    def sample_shocks(self, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
        """Return (n_paths, n_factors) array of factor shocks (e.g. daily returns)."""
        ...

    def build_shocked_market(self, base_market: MarketView, shock_row: np.ndarray) -> MarketView:
        """Return a MarketView with the given shock row applied to base_market."""
        ...

    @property
    def factor_order(self) -> Sequence[SensitivityKey]:
        """Order of factors (for alignment with shock_row)."""
        ...


class DiagonalFactorModel:
    """
    Diagonal factor model: independent normal factor returns and shock builders per factor.

    sample_shocks returns standard_normal * volatilities (daily vols).
    build_shocked_market applies a CompositeShock of shock_builder[key](value) for each factor.
    """

    def __init__(
        self,
        factor_order: Sequence[SensitivityKey],
        factor_volatilities: Dict[SensitivityKey, float],
        shock_builders: Dict[SensitivityKey, Callable[[float], ScenarioShock]],
    ) -> None:
        if not factor_order:
            raise ValueError("factor_order must not be empty.")
        for k in factor_order:
            if k not in factor_volatilities:
                raise ValueError(f"factor_volatilities missing key {k}.")
            if k not in shock_builders:
                raise ValueError(f"shock_builders missing key {k}.")
        self._factor_order = list(factor_order)
        self._vols = np.array([float(factor_volatilities[k]) for k in self._factor_order])
        self._shock_builders = shock_builders

    def sample_shocks(self, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_paths, len(self._factor_order))) * self._vols

    def build_shocked_market(self, base_market: MarketView, shock_row: np.ndarray) -> MarketView:
        shock_row = np.asarray(shock_row, dtype=float).ravel()
        if shock_row.size != len(self._factor_order):
            raise ValueError("shock_row length must match factor_order.")
        shocks: List[ScenarioShock] = []
        for i, key in enumerate(self._factor_order):
            shocks.append(self._shock_builders[key](float(shock_row[i])))
        composite = CompositeShock(name="mc_path", shocks=shocks)
        return composite.apply(base_market)

    @property
    def factor_order(self) -> List[SensitivityKey]:
        return self._factor_order.copy()


def mc_var(
    portfolio: Any,
    market: MarketView,
    portfolio_pricer: Any,
    factor_model: FactorModel,
    config: VarConfig,
    *,
    n_paths: int = 10_000,
    include_cvar: bool = True,
    seed: Optional[int] = None,
) -> VarResult:
    """
    Compute Monte Carlo VaR by simulating factor shocks and full revaluation.

    For each path: sample factor shocks, build shocked market, price portfolio,
    P&L = PV(shocked) - PV(base). VaR = negative of (1 - confidence) quantile of P&L.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio to price.
    market : MarketView
        Base market.
    portfolio_pricer : PortfolioPricer-like
        price(portfolio, market) -> result with totals.pv.
    factor_model : FactorModel
        sample_shocks(n_paths), build_shocked_market(base_market, shock_row).
    config : VarConfig
        confidence, horizon_days; method must be "mc".
    n_paths : int
        Number of simulation paths.
    include_cvar : bool
        If True, CVaR = mean of P&L below VaR threshold.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    VarResult
    """
    if config.method != "mc":
        raise ValueError("VarConfig.method must be 'mc' for mc_var.")
    pv_base = float(portfolio_pricer.price(portfolio, market).totals.pv)
    shocks = factor_model.sample_shocks(n_paths, seed=seed)
    scale = 1.0
    if config.horizon_days > 1:
        scale = np.sqrt(config.horizon_days)
    shocks_scaled = shocks * scale

    pnl = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        shocked = factor_model.build_shocked_market(market, shocks_scaled[i])
        pv_s = float(portfolio_pricer.price(portfolio, shocked).totals.pv)
        pnl[i] = pv_s - pv_base

    q = 1.0 - config.confidence
    var_pnl = float(np.quantile(pnl, q, method="linear"))
    var_scaled = float(-var_pnl)

    cvar_val: Optional[float] = None
    if include_cvar:
        tail = pnl <= var_pnl
        if np.any(tail):
            cvar_val = float(-np.mean(pnl[tail]))
        else:
            cvar_val = var_scaled

    return VarResult(
        var=var_scaled,
        cvar=cvar_val,
        method="mc",
        confidence=config.confidence,
        horizon_days=config.horizon_days,
        metadata={"n_simulations": n_paths},
    )
