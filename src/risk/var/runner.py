"""VaR facade: dispatch by config.method to historical, parametric, or mc."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.risk.sensitivities.result import SensitivitiesReport, SensitivityKey
from src.risk.var.config import VarConfig, VarResult
from src.risk.var.historical import historical_var
from src.risk.var.mc import FactorModel, mc_var
from src.risk.var.parametric import parametric_var


def compute_var(
    config: VarConfig,
    *,
    pnl_series: Optional[np.ndarray] = None,
    sensitivities_report: Optional[SensitivitiesReport] = None,
    factor_volatilities: Optional[dict[SensitivityKey, float]] = None,
    factor_correlation: Optional[np.ndarray] = None,
    factor_order: Optional[list[SensitivityKey]] = None,
    portfolio: Any = None,
    market: Any = None,
    portfolio_pricer: Any = None,
    factor_model: Optional[FactorModel] = None,
    n_paths: int = 10_000,
    include_cvar: bool = True,
    seed: Optional[int] = None,
) -> VarResult:
    """
    Compute VaR by dispatching to historical, parametric, or mc based on config.method.

    Parameters
    ----------
    config : VarConfig
        method must be "historical", "parametric", or "mc".
    pnl_series : np.ndarray, optional
        Required for method="historical".
    sensitivities_report : SensitivitiesReport, optional
        Required for method="parametric".
    factor_volatilities : dict, optional
        Required for method="parametric".
    factor_correlation : np.ndarray, optional
        Optional for method="parametric".
    factor_order : list, optional
        Optional for method="parametric".
    portfolio, market, portfolio_pricer : optional
        Required for method="mc".
    factor_model : FactorModel, optional
        Required for method="mc".
    n_paths : int
        Used for method="mc".
    include_cvar : bool
        Passed to underlying method.
    seed : int, optional
        Passed to method="mc" for reproducibility.

    Returns
    -------
    VarResult
    """
    if config.method == "historical":
        if pnl_series is None:
            raise ValueError("pnl_series is required for historical VaR.")
        return historical_var(pnl_series, config, include_cvar=include_cvar)
    if config.method == "parametric":
        if sensitivities_report is None or factor_volatilities is None:
            raise ValueError(
                "sensitivities_report and factor_volatilities are required for parametric VaR."
            )
        return parametric_var(
            sensitivities_report,
            factor_volatilities,
            config,
            factor_correlation=factor_correlation,
            factor_order=factor_order,
            include_cvar=include_cvar,
        )
    if config.method == "mc":
        if portfolio is None or market is None or portfolio_pricer is None or factor_model is None:
            raise ValueError(
                "portfolio, market, portfolio_pricer, and factor_model are required for mc VaR."
            )
        return mc_var(
            portfolio,
            market,
            portfolio_pricer,
            factor_model,
            config,
            n_paths=n_paths,
            include_cvar=include_cvar,
            seed=seed,
        )
    raise ValueError(f"Unknown config.method: {config.method!r}.")
