"""Historical VaR from P&L series."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.risk.var.config import VarConfig, VarResult


def historical_var(
    pnl_series: np.ndarray,
    config: VarConfig,
    *,
    include_cvar: bool = True,
) -> VarResult:
    """
    Compute Historical VaR from a period P&L series.

    VaR is the negative of the (1 - confidence) quantile of P&L
    (e.g. 99% VaR = 1% quantile of P&L). Optionally scaled to horizon_days
    using sqrt(horizon_days) for i.i.d. period P&L.

    Parameters
    ----------
    pnl_series : np.ndarray
        Period P&L (e.g. daily). Can be from revaluation or backtest.
    config : VarConfig
        confidence, horizon_days (for scaling), method must be "historical".
    include_cvar : bool
        If True, compute CVaR (expected shortfall) as mean of P&L below VaR threshold.

    Returns
    -------
    VarResult
    """
    if config.method != "historical":
        raise ValueError("VarConfig.method must be 'historical' for historical_var.")
    pnl = np.asarray(pnl_series, dtype=float).ravel()
    if pnl.size == 0:
        raise ValueError("pnl_series must not be empty.")
    n = pnl.size
    q = 1.0 - config.confidence  # e.g. 0.01 for 99% VaR
    var_pnl = float(np.quantile(pnl, q, method="linear"))
    # VaR = negative of P&L quantile (loss is positive)
    var_one = float(-var_pnl)
    # Scale to horizon (i.i.d. assumption)
    scale = math.sqrt(max(1, config.horizon_days))
    var_scaled = var_one * scale

    cvar_val: Optional[float] = None
    if include_cvar:
        tail = pnl <= var_pnl
        if np.any(tail):
            cvar_pnl = float(np.mean(pnl[tail]))
            cvar_one = -cvar_pnl
            cvar_val = cvar_one * scale
        else:
            cvar_val = var_scaled

    return VarResult(
        var=var_scaled,
        cvar=cvar_val,
        method="historical",
        confidence=config.confidence,
        horizon_days=config.horizon_days,
        metadata={"n_observations": n},
    )
