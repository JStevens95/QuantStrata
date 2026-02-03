"""
Backtest delta hedging strategies: BSM vs ML.

Runs hedged short option over simulated paths; computes terminal P&L and
transaction costs for BSM delta hedging and (optionally) ML-predicted delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.m_learning.data.delta_hedging import HedgingPath, path_to_feature_target_arrays


def _payoff(spot: float, strike: float, option_type: int) -> float:
    """Option payoff (we are short, so we pay this)."""
    if option_type >= 0:  # call
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


@dataclass
class BacktestResult:
    """
    Per-path backtest outcomes.

    Attributes
    ----------
    terminal_pnl_bsm : np.ndarray
        Shape (n_paths,). Terminal P&L when hedging with BSM delta.
    terminal_pnl_ml : Optional[np.ndarray]
        Shape (n_paths,) if ML was used.
    cost_bsm : np.ndarray
        Total transaction cost per path (BSM strategy).
    cost_ml : Optional[np.ndarray]
        Total transaction cost per path (ML strategy).
    """

    terminal_pnl_bsm: np.ndarray
    terminal_pnl_ml: Optional[np.ndarray] = None
    cost_bsm: np.ndarray = field(default_factory=lambda: np.array([]))
    cost_ml: Optional[np.ndarray] = None

    @property
    def n_paths(self) -> int:
        return len(self.terminal_pnl_bsm)


def run_single_path_backtest(
    path: HedgingPath,
    K: float,
    option_type: int,
    cost_rate: float,
    delta_sequence: np.ndarray,
) -> tuple[float, float]:
    """
    Run backtest on one path for a given delta sequence (BSM or ML).

    P&L: We are short 1 option. Cash starts at V(0,S0) - delta_0*S0.
    At each t_k we rebalance to delta_k: cost c*|delta_k - delta_{k-1}|*S_k;
    cash -= (delta_k - delta_{k-1})*S_k + cost.
    Terminal P&L = cash + delta_N*S_T - payoff (we pay the payoff).

    Parameters
    ----------
    path : HedgingPath
        Simulated path (spot, option_value, etc.).
    K : float
        Strike.
    option_type : int
        1 = call, -1 = put.
    cost_rate : float
        Proportional cost per unit notional traded: cost = cost_rate * |Δ delta| * S.
    delta_sequence : np.ndarray
        Shape (n_steps+1,). Delta to hold on (t_k, t_{k+1}).

    Returns
    -------
    terminal_pnl : float
    total_cost : float
    """
    n = len(path.times)
    S = path.spot
    V0 = path.option_value[0]
    delta_0 = delta_sequence[0]
    cash = V0 - delta_0 * S[0]
    total_cost = 0.0

    for k in range(1, n):
        d_prev = delta_sequence[k - 1]
        d_curr = delta_sequence[k]
        s_k = S[k]
        trade = d_curr - d_prev
        cost_k = cost_rate * abs(trade) * s_k
        total_cost += cost_k
        cash -= trade * s_k + cost_k

    payoff = _payoff(S[-1], K, option_type)
    terminal_pnl = cash + delta_sequence[-1] * S[-1] - payoff
    return terminal_pnl, total_cost


def run_delta_hedging_backtest(
    paths: List[HedgingPath],
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: int,
    cost_rate: float = 0.0,
    ml_delta_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> BacktestResult:
    """
    Run BSM and (optionally) ML delta hedging on the same paths.

    Parameters
    ----------
    paths : list of HedgingPath
        Simulated paths (e.g. from simulate_hedging_paths).
    K, T, r, sigma : float
        Option parameters (used to build features for ML).
    option_type : int
        1 = call, -1 = put.
    cost_rate : float
        Proportional transaction cost per unit notional.
    ml_delta_fn : callable, optional
        If provided, takes features array of shape (n_steps+1, n_features) and
        returns delta array of shape (n_steps+1,). Used for ML strategy.

    Returns
    -------
    BacktestResult
        terminal_pnl_bsm, terminal_pnl_ml (if ml_delta_fn), cost_bsm, cost_ml.
    """
    n_paths = len(paths)
    pnl_bsm = np.zeros(n_paths)
    cost_bsm = np.zeros(n_paths)
    pnl_ml: Optional[np.ndarray] = None
    cost_ml: Optional[np.ndarray] = None
    if ml_delta_fn is not None:
        pnl_ml = np.zeros(n_paths)
        cost_ml = np.zeros(n_paths)

    for i, path in enumerate(paths):
        # BSM strategy: use path.delta
        pnl_bsm[i], cost_bsm[i] = run_single_path_backtest(
            path, K, option_type, cost_rate, path.delta
        )
        if ml_delta_fn is not None:
            features, _ = path_to_feature_target_arrays(
                path, K, T, r, sigma, option_type
            )
            delta_ml = np.squeeze(ml_delta_fn(features))
            if delta_ml.ndim == 0:
                delta_ml = np.full(len(path.times), float(delta_ml))
            pnl_ml[i], cost_ml[i] = run_single_path_backtest(
                path, K, option_type, cost_rate, delta_ml
            )

    return BacktestResult(
        terminal_pnl_bsm=pnl_bsm,
        terminal_pnl_ml=pnl_ml,
        cost_bsm=cost_bsm,
        cost_ml=cost_ml,
    )


def backtest_summary_stats(
    result: BacktestResult,
    var_percentile: float = 5.0,
) -> dict:
    """
    Compute summary statistics for backtest result.

    Returns
    -------
    dict
        Keys: mean_pnl_bsm, std_pnl_bsm, var_bsm, cvar_bsm, mean_cost_bsm;
        and if ML: mean_pnl_ml, std_pnl_ml, var_ml, cvar_ml, mean_cost_ml.
        VaR/CVaR are at the given percentile (e.g. 5% left tail).
    """
    stats = {
        "mean_pnl_bsm": float(np.mean(result.terminal_pnl_bsm)),
        "std_pnl_bsm": float(np.std(result.terminal_pnl_bsm)),
        "var_bsm": float(np.percentile(result.terminal_pnl_bsm, var_percentile)),
        "mean_cost_bsm": float(np.mean(result.cost_bsm)),
    }
    # CVaR = mean of outcomes below VaR
    var_val = stats["var_bsm"]
    stats["cvar_bsm"] = float(
        np.mean(result.terminal_pnl_bsm[result.terminal_pnl_bsm <= var_val])
        if np.any(result.terminal_pnl_bsm <= var_val)
        else var_val
    )
    if result.terminal_pnl_ml is not None and result.cost_ml is not None:
        stats["mean_pnl_ml"] = float(np.mean(result.terminal_pnl_ml))
        stats["std_pnl_ml"] = float(np.std(result.terminal_pnl_ml))
        stats["var_ml"] = float(np.percentile(result.terminal_pnl_ml, var_percentile))
        stats["mean_cost_ml"] = float(np.mean(result.cost_ml))
        var_ml = stats["var_ml"]
        stats["cvar_ml"] = float(
            np.mean(result.terminal_pnl_ml[result.terminal_pnl_ml <= var_ml])
            if np.any(result.terminal_pnl_ml <= var_ml)
            else var_ml
        )
    return stats
