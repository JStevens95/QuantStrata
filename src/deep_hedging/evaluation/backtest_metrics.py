"""
Backtest-specific metrics for hedging evaluation.

Extends the base hedging metrics with backtest-specific measures
like out-of-sample performance, regime analysis, and benchmark comparison.

Example:
    from src.deep_hedging.evaluation.backtest_metrics import HedgingBacktestMetrics
    
    metrics = HedgingBacktestMetrics()
    
    result = metrics.compute(
        agent_pnl=agent_pnl_series,
        benchmark_pnl=delta_hedge_pnl,
        market_data=historical_data,
    )
    
    print(f"Sharpe: {result['sharpe_ratio']:.2f}")
    print(f"Information ratio: {result['information_ratio']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Backtest Metrics Result
# =============================================================================


@dataclass
class BacktestMetricsResult:
    """
    Complete metrics result from hedging backtest.
    
    Attributes
    ----------
    # Performance metrics
    total_return : float
        Total return over backtest period.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    sortino_ratio : float
        Sortino ratio (downside risk adjusted).
    calmar_ratio : float
        Return / max drawdown.
    
    # Risk metrics
    volatility : float
        Annualized volatility.
    max_drawdown : float
        Maximum drawdown.
    var_95 : float
        95% Value at Risk.
    cvar_95 : float
        95% Conditional VaR (Expected Shortfall).
    
    # Benchmark comparison
    excess_return : float
        Return above benchmark.
    tracking_error : float
        Standard deviation of excess returns.
    information_ratio : float
        Excess return / tracking error.
    outperformance_rate : float
        Fraction of periods beating benchmark.
    
    # Hedging-specific
    mean_hedge_cost : float
        Average hedging cost.
    hedge_efficiency : float
        Reduction in P&L volatility vs unhedged.
    turnover : float
        Average position turnover.
    """
    
    # Performance
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Benchmark
    excess_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    outperformance_rate: float = 0.0
    
    # Hedging
    mean_hedge_cost: float = 0.0
    hedge_efficiency: float = 0.0
    turnover: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "excess_return": self.excess_return,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "outperformance_rate": self.outperformance_rate,
            "mean_hedge_cost": self.mean_hedge_cost,
            "hedge_efficiency": self.hedge_efficiency,
            "turnover": self.turnover,
        }


# =============================================================================
# Hedging Backtest Metrics
# =============================================================================


class HedgingBacktestMetrics:
    """
    Compute comprehensive metrics for hedging backtests.
    
    Provides:
    - Standard performance metrics (Sharpe, Sortino, Calmar)
    - Risk metrics (VaR, CVaR, drawdown)
    - Benchmark comparison (information ratio, tracking error)
    - Hedging-specific metrics (hedge efficiency, turnover)
    
    Example:
        metrics = HedgingBacktestMetrics(annualization=252)
        
        result = metrics.compute(
            pnl_series=agent_pnl,
            benchmark_pnl=delta_hedge_pnl,
            costs=transaction_costs,
            positions=position_history,
        )
        
        print(result.sharpe_ratio)
        print(result.information_ratio)
    """
    
    def __init__(
        self,
        annualization: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        annualization : int
            Number of periods per year for annualization.
        risk_free_rate : float
            Annualized risk-free rate.
        """
        self.annualization = annualization
        self.risk_free_rate = risk_free_rate
    
    def compute(
        self,
        pnl_series: np.ndarray,
        benchmark_pnl: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        unhedged_pnl: Optional[np.ndarray] = None,
    ) -> BacktestMetricsResult:
        """
        Compute all backtest metrics.
        
        Parameters
        ----------
        pnl_series : ndarray
            P&L series (daily or per-period).
        benchmark_pnl : ndarray, optional
            Benchmark (e.g., delta hedge) P&L series.
        costs : ndarray, optional
            Transaction cost series.
        positions : ndarray, optional
            Position history for turnover calculation.
        unhedged_pnl : ndarray, optional
            Unhedged P&L for efficiency calculation.
        
        Returns
        -------
        BacktestMetricsResult
            Complete metrics result.
        """
        pnl = np.asarray(pnl_series)
        n_periods = len(pnl)
        
        # Performance metrics
        total_return = float(np.sum(pnl))
        volatility = self._compute_volatility(pnl)
        sharpe = self._compute_sharpe(pnl)
        sortino = self._compute_sortino(pnl)
        max_dd = self._compute_max_drawdown(pnl)
        calmar = total_return / max_dd if max_dd > 1e-8 else 0.0
        
        # Risk metrics
        var_95 = self._compute_var(pnl, 0.05)
        cvar_95 = self._compute_cvar(pnl, 0.05)
        
        # Benchmark metrics
        excess_return = 0.0
        tracking_error = 0.0
        information_ratio = 0.0
        outperformance_rate = 0.0
        
        if benchmark_pnl is not None:
            benchmark = np.asarray(benchmark_pnl)
            excess_returns = pnl - benchmark
            excess_return = float(np.sum(excess_returns))
            tracking_error = float(np.std(excess_returns) * np.sqrt(self.annualization))
            if tracking_error > 1e-8:
                information_ratio = excess_return / tracking_error
            outperformance_rate = float(np.mean(pnl > benchmark))
        
        # Hedging metrics
        mean_hedge_cost = 0.0
        if costs is not None:
            mean_hedge_cost = float(np.mean(costs))
        
        turnover = 0.0
        if positions is not None:
            position_arr = np.asarray(positions)
            if len(position_arr) > 1:
                turnover = float(np.mean(np.abs(np.diff(position_arr))))
        
        hedge_efficiency = 0.0
        if unhedged_pnl is not None:
            unhedged = np.asarray(unhedged_pnl)
            unhedged_vol = np.std(unhedged)
            hedged_vol = np.std(pnl)
            if unhedged_vol > 1e-8:
                hedge_efficiency = 1.0 - hedged_vol / unhedged_vol
        
        return BacktestMetricsResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            volatility=volatility,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            excess_return=excess_return,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            outperformance_rate=outperformance_rate,
            mean_hedge_cost=mean_hedge_cost,
            hedge_efficiency=hedge_efficiency,
            turnover=turnover,
        )
    
    def compute_rolling_metrics(
        self,
        pnl_series: np.ndarray,
        window: int = 60,
    ) -> Dict[str, np.ndarray]:
        """
        Compute rolling metrics over time.
        
        Parameters
        ----------
        pnl_series : ndarray
            P&L series.
        window : int
            Rolling window size.
        
        Returns
        -------
        dict
            Dictionary of rolling metric series.
        """
        pnl = np.asarray(pnl_series)
        n = len(pnl)
        
        rolling_sharpe = np.full(n, np.nan)
        rolling_vol = np.full(n, np.nan)
        rolling_drawdown = np.full(n, np.nan)
        
        for i in range(window, n):
            window_pnl = pnl[i - window:i]
            rolling_sharpe[i] = self._compute_sharpe(window_pnl)
            rolling_vol[i] = self._compute_volatility(window_pnl)
            rolling_drawdown[i] = self._compute_max_drawdown(window_pnl)
        
        return {
            "rolling_sharpe": rolling_sharpe,
            "rolling_volatility": rolling_vol,
            "rolling_drawdown": rolling_drawdown,
        }
    
    def compute_regime_analysis(
        self,
        pnl_series: np.ndarray,
        market_returns: np.ndarray,
        vol_threshold: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance across market regimes.
        
        Parameters
        ----------
        pnl_series : ndarray
            P&L series.
        market_returns : ndarray
            Market return series for regime detection.
        vol_threshold : float
            Volatility threshold for high-vol regime.
        
        Returns
        -------
        dict
            Metrics for each regime.
        """
        pnl = np.asarray(pnl_series)
        market = np.asarray(market_returns)
        
        # Compute rolling vol for regime detection
        rolling_vol = np.zeros(len(market))
        for i in range(20, len(market)):
            rolling_vol[i] = np.std(market[i - 20:i]) * np.sqrt(252)
        rolling_vol[:20] = rolling_vol[20]
        
        # Define regimes
        high_vol = rolling_vol > vol_threshold
        low_vol = ~high_vol
        
        # Compute metrics per regime
        regimes = {
            "high_volatility": {
                "n_periods": int(np.sum(high_vol)),
                "mean_pnl": float(np.mean(pnl[high_vol])) if np.any(high_vol) else 0.0,
                "sharpe": self._compute_sharpe(pnl[high_vol]) if np.any(high_vol) else 0.0,
            },
            "low_volatility": {
                "n_periods": int(np.sum(low_vol)),
                "mean_pnl": float(np.mean(pnl[low_vol])) if np.any(low_vol) else 0.0,
                "sharpe": self._compute_sharpe(pnl[low_vol]) if np.any(low_vol) else 0.0,
            },
        }
        
        return regimes
    
    def _compute_volatility(self, pnl: np.ndarray) -> float:
        """Compute annualized volatility."""
        return float(np.std(pnl) * np.sqrt(self.annualization))
    
    def _compute_sharpe(self, pnl: np.ndarray) -> float:
        """Compute Sharpe ratio."""
        if len(pnl) < 2:
            return 0.0
        
        mean_ret = np.mean(pnl)
        std_ret = np.std(pnl)
        
        if std_ret < 1e-8:
            return 0.0
        
        excess = mean_ret - self.risk_free_rate / self.annualization
        sharpe = excess / std_ret * np.sqrt(self.annualization)
        
        return float(sharpe)
    
    def _compute_sortino(self, pnl: np.ndarray) -> float:
        """Compute Sortino ratio (downside risk adjusted)."""
        if len(pnl) < 2:
            return 0.0
        
        mean_ret = np.mean(pnl)
        downside_returns = pnl[pnl < 0]
        
        if len(downside_returns) < 2:
            return float("inf") if mean_ret > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std < 1e-8:
            return float("inf") if mean_ret > 0 else 0.0
        
        sortino = mean_ret / downside_std * np.sqrt(self.annualization)
        
        return float(sortino)
    
    def _compute_max_drawdown(self, pnl: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        
        if np.max(running_max) < 1e-8:
            return 0.0
        
        return float(np.max(drawdowns))
    
    def _compute_var(self, pnl: np.ndarray, alpha: float) -> float:
        """Compute Value at Risk."""
        return float(np.percentile(pnl, alpha * 100))
    
    def _compute_cvar(self, pnl: np.ndarray, alpha: float) -> float:
        """Compute Conditional VaR (Expected Shortfall)."""
        var = self._compute_var(pnl, alpha)
        tail = pnl[pnl <= var]
        
        if len(tail) == 0:
            return var
        
        return float(np.mean(tail))


__all__ = [
    "HedgingBacktestMetrics",
    "BacktestMetricsResult",
]
