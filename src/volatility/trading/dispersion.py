"""
Dispersion Trading Strategy.

Dispersion trading exploits the difference between index implied
volatility and the volatility of its constituents.

Key insight: Index vol < weighted average of constituent vols
due to correlation (imperfect correlation = diversification).

Strategy:
- Sell index variance
- Buy constituent variance (basket)
- Profit from correlation < implied correlation

Example:
    from src.volatility.trading import DispersionTrader
    
    trader = DispersionTrader(
        index_ticker="SPX",
        constituents=["AAPL", "MSFT", "GOOGL", "AMZN"],
        weights=[0.25, 0.25, 0.25, 0.25],
    )
    
    analysis = trader.analyze(
        index_vol=0.18,
        constituent_vols=[0.25, 0.22, 0.28, 0.30],
        correlation_matrix=corr_matrix,
    )
    
    print(f"Implied correlation: {analysis.implied_correlation:.2%}")
    print(f"Dispersion signal: {analysis.signal}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DispersionConfig:
    """Configuration for dispersion trading."""
    
    # Signal thresholds
    implied_corr_threshold_long: float = 0.6  # Go long dispersion if < this
    implied_corr_threshold_short: float = 0.8  # Go short dispersion if > this
    
    # Position sizing
    index_notional: float = 1_000_000
    hedge_ratio: float = 1.0  # Ratio of basket to index
    
    # Risk limits
    max_correlation_exposure: float = 0.1  # Max correlation point exposure
    max_vega_exposure: float = 50_000


# =============================================================================
# Dispersion Analysis Result
# =============================================================================


@dataclass
class DispersionAnalysis:
    """Result from dispersion analysis."""
    
    # Volatilities
    index_vol: float
    avg_constituent_vol: float
    weighted_constituent_vol: float
    
    # Correlation
    implied_correlation: float
    realized_correlation: Optional[float] = None
    
    # Signal
    signal: str = "neutral"  # "long_dispersion", "short_dispersion", "neutral"
    signal_strength: float = 0.0
    
    # Theoretical values
    theoretical_index_vol: float = 0.0  # From correlation matrix
    
    # P&L attribution
    correlation_pnl: float = 0.0
    vega_pnl: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "index_vol": self.index_vol,
            "weighted_constituent_vol": self.weighted_constituent_vol,
            "implied_correlation": self.implied_correlation,
            "realized_correlation": self.realized_correlation,
            "signal": self.signal,
            "signal_strength": self.signal_strength,
        }


# =============================================================================
# Dispersion Trader
# =============================================================================


class DispersionTrader:
    """
    Dispersion trading strategy implementation.
    
    Dispersion = selling index volatility and buying constituent volatility.
    
    The strategy profits when:
    - Implied correlation > realized correlation (long dispersion)
    - Constituents move more than index implies
    
    Key relationship:
        σ²_index = Σᵢ Σⱼ wᵢwⱼρᵢⱼσᵢσⱼ
    
    Where:
        wᵢ = weight of constituent i
        ρᵢⱼ = correlation between i and j
        σᵢ = volatility of constituent i
    
    Example:
        trader = DispersionTrader(
            index_ticker="SPX",
            constituents=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
        )
        
        # Analyze dispersion opportunity
        analysis = trader.analyze(
            index_vol=0.18,
            constituent_vols=[0.25, 0.22, 0.28],
        )
        
        if analysis.signal == "long_dispersion":
            positions = trader.get_positions(notional=1_000_000)
    """
    
    def __init__(
        self,
        index_ticker: str = "INDEX",
        constituents: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
        config: Optional[DispersionConfig] = None,
    ) -> None:
        """
        Initialize dispersion trader.
        
        Parameters
        ----------
        index_ticker : str
            Index identifier.
        constituents : list of str
            Constituent tickers.
        weights : ndarray
            Portfolio weights (must sum to 1).
        config : DispersionConfig, optional
            Trading configuration.
        """
        self.index_ticker = index_ticker
        self.constituents = constituents or []
        
        if weights is not None:
            self.weights = np.asarray(weights)
            self.weights = self.weights / np.sum(self.weights)  # Normalize
        else:
            n = len(self.constituents)
            self.weights = np.ones(n) / n if n > 0 else np.array([])
        
        self.config = config or DispersionConfig()
    
    def analyze(
        self,
        index_vol: float,
        constituent_vols: np.ndarray,
        correlation_matrix: Optional[np.ndarray] = None,
        realized_correlation: Optional[float] = None,
    ) -> DispersionAnalysis:
        """
        Analyze dispersion trading opportunity.
        
        Parameters
        ----------
        index_vol : float
            Index implied volatility.
        constituent_vols : ndarray
            Constituent implied volatilities.
        correlation_matrix : ndarray, optional
            Correlation matrix. If None, assumes uniform correlation.
        realized_correlation : float, optional
            Historical realized correlation for comparison.
        
        Returns
        -------
        DispersionAnalysis
            Analysis result with signal.
        """
        constituent_vols = np.asarray(constituent_vols)
        n = len(constituent_vols)
        
        # Weighted average constituent vol
        weighted_vol = np.sqrt(np.sum((self.weights * constituent_vols) ** 2))
        avg_vol = np.mean(constituent_vols)
        
        # Implied correlation
        # σ²_index = Σᵢ Σⱼ wᵢwⱼρᵢⱼσᵢσⱼ
        # For uniform correlation ρ:
        # σ²_index = ρ * (Σ wᵢσᵢ)² + (1-ρ) * Σ wᵢ²σᵢ²
        
        weighted_vol_sum = np.sum(self.weights * constituent_vols)
        weighted_var_sum = np.sum(self.weights ** 2 * constituent_vols ** 2)
        
        # Solve for implied correlation
        if weighted_vol_sum ** 2 - weighted_var_sum > 1e-10:
            implied_corr = (
                index_vol ** 2 - weighted_var_sum
            ) / (weighted_vol_sum ** 2 - weighted_var_sum)
            implied_corr = np.clip(implied_corr, -1, 1)
        else:
            implied_corr = 1.0
        
        # Theoretical index vol from correlation matrix
        theoretical_index_vol = 0.0
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix)
            vol_diag = np.diag(constituent_vols)
            cov = vol_diag @ corr @ vol_diag
            theoretical_index_vol = np.sqrt(self.weights @ cov @ self.weights)
        
        # Generate signal
        signal = "neutral"
        signal_strength = 0.0
        
        if implied_corr < self.config.implied_corr_threshold_long:
            signal = "short_dispersion"  # Corr low, expect it to rise
            signal_strength = self.config.implied_corr_threshold_long - implied_corr
        elif implied_corr > self.config.implied_corr_threshold_short:
            signal = "long_dispersion"  # Corr high, expect it to fall
            signal_strength = implied_corr - self.config.implied_corr_threshold_short
        
        # Compare with realized if available
        if realized_correlation is not None:
            corr_diff = implied_corr - realized_correlation
            if corr_diff > 0.1:
                signal = "long_dispersion"
                signal_strength = max(signal_strength, corr_diff)
            elif corr_diff < -0.1:
                signal = "short_dispersion"
                signal_strength = max(signal_strength, -corr_diff)
        
        return DispersionAnalysis(
            index_vol=index_vol,
            avg_constituent_vol=avg_vol,
            weighted_constituent_vol=weighted_vol,
            implied_correlation=implied_corr,
            realized_correlation=realized_correlation,
            signal=signal,
            signal_strength=signal_strength,
            theoretical_index_vol=theoretical_index_vol,
        )
    
    def get_positions(
        self,
        analysis: DispersionAnalysis,
        notional: float = 1_000_000,
    ) -> Dict[str, float]:
        """
        Get position sizes for dispersion trade.
        
        Parameters
        ----------
        analysis : DispersionAnalysis
            Analysis result.
        notional : float
            Total notional.
        
        Returns
        -------
        dict
            Position dictionary {ticker: variance_notional}.
        """
        positions = {}
        
        if analysis.signal == "neutral":
            return positions
        
        sign = 1.0 if analysis.signal == "long_dispersion" else -1.0
        
        # Long dispersion: sell index var, buy constituent var
        # Short dispersion: buy index var, sell constituent var
        
        # Index position (opposite sign)
        positions[self.index_ticker] = -sign * notional
        
        # Constituent positions
        for i, ticker in enumerate(self.constituents):
            positions[ticker] = sign * self.weights[i] * notional * self.config.hedge_ratio
        
        return positions
    
    def compute_pnl(
        self,
        positions: Dict[str, float],
        realized_variances: Dict[str, float],
        strike_variances: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute P&L from realized variances.
        
        Parameters
        ----------
        positions : dict
            Variance notional positions.
        realized_variances : dict
            Realized variances per ticker.
        strike_variances : dict
            Strike variances per ticker.
        
        Returns
        -------
        dict
            P&L breakdown.
        """
        total_pnl = 0.0
        component_pnl = {}
        
        for ticker, notional in positions.items():
            if ticker in realized_variances and ticker in strike_variances:
                pnl = notional * (realized_variances[ticker] - strike_variances[ticker])
                component_pnl[ticker] = pnl
                total_pnl += pnl
        
        component_pnl["total"] = total_pnl
        return component_pnl


def compute_realized_correlation(
    returns_matrix: np.ndarray,
    window: Optional[int] = None,
) -> np.ndarray:
    """
    Compute realized correlation matrix from returns.
    
    Parameters
    ----------
    returns_matrix : ndarray
        Returns of shape (n_observations, n_assets).
    window : int, optional
        Rolling window. If None, uses full sample.
    
    Returns
    -------
    ndarray
        Correlation matrix of shape (n_assets, n_assets).
    """
    if window is not None and window < len(returns_matrix):
        returns_matrix = returns_matrix[-window:]
    
    return np.corrcoef(returns_matrix, rowvar=False)


def compute_average_correlation(corr_matrix: np.ndarray) -> float:
    """
    Compute average pairwise correlation.
    
    Parameters
    ----------
    corr_matrix : ndarray
        Correlation matrix.
    
    Returns
    -------
    float
        Average off-diagonal correlation.
    """
    n = corr_matrix.shape[0]
    if n < 2:
        return 0.0
    
    # Extract upper triangular (excluding diagonal)
    upper = corr_matrix[np.triu_indices(n, k=1)]
    
    return float(np.mean(upper))


__all__ = [
    "DispersionTrader",
    "DispersionAnalysis",
    "DispersionConfig",
    "compute_realized_correlation",
    "compute_average_correlation",
]
