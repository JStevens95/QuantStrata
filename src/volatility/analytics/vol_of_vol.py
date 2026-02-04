"""
Volatility-of-Volatility Analytics.

Measures the volatility of volatility itself, useful for:
- Pricing vol derivatives (VIX options, vol swaps)
- Risk management (tail risk)
- Regime detection

Key metrics:
- Vol of implied vol (VVIX-like)
- Vol of realized vol
- Vol clustering measures
- Regime indicators

Example:
    from src.volatility.analytics import VolOfVolAnalyzer
    
    analyzer = VolOfVolAnalyzer()
    
    metrics = analyzer.analyze(
        implied_vols=historical_iv_series,
        realized_vols=historical_rv_series,
    )
    
    print(f"Vol of implied vol: {metrics.vol_of_iv:.2%}")
    print(f"Vol regime: {metrics.regime}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Vol-of-Vol Metrics
# =============================================================================


@dataclass
class VolOfVolMetrics:
    """
    Volatility-of-volatility metrics.
    
    Attributes
    ----------
    vol_of_iv : float
        Volatility of implied volatility (annualized).
    vol_of_rv : float
        Volatility of realized volatility.
    mean_iv : float
        Mean implied volatility.
    mean_rv : float
        Mean realized volatility.
    iv_rv_spread : float
        IV - RV spread (risk premium).
    vol_persistence : float
        GARCH-like persistence measure.
    vol_mean_reversion : float
        Mean reversion speed estimate.
    regime : str
        Current volatility regime.
    regime_score : float
        Regime confidence score.
    """
    
    vol_of_iv: float
    vol_of_rv: float
    mean_iv: float
    mean_rv: float
    iv_rv_spread: float
    vol_persistence: float
    vol_mean_reversion: float
    regime: str
    regime_score: float
    
    # Additional statistics
    iv_percentile: float = 0.5
    rv_percentile: float = 0.5
    iv_skewness: float = 0.0
    iv_kurtosis: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "vol_of_iv": self.vol_of_iv,
            "vol_of_rv": self.vol_of_rv,
            "mean_iv": self.mean_iv,
            "mean_rv": self.mean_rv,
            "iv_rv_spread": self.iv_rv_spread,
            "regime": self.regime,
            "regime_score": self.regime_score,
            "iv_percentile": self.iv_percentile,
        }


# =============================================================================
# Vol-of-Vol Analyzer
# =============================================================================


class VolOfVolAnalyzer:
    """
    Analyze volatility-of-volatility.
    
    Provides metrics for understanding the dynamics of volatility itself,
    including:
    - Second-order volatility (vol of vol)
    - Volatility regimes
    - Mean reversion characteristics
    - Risk premium (IV vs RV)
    
    Example:
        analyzer = VolOfVolAnalyzer(window=20, annualization=252)
        
        metrics = analyzer.analyze(
            implied_vols=iv_series,    # Daily IV observations
            realized_vols=rv_series,   # Daily RV observations
        )
        
        print(f"Vol-of-IV: {metrics.vol_of_iv:.2%}")
        print(f"Regime: {metrics.regime}")
        
        # Compute rolling vol-of-vol
        rolling = analyzer.compute_rolling(implied_vols=iv_series, window=20)
    """
    
    def __init__(
        self,
        window: int = 20,
        annualization: int = 252,
        regime_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize analyzer.
        
        Parameters
        ----------
        window : int
            Window for vol-of-vol calculation.
        annualization : int
            Trading days per year.
        regime_thresholds : dict, optional
            Thresholds for regime classification.
        """
        self.window = window
        self.annualization = annualization
        
        self.regime_thresholds = regime_thresholds or {
            "low": 0.15,       # IV below this = low vol regime
            "high": 0.25,      # IV above this = high vol regime
            "crisis": 0.40,    # IV above this = crisis regime
        }
    
    def analyze(
        self,
        implied_vols: np.ndarray,
        realized_vols: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
    ) -> VolOfVolMetrics:
        """
        Perform comprehensive vol-of-vol analysis.
        
        Parameters
        ----------
        implied_vols : ndarray
            Time series of implied volatilities.
        realized_vols : ndarray, optional
            Time series of realized volatilities.
        prices : ndarray, optional
            Underlying prices (for RV calculation if not provided).
        
        Returns
        -------
        VolOfVolMetrics
            Analysis metrics.
        """
        iv = np.asarray(implied_vols)
        
        # Compute realized vol if needed
        if realized_vols is not None:
            rv = np.asarray(realized_vols)
        elif prices is not None:
            rv = self._compute_realized_vol(prices)
        else:
            rv = np.zeros_like(iv)
        
        # Basic statistics
        mean_iv = float(np.mean(iv))
        mean_rv = float(np.mean(rv))
        
        # Vol of vol (using log returns of vol)
        iv_returns = np.diff(np.log(np.maximum(iv, 1e-8)))
        vol_of_iv = float(np.std(iv_returns) * np.sqrt(self.annualization))
        
        rv_returns = np.diff(np.log(np.maximum(rv, 1e-8))) if np.any(rv > 0) else np.zeros(len(iv) - 1)
        vol_of_rv = float(np.std(rv_returns) * np.sqrt(self.annualization))
        
        # IV-RV spread
        iv_rv_spread = mean_iv - mean_rv
        
        # Persistence (AR(1) coefficient approximation)
        vol_persistence = self._estimate_persistence(iv)
        
        # Mean reversion (half-life)
        vol_mean_reversion = self._estimate_mean_reversion(iv)
        
        # Current regime
        current_iv = iv[-1] if len(iv) > 0 else mean_iv
        regime, regime_score = self._classify_regime(current_iv, iv)
        
        # Percentiles
        iv_percentile = float(np.mean(iv <= current_iv)) if len(iv) > 0 else 0.5
        rv_percentile = float(np.mean(rv <= rv[-1])) if len(rv) > 0 else 0.5
        
        # Higher moments
        iv_skewness = self._compute_skewness(iv_returns)
        iv_kurtosis = self._compute_kurtosis(iv_returns)
        
        return VolOfVolMetrics(
            vol_of_iv=vol_of_iv,
            vol_of_rv=vol_of_rv,
            mean_iv=mean_iv,
            mean_rv=mean_rv,
            iv_rv_spread=iv_rv_spread,
            vol_persistence=vol_persistence,
            vol_mean_reversion=vol_mean_reversion,
            regime=regime,
            regime_score=regime_score,
            iv_percentile=iv_percentile,
            rv_percentile=rv_percentile,
            iv_skewness=iv_skewness,
            iv_kurtosis=iv_kurtosis,
        )
    
    def compute_rolling(
        self,
        implied_vols: np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute rolling vol-of-vol.
        
        Parameters
        ----------
        implied_vols : ndarray
            Time series of implied volatilities.
        window : int, optional
            Rolling window size.
        
        Returns
        -------
        ndarray
            Rolling vol-of-vol series.
        """
        window = window or self.window
        iv = np.asarray(implied_vols)
        n = len(iv)
        
        rolling_vov = np.full(n, np.nan)
        
        for i in range(window, n):
            window_iv = iv[i - window:i]
            iv_returns = np.diff(np.log(np.maximum(window_iv, 1e-8)))
            rolling_vov[i] = np.std(iv_returns) * np.sqrt(self.annualization)
        
        return rolling_vov
    
    def detect_regime_changes(
        self,
        implied_vols: np.ndarray,
        min_duration: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Detect volatility regime changes.
        
        Parameters
        ----------
        implied_vols : ndarray
            Time series of implied volatilities.
        min_duration : int
            Minimum duration for a regime.
        
        Returns
        -------
        list of dict
            Regime change events.
        """
        iv = np.asarray(implied_vols)
        regimes = []
        events = []
        
        # Classify each point
        for i, v in enumerate(iv):
            regime, _ = self._classify_regime(v, iv[:i + 1] if i > 0 else iv)
            regimes.append(regime)
        
        # Find regime changes
        current_regime = regimes[0]
        regime_start = 0
        
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # Potential regime change
                if i - regime_start >= min_duration:
                    events.append({
                        "index": i,
                        "from_regime": current_regime,
                        "to_regime": regimes[i],
                        "duration": i - regime_start,
                    })
                current_regime = regimes[i]
                regime_start = i
        
        return events
    
    def _estimate_persistence(self, vol_series: np.ndarray) -> float:
        """Estimate AR(1) persistence coefficient."""
        if len(vol_series) < 3:
            return 0.0
        
        # Simple AR(1): vol_t = a + b * vol_{t-1} + e_t
        y = vol_series[1:]
        x = vol_series[:-1]
        
        # OLS for coefficient b
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator < 1e-10:
            return 0.0
        
        return float(np.clip(numerator / denominator, -1, 1))
    
    def _estimate_mean_reversion(self, vol_series: np.ndarray) -> float:
        """Estimate mean reversion half-life."""
        persistence = self._estimate_persistence(vol_series)
        
        if persistence >= 1.0 or persistence <= 0:
            return float("inf")
        
        # Half-life = -log(2) / log(persistence)
        half_life = -np.log(2) / np.log(abs(persistence))
        
        return float(half_life)
    
    def _classify_regime(
        self,
        current_vol: float,
        historical_vols: np.ndarray,
    ) -> Tuple[str, float]:
        """Classify current volatility regime."""
        if current_vol >= self.regime_thresholds["crisis"]:
            return "crisis", min(1.0, (current_vol - self.regime_thresholds["crisis"]) / 0.1)
        elif current_vol >= self.regime_thresholds["high"]:
            return "high", (current_vol - self.regime_thresholds["high"]) / \
                   (self.regime_thresholds["crisis"] - self.regime_thresholds["high"])
        elif current_vol <= self.regime_thresholds["low"]:
            return "low", (self.regime_thresholds["low"] - current_vol) / self.regime_thresholds["low"]
        else:
            return "normal", 0.5
    
    def _compute_realized_vol(
        self,
        prices: np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """Compute realized volatility from prices."""
        window = window or self.window
        
        returns = np.diff(np.log(np.maximum(prices, 1e-8)))
        n = len(returns)
        
        rv = np.zeros(n + 1)
        for i in range(window, n + 1):
            rv[i] = np.std(returns[i - window:i]) * np.sqrt(self.annualization)
        
        rv[:window] = rv[window] if n >= window else np.mean(np.abs(returns)) * np.sqrt(self.annualization)
        
        return rv
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness."""
        if len(x) < 3:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        if len(x) < 4:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3)


__all__ = [
    "VolOfVolAnalyzer",
    "VolOfVolMetrics",
]
