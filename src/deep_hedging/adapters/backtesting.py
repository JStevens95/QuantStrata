"""
Backtest Engine Adapter for deep hedging agents.

Bridges deep hedging agents to the backtesting framework,
allowing trained hedging agents to be evaluated on historical data.

Example:
    from src.deep_hedging.adapters import BacktestEngineAdapter
    from src.deep_hedging.agents import DeepHedgingAgent
    
    # Load trained agent
    agent = DeepHedgingAgent.load("path/to/agent")
    
    # Create adapter
    adapter = BacktestEngineAdapter(
        agent=agent,
        config=BacktestConfig(transaction_cost=0.001),
    )
    
    # Run backtest
    result = adapter.run_backtest(
        data_provider=historical_data,
        option_params=option_params,
    )
    
    print(f"Hedging P&L: {result.total_pnl:.2f}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


# =============================================================================
# Protocols
# =============================================================================


class HedgingAgentProtocol(Protocol):
    """Protocol for hedging agents."""
    
    def select_action(
        self,
        state: np.ndarray,
        *,
        training: bool = False,
    ) -> float:
        """Select hedge action given current state."""
        ...
    
    def get_hedge_ratio(
        self,
        spot: float,
        time_to_expiry: float,
        position: float,
        delta: float,
        **kwargs: Any,
    ) -> float:
        """Get hedge ratio for given market state."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BacktestConfig:
    """Configuration for hedging backtest."""
    
    # Transaction costs
    transaction_cost: float = 0.001
    fixed_cost: float = 0.0
    
    # Option parameters
    strike: Optional[float] = None  # Use ATM if None
    maturity_days: int = 30
    option_type: str = "call"
    
    # Hedging frequency
    rehedge_frequency: str = "daily"  # "daily", "weekly", "hourly"
    
    # State features for agent
    include_delta: bool = True
    include_gamma: bool = True
    include_vega: bool = False
    
    # Position limits
    max_position: float = 2.0  # Max position as multiple of delta


@dataclass
class OptionParams:
    """Parameters for the option being hedged."""
    
    strike: float
    maturity: date
    option_type: str = "call"
    notional: float = 1.0


# =============================================================================
# Hedging Strategy Wrapper
# =============================================================================


class HedgingStrategy:
    """
    Wraps a hedging agent as a backtesting strategy.
    
    Implements the strategy interface expected by BacktestEngine,
    translating between backtest context and agent state.
    """
    
    def __init__(
        self,
        agent: Any,  # HedgingAgentProtocol
        option_params: OptionParams,
        config: BacktestConfig,
    ) -> None:
        """
        Initialize hedging strategy.
        
        Parameters
        ----------
        agent : HedgingAgentProtocol
            Trained hedging agent.
        option_params : OptionParams
            Option being hedged.
        config : BacktestConfig
            Backtest configuration.
        """
        self.agent = agent
        self.option_params = option_params
        self.config = config
        
        # Tracking
        self._initial_spot: Optional[float] = None
        self._current_position: float = 0.0
        self._pnl_history: List[float] = []
        self._position_history: List[float] = []
        self._cost_history: List[float] = []
    
    def on_start(
        self,
        initial_spot: float,
        initial_date: date,
    ) -> Dict[str, Any]:
        """Called at backtest start."""
        self._initial_spot = initial_spot
        self._current_position = 0.0
        self._pnl_history = []
        self._position_history = []
        self._cost_history = []
        
        return {"initial_spot": initial_spot}
    
    def on_data(
        self,
        spot: float,
        current_date: date,
        volatility: float,
        risk_free_rate: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Process market data and return hedge action.
        
        Parameters
        ----------
        spot : float
            Current spot price.
        current_date : date
            Current date.
        volatility : float
            Implied or historical volatility.
        risk_free_rate : float
            Risk-free rate.
        
        Returns
        -------
        trade_qty : float
            Trade quantity (positive = buy, negative = sell).
        info : dict
            Additional information.
        """
        # Calculate time to expiry
        time_to_expiry = (self.option_params.maturity - current_date).days / 365.0
        
        if time_to_expiry <= 0:
            # Option expired, close position
            trade_qty = -self._current_position
            return trade_qty, {"reason": "expiry"}
        
        # Calculate delta for state
        delta = self._calculate_delta(
            spot=spot,
            strike=self.option_params.strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
        )
        
        # Build state for agent
        state = self._build_state(
            spot=spot,
            time_to_expiry=time_to_expiry,
            position=self._current_position,
            delta=delta,
            volatility=volatility,
        )
        
        # Get hedge action from agent
        if hasattr(self.agent, "select_action"):
            # RL-style agent
            hedge_ratio = self.agent.select_action(state, training=False)
        elif hasattr(self.agent, "get_hedge_ratio"):
            # Simple agent interface
            hedge_ratio = self.agent.get_hedge_ratio(
                spot=spot,
                time_to_expiry=time_to_expiry,
                position=self._current_position,
                delta=delta,
            )
        else:
            # Fallback to delta hedging
            hedge_ratio = 1.0
        
        # Convert hedge ratio to target position
        target_position = hedge_ratio * abs(delta) * self.option_params.notional
        
        # Apply position limits
        target_position = np.clip(
            target_position,
            -self.config.max_position * abs(delta),
            self.config.max_position * abs(delta),
        )
        
        # Calculate trade
        trade_qty = target_position - self._current_position
        
        # Update tracking
        self._current_position = target_position
        self._position_history.append(target_position)
        
        info = {
            "delta": delta,
            "hedge_ratio": hedge_ratio,
            "target_position": target_position,
            "time_to_expiry": time_to_expiry,
        }
        
        return trade_qty, info
    
    def on_trade(
        self,
        trade_qty: float,
        trade_price: float,
        cost: float,
    ) -> None:
        """Record trade execution."""
        self._cost_history.append(cost)
    
    def _build_state(
        self,
        spot: float,
        time_to_expiry: float,
        position: float,
        delta: float,
        volatility: float,
    ) -> np.ndarray:
        """Build state vector for agent."""
        features = [
            spot / self._initial_spot - 1.0,  # Normalized spot
            time_to_expiry,
        ]
        
        if self.config.include_delta:
            features.append(delta)
        
        if self.config.include_gamma:
            gamma = self._calculate_gamma(
                spot=spot,
                strike=self.option_params.strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
            )
            features.append(gamma * spot)
        
        features.append(position / (abs(delta) + 1e-8))  # Position relative to delta
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_delta(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Black-Scholes delta."""
        if time_to_expiry <= 0:
            if self.option_params.option_type == "call":
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0
        
        from scipy.stats import norm
        
        d1 = (
            np.log(spot / strike) + 
            (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))
        
        if self.option_params.option_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    
    def _calculate_gamma(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Black-Scholes gamma."""
        if time_to_expiry <= 0:
            return 0.0
        
        from scipy.stats import norm
        
        d1 = (
            np.log(spot / strike) + 
            (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))
        
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
        
        return float(gamma)


# =============================================================================
# Backtest Result
# =============================================================================


@dataclass
class HedgingBacktestResult:
    """Result from hedging backtest."""
    
    # P&L
    total_pnl: float
    hedging_pnl: float
    option_pnl: float
    total_cost: float
    
    # Statistics
    mean_daily_pnl: float
    std_daily_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Position tracking
    mean_position: float
    max_position: float
    
    # Comparison (if delta hedge benchmark)
    benchmark_pnl: Optional[float] = None
    outperformance: Optional[float] = None
    
    # History
    pnl_history: List[float] = field(default_factory=list)
    position_history: List[float] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "hedging_pnl": self.hedging_pnl,
            "option_pnl": self.option_pnl,
            "total_cost": self.total_cost,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "outperformance": self.outperformance,
        }


# =============================================================================
# Backtest Engine Adapter
# =============================================================================


class BacktestEngineAdapter:
    """
    Adapter for running hedging agents through backtesting framework.
    
    Bridges deep hedging agents to the backtesting infrastructure,
    enabling evaluation on historical data.
    
    Example:
        adapter = BacktestEngineAdapter(
            agent=trained_agent,
            config=BacktestConfig(transaction_cost=0.001),
        )
        
        result = adapter.run_backtest(
            prices=historical_prices,
            volatilities=historical_vols,
            option_params=OptionParams(strike=100, maturity=date(2024, 6, 30)),
        )
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        """
        Initialize adapter.
        
        Parameters
        ----------
        agent : HedgingAgentProtocol
            Trained hedging agent.
        config : BacktestConfig, optional
            Backtest configuration.
        """
        self.agent = agent
        self.config = config or BacktestConfig()
    
    def run_backtest(
        self,
        prices: np.ndarray,
        volatilities: np.ndarray,
        dates: Optional[Sequence[date]] = None,
        option_params: Optional[OptionParams] = None,
        risk_free_rate: float = 0.05,
        run_benchmark: bool = True,
    ) -> HedgingBacktestResult:
        """
        Run hedging backtest.
        
        Parameters
        ----------
        prices : ndarray
            Historical price series.
        volatilities : ndarray
            Historical volatility series (same length as prices).
        dates : sequence of date, optional
            Dates corresponding to prices.
        option_params : OptionParams, optional
            Option parameters. If None, creates ATM option.
        risk_free_rate : float
            Risk-free rate.
        run_benchmark : bool
            Run delta hedging benchmark for comparison.
        
        Returns
        -------
        HedgingBacktestResult
            Backtest results.
        """
        n_steps = len(prices)
        
        # Generate dates if not provided
        if dates is None:
            from datetime import timedelta
            base_date = date.today()
            dates = [base_date + timedelta(days=i) for i in range(n_steps)]
        
        # Create option params if not provided
        if option_params is None:
            option_params = OptionParams(
                strike=prices[0],  # ATM
                maturity=dates[min(self.config.maturity_days, n_steps - 1)],
            )
        
        # Create strategy
        strategy = HedgingStrategy(
            agent=self.agent,
            option_params=option_params,
            config=self.config,
        )
        
        # Run backtest
        pnl_history, position_history, cost_history = self._run_simulation(
            strategy=strategy,
            prices=prices,
            volatilities=volatilities,
            dates=list(dates),
            risk_free_rate=risk_free_rate,
        )
        
        # Compute option P&L
        final_payoff = max(prices[-1] - option_params.strike, 0) \
            if option_params.option_type == "call" \
            else max(option_params.strike - prices[-1], 0)
        
        initial_option_value = self._price_option(
            spot=prices[0],
            strike=option_params.strike,
            T=(option_params.maturity - dates[0]).days / 365.0,
            vol=volatilities[0],
            r=risk_free_rate,
            option_type=option_params.option_type,
        )
        
        option_pnl = -(final_payoff - initial_option_value)  # Short option
        hedging_pnl = sum(pnl_history)
        total_cost = sum(cost_history)
        total_pnl = hedging_pnl + option_pnl - total_cost
        
        # Statistics
        pnl_array = np.array(pnl_history)
        sharpe = self._compute_sharpe(pnl_array)
        max_dd = self._compute_max_drawdown(pnl_array)
        
        # Run benchmark if requested
        benchmark_pnl = None
        outperformance = None
        if run_benchmark:
            benchmark_pnl = self._run_delta_hedge_benchmark(
                prices=prices,
                volatilities=volatilities,
                dates=list(dates),
                option_params=option_params,
                risk_free_rate=risk_free_rate,
            )
            outperformance = total_pnl - benchmark_pnl
        
        return HedgingBacktestResult(
            total_pnl=total_pnl,
            hedging_pnl=hedging_pnl,
            option_pnl=option_pnl,
            total_cost=total_cost,
            mean_daily_pnl=float(np.mean(pnl_array)),
            std_daily_pnl=float(np.std(pnl_array)),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            mean_position=float(np.mean(np.abs(position_history))),
            max_position=float(np.max(np.abs(position_history))),
            benchmark_pnl=benchmark_pnl,
            outperformance=outperformance,
            pnl_history=pnl_history,
            position_history=position_history,
        )
    
    def _run_simulation(
        self,
        strategy: HedgingStrategy,
        prices: np.ndarray,
        volatilities: np.ndarray,
        dates: List[date],
        risk_free_rate: float,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Run simulation and collect results."""
        strategy.on_start(prices[0], dates[0])
        
        pnl_history: List[float] = []
        position_history: List[float] = []
        cost_history: List[float] = []
        
        position = 0.0
        prev_price = prices[0]
        
        for i in range(1, len(prices)):
            spot = prices[i]
            vol = volatilities[i] if i < len(volatilities) else volatilities[-1]
            
            # Get hedge action
            trade_qty, info = strategy.on_data(
                spot=spot,
                current_date=dates[i],
                volatility=vol,
                risk_free_rate=risk_free_rate,
            )
            
            # Compute P&L from position
            price_change = spot - prev_price
            pnl = position * price_change
            pnl_history.append(pnl)
            
            # Execute trade
            cost = (
                abs(trade_qty * spot) * self.config.transaction_cost +
                self.config.fixed_cost * (1 if abs(trade_qty) > 1e-8 else 0)
            )
            
            position += trade_qty
            position_history.append(position)
            cost_history.append(cost)
            
            strategy.on_trade(trade_qty, spot, cost)
            
            prev_price = spot
        
        return pnl_history, position_history, cost_history
    
    def _run_delta_hedge_benchmark(
        self,
        prices: np.ndarray,
        volatilities: np.ndarray,
        dates: List[date],
        option_params: OptionParams,
        risk_free_rate: float,
    ) -> float:
        """Run delta hedging benchmark."""
        from src.deep_hedging.agents.delta import DeltaHedgingAgent
        
        delta_agent = DeltaHedgingAgent()
        
        adapter = BacktestEngineAdapter(
            agent=delta_agent,
            config=self.config,
        )
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
            dates=dates,
            option_params=option_params,
            risk_free_rate=risk_free_rate,
            run_benchmark=False,
        )
        
        return result.total_pnl
    
    def _price_option(
        self,
        spot: float,
        strike: float,
        T: float,
        vol: float,
        r: float,
        option_type: str,
    ) -> float:
        """Price option using Black-Scholes."""
        if T <= 0:
            if option_type == "call":
                return max(spot - strike, 0)
            else:
                return max(strike - spot, 0)
        
        from scipy.stats import norm
        
        d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        if option_type == "call":
            return spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:
            return strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    def _compute_sharpe(self, pnl: np.ndarray, annualization: float = 252) -> float:
        """Compute Sharpe ratio."""
        if len(pnl) < 2:
            return 0.0
        mean_ret = np.mean(pnl)
        std_ret = np.std(pnl)
        if std_ret < 1e-8:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(annualization))
    
    def _compute_max_drawdown(self, pnl: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        if running_max.max() < 1e-8:
            return 0.0
        return float(np.max(drawdowns) / (running_max.max() + 1e-8))


__all__ = [
    "BacktestEngineAdapter",
    "HedgingStrategy",
    "BacktestConfig",
    "OptionParams",
    "HedgingBacktestResult",
]
