"""
Backtest Engine.

This module provides the core backtesting infrastructure for evaluating
trading strategies against historical data.

The BacktestEngine:
- Replays historical market data
- Executes strategy logic at each timestep
- Tracks portfolio value and positions
- Computes performance metrics

Example
-------
>>> from src.backtesting.core import BacktestEngine, BacktestConfig
>>> from src.backtesting.data import HistoricalDataProvider
>>>
>>> # Define strategy
>>> def my_strategy(market, portfolio, context):
...     # Trading logic here
...     return orders
>>>
>>> # Run backtest
>>> engine = BacktestEngine(config=BacktestConfig())
>>> result = engine.run(
...     strategy=my_strategy,
...     data_provider=historical_data,
...     initial_capital=1_000_000,
... )
>>> print(result.metrics)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

from src.backtesting.core.metrics import PerformanceMetrics, compute_all_metrics


# =============================================================================
# Type Definitions
# =============================================================================

DateLike = Union[str, date, datetime]


class MarketSnapshot(Protocol):
    """Protocol for market data at a point in time."""
    
    @property
    def asof(self) -> date:
        """The date of this snapshot."""
        ...


class DataProvider(Protocol):
    """Protocol for historical data providers."""
    
    def get_dates(self) -> Sequence[date]:
        """Return all available dates in order."""
        ...
    
    def get_snapshot(self, dt: date) -> MarketSnapshot:
        """Return market snapshot for a given date."""
        ...


class Order(Protocol):
    """Protocol for trade orders."""
    
    @property
    def instrument_id(self) -> str:
        """Identifier for the instrument."""
        ...
    
    @property
    def quantity(self) -> float:
        """Signed quantity (positive = buy, negative = sell)."""
        ...


# Strategy function type
StrategyFunc = Callable[
    [MarketSnapshot, "PortfolioState", "BacktestContext"],
    Sequence[Order]
]


# =============================================================================
# Portfolio State
# =============================================================================

@dataclass
class Position:
    """A single position in an instrument."""
    
    instrument_id: str
    quantity: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L."""
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = self.quantity * (current_price - self.avg_price)


@dataclass
class PortfolioState:
    """
    Current state of the portfolio during backtest.
    
    Attributes
    ----------
    cash : float
        Available cash.
    positions : dict
        Map of instrument_id -> Position.
    total_value : float
        Total portfolio value (cash + positions).
    """
    
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def get_position(self, instrument_id: str) -> Optional[Position]:
        """Get position for an instrument, or None if not held."""
        return self.positions.get(instrument_id)
    
    def get_quantity(self, instrument_id: str) -> float:
        """Get quantity for an instrument (0 if not held)."""
        pos = self.positions.get(instrument_id)
        return pos.quantity if pos else 0.0


@dataclass
class BacktestContext:
    """
    Context passed to strategy at each step.
    
    Attributes
    ----------
    current_date : date
        Current backtest date.
    step : int
        Current step number (0-indexed).
    total_steps : int
        Total number of steps in backtest.
    user_data : dict
        User-defined data persisted across steps.
    """
    
    current_date: date
    step: int
    total_steps: int
    user_data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """
    Configuration for backtest execution.
    
    Parameters
    ----------
    transaction_cost : float
        Transaction cost per trade (as fraction of trade value).
        E.g., 0.001 = 0.1% = 10 bps.
    slippage : float
        Slippage per trade (as fraction of price).
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation.
    periods_per_year : int
        Number of trading periods per year.
    initial_margin : float
        Initial margin requirement (1.0 = fully funded).
    allow_short : bool
        Whether short selling is allowed.
    verbose : bool
        Print progress during backtest.
    """
    
    transaction_cost: float = 0.0
    slippage: float = 0.0
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    initial_margin: float = 1.0
    allow_short: bool = True
    verbose: bool = False


# =============================================================================
# Backtest Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class BacktestResult:
    """
    Result of a backtest run.
    
    Attributes
    ----------
    dates : list[date]
        All backtest dates.
    portfolio_values : np.ndarray
        Portfolio value at each date.
    returns : np.ndarray
        Period returns.
    cash_series : np.ndarray
        Cash at each date.
    position_values : np.ndarray
        Total position value at each date.
    metrics : PerformanceMetrics
        Computed performance metrics.
    trades : list[dict]
        Record of all trades executed.
    config : BacktestConfig
        Configuration used.
    initial_capital : float
        Starting capital.
    final_value : float
        Ending portfolio value.
    """
    
    dates: List[date]
    portfolio_values: np.ndarray
    returns: np.ndarray
    cash_series: np.ndarray
    position_values: np.ndarray
    metrics: PerformanceMetrics
    trades: List[Dict[str, Any]]
    config: BacktestConfig
    initial_capital: float
    final_value: float
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"BacktestResult\n"
            f"  Period: {self.dates[0]} to {self.dates[-1]}\n"
            f"  Initial Capital: ${self.initial_capital:,.2f}\n"
            f"  Final Value:     ${self.final_value:,.2f}\n"
            f"  Total Return:    {self.metrics.total_return:+.2%}\n"
            f"  Sharpe Ratio:    {self.metrics.sharpe_ratio:.2f}\n"
            f"  Max Drawdown:    {self.metrics.max_drawdown:.2%}\n"
            f"  Trades:          {len(self.trades)}"
        )
    
    @property
    def cumulative_returns(self) -> np.ndarray:
        """Cumulative returns series."""
        return np.cumprod(1 + self.returns) - 1
    
    @property
    def drawdown_series(self) -> np.ndarray:
        """Drawdown at each point."""
        cumret = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumret)
        return (running_max - cumret) / running_max


# =============================================================================
# Backtest Engine
# =============================================================================

class BacktestEngine:
    """
    Engine for running backtests.
    
    The engine handles:
    - Market data replay
    - Strategy execution
    - Position and P&L tracking
    - Transaction costs and slippage
    - Performance metric computation
    
    Parameters
    ----------
    config : BacktestConfig
        Backtest configuration.
    
    Examples
    --------
    >>> engine = BacktestEngine(config=BacktestConfig(transaction_cost=0.001))
    >>> result = engine.run(
    ...     strategy=my_strategy,
    ...     data_provider=data,
    ...     initial_capital=1_000_000,
    ... )
    """
    
    def __init__(self, config: BacktestConfig = BacktestConfig()) -> None:
        self.config = config
    
    def run(
        self,
        strategy: StrategyFunc,
        data_provider: DataProvider,
        initial_capital: float,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        price_func: Optional[Callable[[MarketSnapshot, str], float]] = None,
    ) -> BacktestResult:
        """
        Run a backtest.
        
        Parameters
        ----------
        strategy : Callable
            Strategy function: (market, portfolio, context) -> orders.
        data_provider : DataProvider
            Provider of historical market data.
        initial_capital : float
            Starting capital.
        start_date : date-like, optional
            Start date (defaults to first available).
        end_date : date-like, optional
            End date (defaults to last available).
        price_func : Callable, optional
            Function to get price from market snapshot for an instrument.
            Signature: (market, instrument_id) -> price.
            If None, assumes market has a `get_price(id)` method.
        
        Returns
        -------
        BacktestResult
            Complete backtest results.
        """
        # Get date range
        all_dates = list(data_provider.get_dates())
        if not all_dates:
            raise ValueError("Data provider has no dates.")
        
        if start_date:
            start_date = _parse_date(start_date)
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            end_date = _parse_date(end_date)
            all_dates = [d for d in all_dates if d <= end_date]
        
        if not all_dates:
            raise ValueError("No dates in specified range.")
        
        n_steps = len(all_dates)
        
        # Initialize tracking arrays
        portfolio_values = np.zeros(n_steps, dtype=float)
        cash_series = np.zeros(n_steps, dtype=float)
        position_values = np.zeros(n_steps, dtype=float)
        trades: List[Dict[str, Any]] = []
        
        # Initialize portfolio
        portfolio = PortfolioState(cash=initial_capital)
        
        # Default price function
        if price_func is None:
            def price_func(market: MarketSnapshot, inst_id: str) -> float:
                return float(getattr(market, "get_price")(inst_id))
        
        # Run backtest
        for step, dt in enumerate(all_dates):
            market = data_provider.get_snapshot(dt)
            context = BacktestContext(
                current_date=dt,
                step=step,
                total_steps=n_steps,
            )
            
            # Update position market values
            for pos in portfolio.positions.values():
                try:
                    price = price_func(market, pos.instrument_id)
                    pos.update_market_value(price)
                except Exception:
                    pass  # Keep previous market value if price unavailable
            
            # Record portfolio state before trading
            portfolio_values[step] = portfolio.total_value
            cash_series[step] = portfolio.cash
            position_values[step] = sum(p.market_value for p in portfolio.positions.values())
            
            # Execute strategy
            try:
                orders = strategy(market, portfolio, context)
            except Exception as e:
                if self.config.verbose:
                    print(f"Strategy error at {dt}: {e}")
                orders = []
            
            # Process orders
            for order in orders or []:
                try:
                    trade = self._execute_order(order, market, portfolio, price_func, dt)
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Order execution error: {e}")
            
            if self.config.verbose and step % 50 == 0:
                print(f"Step {step}/{n_steps}: {dt} | Value: ${portfolio.total_value:,.2f}")
        
        # Compute returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0.0], returns])  # First return is 0
        
        # Compute metrics
        metrics = compute_all_metrics(
            returns[1:],  # Exclude first zero return
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.config.periods_per_year,
        )
        
        return BacktestResult(
            dates=all_dates,
            portfolio_values=portfolio_values,
            returns=returns,
            cash_series=cash_series,
            position_values=position_values,
            metrics=metrics,
            trades=trades,
            config=self.config,
            initial_capital=initial_capital,
            final_value=float(portfolio_values[-1]),
        )
    
    def _execute_order(
        self,
        order: Order,
        market: MarketSnapshot,
        portfolio: PortfolioState,
        price_func: Callable[[MarketSnapshot, str], float],
        dt: date,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single order and update portfolio."""
        inst_id = order.instrument_id
        quantity = float(order.quantity)
        
        if abs(quantity) < 1e-12:
            return None
        
        # Get execution price
        try:
            price = price_func(market, inst_id)
        except Exception:
            return None  # Can't price, skip order
        
        # Apply slippage
        if quantity > 0:
            exec_price = price * (1 + self.config.slippage)
        else:
            exec_price = price * (1 - self.config.slippage)
        
        # Check short selling
        current_qty = portfolio.get_quantity(inst_id)
        new_qty = current_qty + quantity
        if new_qty < 0 and not self.config.allow_short:
            return None  # Short not allowed
        
        # Calculate cost
        trade_value = abs(quantity * exec_price)
        transaction_cost = trade_value * self.config.transaction_cost
        
        # For buys, check cash
        if quantity > 0:
            total_cost = quantity * exec_price + transaction_cost
            if total_cost > portfolio.cash:
                return None  # Insufficient funds
            portfolio.cash -= total_cost
        else:
            # For sells, receive proceeds minus costs
            proceeds = abs(quantity) * exec_price - transaction_cost
            portfolio.cash += proceeds
        
        # Update position
        if inst_id in portfolio.positions:
            pos = portfolio.positions[inst_id]
            if new_qty == 0:
                del portfolio.positions[inst_id]
            else:
                # Update average price for adds
                if quantity > 0 and pos.quantity > 0:
                    total_value = pos.quantity * pos.avg_price + quantity * exec_price
                    pos.avg_price = total_value / (pos.quantity + quantity)
                pos.quantity = new_qty
        else:
            if new_qty != 0:
                portfolio.positions[inst_id] = Position(
                    instrument_id=inst_id,
                    quantity=new_qty,
                    avg_price=exec_price,
                    market_value=new_qty * exec_price,
                )
        
        return {
            "date": dt,
            "instrument_id": inst_id,
            "quantity": quantity,
            "price": exec_price,
            "value": trade_value,
            "cost": transaction_cost,
            "side": "buy" if quantity > 0 else "sell",
        }


# =============================================================================
# Utilities
# =============================================================================

def _parse_date(d: DateLike) -> date:
    """Parse various date formats to date object."""
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise ValueError(f"Cannot parse date: {d}")
