# Backtesting Framework - Technical Reference

**Module:** `src.backtesting`

This document provides technical details on the backtesting infrastructure for strategy evaluation.

---

## Overview

The backtesting framework enables systematic evaluation of trading strategies against historical data. It provides:

- **BacktestEngine**: Core engine for running backtests
- **Data Providers**: Historical data loading (CSV, in-memory)
- **Performance Metrics**: Sharpe, Sortino, drawdown, Calmar
- **P&L Attribution**: Greeks-based P&L decomposition

---

## Architecture

```
src/backtesting/
├── core/
│   ├── engine.py       # BacktestEngine, BacktestResult
│   └── metrics.py      # PerformanceMetrics, compute_*
├── data/
│   └── providers.py    # HistoricalDataProvider, CsvDataProvider
└── attribution/
    └── pnl.py          # PnLBreakdown, attribute_pnl_to_greeks
```

---

## Core Components

### BacktestEngine

The `BacktestEngine` orchestrates backtest execution:

```python
@dataclass(frozen=True, slots=True)
class BacktestConfig:
    transaction_cost: float = 0.0      # As fraction (0.001 = 10bps)
    slippage: float = 0.0              # As fraction of price
    risk_free_rate: float = 0.0        # Annual, for Sharpe
    periods_per_year: int = 252        # Trading days
    allow_short: bool = True
    verbose: bool = False

class BacktestEngine:
    def run(
        self,
        strategy: StrategyFunc,
        data_provider: DataProvider,
        initial_capital: float,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        price_func: Callable | None = None,
    ) -> BacktestResult
```

#### Execution Flow

1. **Initialization**: Load dates, create portfolio state
2. **For each date**:
   - Update position market values
   - Record portfolio state
   - Execute strategy to get orders
   - Process orders (apply costs/slippage)
3. **Finalization**: Compute returns and metrics

### Strategy Interface

Strategies are simple callables:

```python
def strategy(
    market: MarketSnapshot,
    portfolio: PortfolioState,
    context: BacktestContext,
) -> Sequence[Order]
```

- **market**: Current market data snapshot
- **portfolio**: Current portfolio state (cash, positions)
- **context**: Step number, date, user data

### BacktestResult

Contains all backtest outputs:

```python
@dataclass(frozen=True, slots=True)
class BacktestResult:
    dates: List[date]                # All backtest dates
    portfolio_values: np.ndarray     # Value at each date
    returns: np.ndarray              # Period returns
    cash_series: np.ndarray          # Cash at each date
    position_values: np.ndarray      # Position value at each date
    metrics: PerformanceMetrics      # Computed metrics
    trades: List[Dict]               # Trade records
    config: BacktestConfig
    initial_capital: float
    final_value: float
    
    # Properties
    cumulative_returns: np.ndarray   # Cumulative return series
    drawdown_series: np.ndarray      # Drawdown at each point
```

---

## Performance Metrics

### Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| **Sharpe Ratio** | (E[R] - Rf) / σ(R) × √252 | Risk-adjusted return |
| **Sortino Ratio** | (E[R] - Rf) / σ_down(R) × √252 | Downside-adjusted return |
| **Max Drawdown** | max(peak - trough) / peak | Worst peak-to-trough decline |
| **Calmar Ratio** | Ann. Return / Max DD | Return per unit drawdown |
| **Win Rate** | count(R > 0) / n | Fraction of winning periods |
| **Profit Factor** | Σ(wins) / |Σ(losses)| | Gross profit / gross loss |

### PerformanceMetrics Class

```python
@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    best_return: float
    worst_return: float
    avg_return: float
    avg_win: float
    avg_loss: float
    risk_free_rate: float
    periods_per_year: int
```

### Computation Functions

```python
def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float

def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float

def compute_max_drawdown(
    returns: np.ndarray,
) -> tuple[float, int, int, int]  # (max_dd, duration, peak_idx, trough_idx)

def compute_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float

def compute_all_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics
```

---

## Data Providers

### HistoricalDataProvider Protocol

```python
class HistoricalDataProvider(ABC):
    @abstractmethod
    def get_dates(self) -> Sequence[date]: ...
    
    @abstractmethod
    def get_snapshot(self, dt: date) -> SimpleMarketSnapshot: ...
    
    @property
    def start_date(self) -> date: ...
    
    @property
    def end_date(self) -> date: ...
```

### Implementations

#### DictDataProvider

In-memory data for testing:

```python
data = {
    date(2024, 1, 1): {"AAPL": 150.0, "GOOGL": 140.0},
    date(2024, 1, 2): {"AAPL": 152.0, "GOOGL": 142.0},
}
provider = DictDataProvider(data)
```

#### CsvDataProvider

Load from CSV files:

```python
# Wide format: date,AAPL,GOOGL,...
provider = CsvDataProvider("prices.csv", format="wide")

# Long format: date,instrument,price
provider = CsvDataProvider("prices.csv", format="long")
```

### SimpleMarketSnapshot

```python
@dataclass
class SimpleMarketSnapshot:
    asof: date
    prices: Dict[str, float]
    data: Dict[str, Any]
    
    def get_price(self, instrument_id: str) -> float
    def get(self, key: str, default: Any = None) -> Any
```

### Integration with marketdata

Backtesting data loading is implemented via **marketdata/providers**. This keeps a single place for historical data and lets backtests use the same provider interface as pricing pipelines.

- **HistoricalProvider** (`src.marketdata.providers.historical`): Loads price series from dict or CSV and implements `MarketDataProvider` (get_market, get_timeseries). Produces quote-only `Market` snapshots.
- **BacktestDataAdapter** (`src.backtesting.data.adapter`): Wraps any `MarketDataProvider` so `BacktestEngine` can use it. Exposes `get_dates()` and `get_snapshot(date)` by calling `get_market(asof=date)` and wrapping the `Market` in a snapshot with `get_price(symbol)`.

**Flow:**
- `DictDataProvider(data)` and `CsvDataProvider(path)` build `HistoricalProvider` + `BacktestDataAdapter` under the hood.
- To use **StaticProvider** or **SyntheticProvider** with backtesting: `BacktestDataAdapter(provider=static_or_synthetic_provider, universe=...)`.

**Symbols:** Strategies use string ids (e.g. `"AAPL"`). The adapter maps them to `MarketId` via `MarketId.name` by default, or you can pass `symbol_to_mid` to `BacktestDataAdapter`.

---

## P&L Attribution

### Greeks-Based Attribution

P&L is decomposed using Taylor expansion:

```
P&L ≈ Δ × dS + ½Γ × dS² + Θ × dt + ν × dσ + ρ × dr + residual
```

### PnLBreakdown

```python
@dataclass(frozen=True, slots=True)
class PnLBreakdown:
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    residual: float
    
    @property
    def explained_pnl(self) -> float
    
    @property
    def explanation_ratio(self) -> float
```

### Attribution Function

```python
def attribute_pnl_to_greeks(
    pnl: float,
    delta: float = 0.0,
    gamma: float = 0.0,
    theta: float = 0.0,
    vega: float = 0.0,
    rho: float = 0.0,
    spot_move: float = 0.0,
    vol_move: float = 0.0,
    rate_move: float = 0.0,
    dt: float = 1/252,
) -> PnLBreakdown
```

### Aggregation

```python
def aggregate_attribution(
    attribution: PnLAttribution,
    frequency: str = "weekly",  # or "monthly"
) -> PnLAttribution
```

---

## Portfolio Management

### PortfolioState

```python
@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position]
    
    @property
    def total_value(self) -> float
    
    def get_quantity(self, instrument_id: str) -> float
```

### Position

```python
@dataclass
class Position:
    instrument_id: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    
    def update_market_value(self, current_price: float) -> None
```

---

## Transaction Costs

### Cost Model

Total execution cost = trade_value × (transaction_cost + slippage)

- **transaction_cost**: Explicit fees (commission, exchange fees)
- **slippage**: Market impact (bid-ask spread, price movement)

### Application

```python
if quantity > 0:  # Buy
    exec_price = price * (1 + slippage)
else:  # Sell
    exec_price = price * (1 - slippage)

cost = abs(quantity * exec_price) * transaction_cost
```

---

## Design Decisions

### 1. Strategy as Callable

Strategies are simple functions rather than classes:
- Easier to write quick prototypes
- No boilerplate inheritance
- State managed via `context.user_data`

### 2. Immutable Results

`BacktestResult` is frozen dataclass:
- Thread-safe
- Prevents accidental mutation
- Clear separation of run vs. analysis

### 3. Protocol-Based Data Loading

Data providers follow a protocol:
- Easy to add new sources (database, API)
- Testable with mock providers
- No tight coupling to file formats

### 4. Period-Agnostic Metrics

Metrics accept `periods_per_year` parameter:
- Works for daily, weekly, monthly data
- Consistent annualization
- Flexible for different asset classes

---

## Testing

The framework includes 65 unit tests covering:

- Performance metric calculations
- Drawdown computation
- Engine execution
- Data provider implementations
- P&L attribution

Run tests:
```bash
pytest tests/unit/backtesting/ -v
```

---

*See also: [User Guide](../../guides/backtesting/backtesting_framework.md) | [Tutorial](../../tutorials/backtesting/backtesting_introduction.ipynb)*
