# Backtesting Framework - User Guide

This guide demonstrates how to use QuantStrata's backtesting infrastructure for strategy evaluation.

---

## Quick Start

### Basic Example

```python
from datetime import date
from dataclasses import dataclass
from src.backtesting.core import BacktestEngine, BacktestConfig
from src.backtesting.data import DictDataProvider

# 1. Prepare historical data
data = {
    date(2024, 1, 1): {"AAPL": 150.0},
    date(2024, 1, 2): {"AAPL": 152.0},
    date(2024, 1, 3): {"AAPL": 151.0},
    date(2024, 1, 4): {"AAPL": 155.0},
    date(2024, 1, 5): {"AAPL": 158.0},
}
provider = DictDataProvider(data)

# 2. Define a simple order
@dataclass
class Order:
    instrument_id: str
    quantity: float

# 3. Define strategy
def buy_and_hold(market, portfolio, context):
    if context.step == 0:
        # Buy 100 shares on first day
        return [Order("AAPL", 100)]
    return []

# 4. Run backtest
engine = BacktestEngine(config=BacktestConfig())
result = engine.run(
    strategy=buy_and_hold,
    data_provider=provider,
    initial_capital=100_000,
)

# 5. View results
print(result)
print(result.metrics)
```

---

## Configuration

### BacktestConfig

```python
from src.backtesting.core import BacktestConfig

config = BacktestConfig(
    transaction_cost=0.001,    # 0.1% = 10 bps per trade
    slippage=0.0005,           # 0.05% slippage
    risk_free_rate=0.02,       # 2% annual for Sharpe
    periods_per_year=252,      # Trading days
    allow_short=True,          # Allow short selling
    verbose=True,              # Print progress
)

engine = BacktestEngine(config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `transaction_cost` | 0.0 | Cost per trade as fraction |
| `slippage` | 0.0 | Price impact as fraction |
| `risk_free_rate` | 0.0 | Annual rate for Sharpe |
| `periods_per_year` | 252 | Periods for annualization |
| `allow_short` | True | Allow short positions |
| `verbose` | False | Print progress |

---

## Writing Strategies

### Strategy Function Signature

```python
def my_strategy(market, portfolio, context):
    """
    Parameters
    ----------
    market : SimpleMarketSnapshot
        Current market data.
    portfolio : PortfolioState
        Current portfolio (cash, positions).
    context : BacktestContext
        step, current_date, total_steps, user_data.
    
    Returns
    -------
    list of Order
        Orders to execute.
    """
    return []
```

### Accessing Market Data

```python
def strategy(market, portfolio, context):
    # Get current price
    price = market.get_price("AAPL")
    
    # Check if instrument exists
    if "GOOGL" in market:
        googl_price = market.get_price("GOOGL")
    
    # Access additional data
    volume = market.get("volume", default=0)
```

### Accessing Portfolio State

```python
def strategy(market, portfolio, context):
    # Check cash
    if portfolio.cash < 10000:
        return []
    
    # Get current position
    aapl_qty = portfolio.get_quantity("AAPL")
    
    # Get position details
    pos = portfolio.get_position("AAPL")
    if pos:
        avg_price = pos.avg_price
        unrealized = pos.unrealized_pnl
    
    # Total portfolio value
    total_value = portfolio.total_value
```

### Using Context for State

```python
def strategy(market, portfolio, context):
    # Initialize state on first step
    if context.step == 0:
        context.user_data["high_water_mark"] = portfolio.total_value
        context.user_data["trades_today"] = 0
    
    # Access persistent state
    hwm = context.user_data["high_water_mark"]
    
    # Update state
    context.user_data["high_water_mark"] = max(hwm, portfolio.total_value)
```

---

## Example Strategies

### Moving Average Crossover

```python
def ma_crossover(market, portfolio, context):
    # Initialize price history
    if "prices" not in context.user_data:
        context.user_data["prices"] = []
    
    prices = context.user_data["prices"]
    price = market.get_price("AAPL")
    prices.append(price)
    
    # Need enough history
    if len(prices) < 20:
        return []
    
    # Compute MAs
    ma_fast = sum(prices[-5:]) / 5
    ma_slow = sum(prices[-20:]) / 20
    
    current_qty = portfolio.get_quantity("AAPL")
    
    # Generate signals
    if ma_fast > ma_slow and current_qty == 0:
        # Buy signal
        shares = int(portfolio.cash * 0.95 / price)
        return [Order("AAPL", shares)]
    elif ma_fast < ma_slow and current_qty > 0:
        # Sell signal
        return [Order("AAPL", -current_qty)]
    
    return []
```

### Mean Reversion

```python
def mean_reversion(market, portfolio, context):
    if "prices" not in context.user_data:
        context.user_data["prices"] = []
    
    prices = context.user_data["prices"]
    price = market.get_price("AAPL")
    prices.append(price)
    
    if len(prices) < 20:
        return []
    
    mean = sum(prices[-20:]) / 20
    std = (sum((p - mean)**2 for p in prices[-20:]) / 20) ** 0.5
    
    z_score = (price - mean) / std if std > 0 else 0
    current_qty = portfolio.get_quantity("AAPL")
    
    if z_score < -2 and current_qty == 0:
        # Price below -2 std: buy
        shares = int(portfolio.cash * 0.5 / price)
        return [Order("AAPL", shares)]
    elif z_score > 0 and current_qty > 0:
        # Reversion to mean: sell
        return [Order("AAPL", -current_qty)]
    
    return []
```

---

## Data Providers

Backtesting uses **marketdata** for historical data. `DictDataProvider` and `CsvDataProvider` delegate to `marketdata.providers.historical.HistoricalProvider`, so loading logic lives in one place. You can also use any `MarketDataProvider` (e.g. Static, Synthetic) via `BacktestDataAdapter`.

### In-Memory (Testing)

```python
from src.backtesting.data import DictDataProvider

data = {
    date(2024, 1, 1): {"AAPL": 150.0, "GOOGL": 140.0},
    date(2024, 1, 2): {"AAPL": 152.0, "GOOGL": 142.0},
}
provider = DictDataProvider(data)
```

### From CSV

```python
from src.backtesting.data import CsvDataProvider

# Wide format: date,AAPL,GOOGL,MSFT
provider = CsvDataProvider(
    "data/prices.csv",
    date_column="date",
    date_format="%Y-%m-%d",
    format="wide",
)

# Long format: date,ticker,close
provider = CsvDataProvider(
    "data/prices_long.csv",
    format="long",
    instrument_column="ticker",
    price_column="close",
)
```

### Using Other marketdata Providers

To backtest against a pre-built dataset or synthetic provider, use `BacktestDataAdapter` with a `Universe` of the MarketIds you need:

```python
from src.backtesting.data import BacktestDataAdapter
from src.marketdata.core.requests import Universe
from src.marketdata.providers.historical import HistoricalProvider

# Example: use HistoricalProvider directly (same as DictDataProvider under the hood)
hp = HistoricalProvider(data={date(2024, 1, 1): {"AAPL": 150.0}})
adapter = BacktestDataAdapter(provider=hp, universe=hp.universe)

# Example: use StaticProvider (replay a MarketDataset) — build universe from dataset panels
# universe = Universe(ids=list(static_provider.dataset.panels.keys()))
# adapter = BacktestDataAdapter(provider=static_provider, universe=universe)

result = engine.run(strategy=my_strategy, data_provider=adapter, initial_capital=100_000)
```

---

## Analyzing Results

### BacktestResult Properties

```python
result = engine.run(...)

# Summary
print(result)  # Human-readable summary

# Key metrics
print(f"Total Return: {result.metrics.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")

# Arrays for plotting
dates = result.dates
values = result.portfolio_values
returns = result.returns
cum_returns = result.cumulative_returns
drawdowns = result.drawdown_series

# Trade history
for trade in result.trades:
    print(f"{trade['date']}: {trade['side']} {trade['quantity']} @ {trade['price']}")
```

### Plotting Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Portfolio value
axes[0].plot(result.dates, result.portfolio_values)
axes[0].set_ylabel("Portfolio Value")
axes[0].set_title("Backtest Results")

# Cumulative returns
axes[1].plot(result.dates, result.cumulative_returns * 100)
axes[1].set_ylabel("Cumulative Return (%)")

# Drawdown
axes[2].fill_between(result.dates, -result.drawdown_series * 100, 0, alpha=0.3, color='red')
axes[2].set_ylabel("Drawdown (%)")
axes[2].set_xlabel("Date")

plt.tight_layout()
plt.show()
```

---

## P&L Attribution

### Single-Period Attribution

```python
from src.backtesting.attribution import attribute_pnl_to_greeks

breakdown = attribute_pnl_to_greeks(
    pnl=1500.0,
    delta=500.0,
    gamma=10.0,
    theta=-100.0,
    vega=5000.0,
    spot_move=2.0,
    vol_move=0.01,
    dt=1/252,
)

print(breakdown)
# Shows delta, gamma, theta, vega contributions + residual
```

### Multi-Period Attribution

```python
from src.backtesting.attribution import PnLAttribution, aggregate_attribution

# Build daily attribution
attribution = PnLAttribution()
for dt, pnl, greeks, moves in daily_data:
    breakdown = attribute_pnl_to_greeks(pnl, **greeks, **moves)
    attribution.add(dt, breakdown)

# View cumulative
print(attribution.cumulative)

# Aggregate to weekly
weekly = aggregate_attribution(attribution, frequency="weekly")
```

---

## Best Practices

### 1. Use Realistic Transaction Costs

```python
# Equity market maker
config = BacktestConfig(transaction_cost=0.0005, slippage=0.0002)

# Retail broker
config = BacktestConfig(transaction_cost=0.001, slippage=0.001)

# Institutional
config = BacktestConfig(transaction_cost=0.0002, slippage=0.0005)
```

### 2. Avoid Look-Ahead Bias

```python
def strategy(market, portfolio, context):
    # WRONG: Using future data
    # future_price = data[context.current_date + timedelta(days=1)]
    
    # CORRECT: Only use current and past data
    current_price = market.get_price("AAPL")
    past_prices = context.user_data.get("prices", [])
```

### 3. Handle Missing Data

```python
def strategy(market, portfolio, context):
    try:
        price = market.get_price("AAPL")
    except KeyError:
        # No data for this date
        return []
```

### 4. Limit Position Sizes

```python
def strategy(market, portfolio, context):
    price = market.get_price("AAPL")
    
    # Don't use more than 10% of portfolio per position
    max_value = portfolio.total_value * 0.10
    max_shares = int(max_value / price)
    
    shares = min(desired_shares, max_shares)
    return [Order("AAPL", shares)]
```

---

## Complete Example

```python
from datetime import date, timedelta
import numpy as np
from dataclasses import dataclass
from src.backtesting.core import BacktestEngine, BacktestConfig
from src.backtesting.data import DictDataProvider

@dataclass
class Order:
    instrument_id: str
    quantity: float

# Generate synthetic data
np.random.seed(42)
dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(252)]
prices = 100 * np.cumprod(1 + np.random.randn(252) * 0.02)
data = {d: {"STOCK": p} for d, p in zip(dates, prices)}

provider = DictDataProvider(data)

# Momentum strategy
def momentum(market, portfolio, context):
    if "prices" not in context.user_data:
        context.user_data["prices"] = []
    
    price = market.get_price("STOCK")
    context.user_data["prices"].append(price)
    
    if len(context.user_data["prices"]) < 10:
        return []
    
    # Buy if price above 10-day high
    recent = context.user_data["prices"][-10:]
    if price >= max(recent) and portfolio.get_quantity("STOCK") == 0:
        shares = int(portfolio.cash * 0.9 / price)
        return [Order("STOCK", shares)]
    
    # Sell if price below 10-day low
    if price <= min(recent) and portfolio.get_quantity("STOCK") > 0:
        return [Order("STOCK", -portfolio.get_quantity("STOCK"))]
    
    return []

# Run backtest
config = BacktestConfig(transaction_cost=0.001, risk_free_rate=0.02)
engine = BacktestEngine(config=config)
result = engine.run(
    strategy=momentum,
    data_provider=provider,
    initial_capital=100_000,
)

print(result)
print("\nPerformance Metrics:")
print(result.metrics)
```

---

*See also: [Technical Reference](../../reference/backtesting/backtesting_framework.md) | [Tutorial](../../tutorials/backtesting/backtesting_introduction.ipynb)*
