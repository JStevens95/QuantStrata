# Volatility Trading Reference

Technical specification and API reference for volatility trading components.

## Overview

The volatility trading module provides:
- **Variance Swaps**: Pricing via log-strip replication
- **Dispersion Trading**: Index vs. constituent volatility analysis
- **Vol-of-Vol Analytics**: Volatility regime detection and metrics

---

## Variance Swaps

### Module: `src.volatility.trading.variance_swap`

### VarianceSwap Instrument

```python
from src.volatility.trading import VarianceSwap

# Create from variance terms
swap = VarianceSwap(
    strike_var=0.04,          # Strike variance (20% vol squared)
    maturity=0.5,             # Time to maturity in years
    notional=100_000,         # Variance notional
    observation_frequency="daily",
)

# Properties
swap.strike_vol      # √0.04 = 0.20 (20%)
swap.vega_notional   # notional * 2 * strike_vol

# Create from volatility terms
swap = VarianceSwap.from_vol(
    strike_vol=0.20,
    maturity=0.5,
    vega_notional=40_000,
)
```

### VarianceSwapPricer

Price variance swaps using log-strip replication (Carr-Madan).

```python
from src.volatility.trading import VarianceSwapPricer

pricer = VarianceSwapPricer(n_integration_points=100)

result = pricer.price(
    swap=swap,
    spot=100.0,
    forward=101.0,
    option_strikes=strikes,      # Array of option strikes
    option_prices_call=calls,    # OTM call prices
    option_prices_put=puts,      # OTM put prices
    rate=0.05,
)
```

### VarianceSwapResult

```python
@dataclass
class VarianceSwapResult:
    fair_variance: float        # Replication-implied variance
    strike_variance: float      # Trade strike variance
    mtm: float                  # Mark-to-market value
    notional: float
    maturity: float
    
    @property
    def fair_vol(self) -> float:
        """Fair volatility (√fair_variance)."""
        return np.sqrt(self.fair_variance)
```

### Realized Variance

```python
from src.volatility.trading.variance_swap import calculate_realized_variance

# From price series
realized_var = calculate_realized_variance(
    prices=price_array,
    annualization=252,    # Trading days per year
)

realized_vol = np.sqrt(realized_var)
```

---

## Dispersion Trading

### Module: `src.volatility.trading.dispersion`

### DispersionTrader

Analyze dispersion opportunities (index vs. constituent volatility).

```python
from src.volatility.trading import DispersionTrader, DispersionConfig

# Configuration
config = DispersionConfig(
    min_spread=0.01,          # Minimum attractive spread
    target_vega_neutral=0.9,  # Vega neutrality target
)

# Create trader
trader = DispersionTrader(
    index_ticker="SPX",
    constituents=["AAPL", "MSFT", "GOOGL", "AMZN"],
    weights=np.array([0.3, 0.25, 0.25, 0.2]),  # Market cap weights
    config=config,
)

# Analyze opportunity
analysis = trader.analyze(
    index_vol=0.18,
    constituent_vols=np.array([0.25, 0.22, 0.28, 0.24]),
    correlation_matrix=corr_matrix,  # Optional
)
```

### DispersionAnalysis

```python
@dataclass
class DispersionAnalysis:
    index_vol: float              # Index implied volatility
    constituent_vols: np.ndarray  # Constituent implied vols
    implied_correlation: float    # Correlation implied by vols
    dispersion_spread: float      # Opportunity size
    is_attractive: bool           # Above min spread threshold
```

### Correlation Utilities

```python
from src.volatility.trading.dispersion import (
    compute_realized_correlation,
    compute_average_correlation,
)

# From returns matrix (n_obs x n_assets)
realized_corr = compute_realized_correlation(returns_matrix)

# From correlation matrix
avg_corr = compute_average_correlation(correlation_matrix)
```

---

## Vol-of-Vol Analytics

### Module: `src.volatility.analytics.vol_of_vol`

### VolOfVolAnalyzer

Analyze volatility-of-volatility and detect regimes.

```python
from src.volatility.analytics import VolOfVolAnalyzer

# Configure analyzer
analyzer = VolOfVolAnalyzer(
    window=20,                  # Rolling window
    annualization=252,          # Trading days
    regime_thresholds={         # Regime boundaries
        "low": 0.15,
        "high": 0.25,
        "crisis": 0.40,
    },
)

# Analyze vol series
metrics = analyzer.analyze(
    implied_vols=iv_series,         # Implied vol time series
    realized_vols=rv_series,        # Optional: realized vol
    prices=price_series,            # Optional: for RV calculation
)
```

### VolOfVolMetrics

```python
@dataclass
class VolOfVolMetrics:
    vol_of_implied_vol: float       # Std of implied vol
    vol_of_realized_vol: float      # Std of realized vol
    mean_implied_vol: float         # Average IV
    mean_realized_vol: float        # Average RV
    vol_premium: float              # IV - RV
    current_regime: str             # "low", "normal", "high", "crisis"
```

### Regime Detection

| Regime | Condition |
|--------|-----------|
| Low | IV < low_threshold |
| Normal | low_threshold ≤ IV < high_threshold |
| High | high_threshold ≤ IV < crisis_threshold |
| Crisis | IV ≥ crisis_threshold |

---

## Complete Example

```python
from src.volatility.trading import (
    VarianceSwap, VarianceSwapPricer,
    DispersionTrader, DispersionConfig,
)
from src.volatility.analytics import VolOfVolAnalyzer

# 1. Price variance swap
swap = VarianceSwap(strike_var=0.04, maturity=0.5, notional=100_000)
pricer = VarianceSwapPricer()

result = pricer.price(
    swap=swap,
    spot=100.0,
    forward=100.5,
    option_strikes=np.linspace(80, 120, 21),
    option_prices_call=call_prices,
    option_prices_put=put_prices,
)

print(f"Fair variance: {result.fair_variance:.4f}")
print(f"Fair vol: {result.fair_vol:.2%}")
print(f"MTM: ${result.mtm:,.2f}")

# 2. Analyze dispersion opportunity
trader = DispersionTrader(
    index_ticker="SPX",
    constituents=["AAPL", "MSFT", "GOOGL"],
)

dispersion = trader.analyze(
    index_vol=0.18,
    constituent_vols=np.array([0.25, 0.22, 0.24]),
)

print(f"Implied correlation: {dispersion.implied_correlation:.2%}")
print(f"Dispersion spread: {dispersion.dispersion_spread:.2%}")
print(f"Attractive: {dispersion.is_attractive}")

# 3. Vol-of-vol analysis
analyzer = VolOfVolAnalyzer(window=20)

metrics = analyzer.analyze(
    implied_vols=vix_history,
    realized_vols=realized_vol_history,
)

print(f"Vol of implied vol: {metrics.vol_of_implied_vol:.2%}")
print(f"Vol premium: {metrics.vol_premium:.2%}")
print(f"Current regime: {metrics.current_regime}")
```
