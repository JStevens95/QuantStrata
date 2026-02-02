# Rainbow Options User Guide

This guide explains how to price rainbow options (best-of, worst-of) in QuantStrata.

---

## Overview

Rainbow options have payoffs that depend on the ordering of multiple assets:

| Option | Payoff |
|--------|--------|
| Best-of Call | max(max(S₁, S₂, ..., Sₙ) - K, 0) |
| Best-of Put | max(K - max(S₁, S₂, ..., Sₙ), 0) |
| Worst-of Call | max(min(S₁, S₂, ..., Sₙ) - K, 0) |
| Worst-of Put | max(K - min(S₁, S₂, ..., Sₙ), 0) |

**Use cases:**
- Equity-linked notes
- Principal-protected notes
- Structured deposits
- Multi-asset warrants

---

## Quick Start

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.instruments.multi_asset import (
    MultiAssetBestOfEuropeanOption,
    MultiAssetWorstOfEuropeanOption,
)
from src.pricers.multi_asset import (
    MultiAssetBestOfEuropeanOptionMcPricer,
    MultiAssetWorstOfEuropeanOptionMcPricer,
)
from src.models.numeric.monte_carlo.multi_asset import CorrelationMatrix

# Create market IDs
def make_id(name: str) -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="SPOT", name=name)

# Define best-of call
best_of_call = MultiAssetBestOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("AAPL"), make_id("GOOGL"), make_id("MSFT")),
    strike=100.0,
    expiry=1.0,
)

# Define worst-of call
worst_of_call = MultiAssetWorstOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("AAPL"), make_id("GOOGL"), make_id("MSFT")),
    strike=100.0,
    expiry=1.0,
)

# Market data
spots = np.array([100.0, 100.0, 100.0])
r = 0.05
dividends = np.array([0.02, 0.02, 0.02])
volatilities = np.array([0.2, 0.25, 0.3])
correlation = CorrelationMatrix.from_flat(0.5, n=3)

# Price best-of call
best_pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=100000, seed=42)
best_price, best_std = best_pricer.price_with_std_error(
    best_of_call, spots, r, dividends, volatilities, correlation
)

# Price worst-of call
worst_pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=100000, seed=42)
worst_price, worst_std = worst_pricer.price_with_std_error(
    worst_of_call, spots, r, dividends, volatilities, correlation
)

print(f"Best-of Call: {best_price:.4f} ± {best_std:.4f}")
print(f"Worst-of Call: {worst_price:.4f} ± {worst_std:.4f}")
```

---

## Instrument Definitions

### Best-of Options

```python
from src.instruments.multi_asset import MultiAssetBestOfEuropeanOption

# Using constructor
best_of_call = MultiAssetBestOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("A"), make_id("B"), make_id("C")),
    strike=100.0,
    expiry=1.0,
    notional=1_000_000,
)

# Using convenience constructor
best_of_put = MultiAssetBestOfEuropeanOption.from_list(
    option_type="put",
    underlyings=[make_id("A"), make_id("B")],
    strike=100.0,
    expiry=0.5,
)

# Properties
print(f"Number of assets: {best_of_call.n_assets}")
print(f"Option type: {best_of_call.option_type}")
```

### Worst-of Options

```python
from src.instruments.multi_asset import MultiAssetWorstOfEuropeanOption

worst_of_call = MultiAssetWorstOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("A"), make_id("B")),
    strike=100.0,
    expiry=1.0,
)

worst_of_put = MultiAssetWorstOfEuropeanOption.from_list(
    option_type="put",
    underlyings=[make_id("A"), make_id("B"), make_id("C")],
    strike=100.0,
    expiry=0.5,
)
```

---

## Pricer Classes

### Best-of Monte Carlo Pricer

```python
from src.pricers.multi_asset import MultiAssetBestOfEuropeanOptionMcPricer

pricer = MultiAssetBestOfEuropeanOptionMcPricer(
    n_paths=200_000,
    seed=42,
    antithetic=True,
)

# Price the option
price = pricer.price(best_of_call, spots, r, dividends, volatilities, correlation)

# Get price with standard error
price, std = pricer.price_with_std_error(
    best_of_call, spots, r, dividends, volatilities, correlation
)

# Run full simulation
sim = pricer.run(best_of_call, spots, r, dividends, volatilities, correlation)
print(f"Best values shape: {sim.best_values.shape}")
```

### Worst-of Monte Carlo Pricer

```python
from src.pricers.multi_asset import MultiAssetWorstOfEuropeanOptionMcPricer

pricer = MultiAssetWorstOfEuropeanOptionMcPricer(
    n_paths=200_000,
    seed=42,
    antithetic=True,
)

# Price the option
price = pricer.price(worst_of_call, spots, r, dividends, volatilities, correlation)

# Run full simulation
sim = pricer.run(worst_of_call, spots, r, dividends, volatilities, correlation)
print(f"Worst values shape: {sim.worst_values.shape}")
```

---

## Price Ordering

Rainbow options have natural ordering relationships:

```python
from scipy.stats import norm

# Single-asset Black-Scholes call for comparison
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
single_call = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# For calls: Best-of >= Single-asset >= Worst-of
print(f"Best-of Call:   {best_price:.4f}")
print(f"Single Call:    {single_call:.4f}")
print(f"Worst-of Call:  {worst_price:.4f}")
print(f"Ordering valid: {best_price >= single_call >= worst_price}")
```

---

## Correlation Effects

Correlation has opposite effects on best-of vs worst-of options:

```python
import matplotlib.pyplot as plt

correlations = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
best_prices = []
worst_prices = []

best_pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=50000, seed=42)
worst_pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=50000, seed=42)

for rho in correlations:
    corr = CorrelationMatrix.from_flat(rho, n=2)
    
    best_inst = MultiAssetBestOfEuropeanOption(
        option_type="call",
        underlyings=(make_id("A"), make_id("B")),
        strike=100.0,
        expiry=1.0,
    )
    worst_inst = MultiAssetWorstOfEuropeanOption(
        option_type="call",
        underlyings=(make_id("A"), make_id("B")),
        strike=100.0,
        expiry=1.0,
    )
    
    params = {
        'spots': np.array([100.0, 100.0]),
        'r': 0.05,
        'dividends': np.array([0.0, 0.0]),
        'volatilities': np.array([0.3, 0.3]),
        'correlation': corr,
    }
    
    best_prices.append(best_pricer.price(best_inst, **params))
    worst_prices.append(worst_pricer.price(worst_inst, **params))

plt.figure(figsize=(10, 6))
plt.plot(correlations, best_prices, 'b-o', label='Best-of Call')
plt.plot(correlations, worst_prices, 'r-s', label='Worst-of Call')
plt.xlabel('Correlation')
plt.ylabel('Option Price')
plt.title('Rainbow Option Prices vs Correlation')
plt.legend()
plt.grid(True)
plt.show()
```

**Key insight:**
- **Higher correlation → Lower best-of value** (best performer less likely to excel)
- **Higher correlation → Higher worst-of value** (worst performer less likely to crash)

---

## Number of Assets

Rainbow options can have any number of underlying assets:

```python
# 5-asset best-of call
corr = CorrelationMatrix.from_flat(rho=0.4, n=5)

best_5 = MultiAssetBestOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("A"), make_id("B"), make_id("C"), make_id("D"), make_id("E")),
    strike=100.0,
    expiry=1.0,
)

pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=100000, seed=42)
price = pricer.price(
    best_5,
    spots=np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
    r=0.05,
    dividends=np.zeros(5),
    volatilities=np.array([0.2, 0.22, 0.25, 0.28, 0.3]),
    correlation=corr,
)

print(f"5-asset Best-of Call: {price:.4f}")
```

---

## Structured Products Example

Worst-of puts are common in structured products (e.g., reverse convertibles):

```python
# Reverse convertible: investor sells worst-of put
# Higher correlation = cheaper protection = lower coupon to investor

worst_put = MultiAssetWorstOfEuropeanOption(
    option_type="put",
    underlyings=(make_id("A"), make_id("B"), make_id("C")),
    strike=80.0,  # 80% barrier
    expiry=1.0,
)

pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=100000, seed=42)

# Low correlation
low_corr = CorrelationMatrix.from_flat(0.3, n=3)
put_low = pricer.price(
    worst_put,
    spots=np.array([100.0, 100.0, 100.0]),
    r=0.03,
    dividends=np.zeros(3),
    volatilities=np.array([0.25, 0.25, 0.25]),
    correlation=low_corr,
)

# High correlation
high_corr = CorrelationMatrix.from_flat(0.8, n=3)
put_high = pricer.price(
    worst_put,
    spots=np.array([100.0, 100.0, 100.0]),
    r=0.03,
    dividends=np.zeros(3),
    volatilities=np.array([0.25, 0.25, 0.25]),
    correlation=high_corr,
)

print(f"Worst-of Put (low corr):  {put_low:.4f}")
print(f"Worst-of Put (high corr): {put_high:.4f}")
print(f"Higher corr means lower put value (less protection needed)")
```

---

## Tips

### Choosing Number of Paths

| Assets | Recommended Paths |
|--------|------------------|
| 2 | 50,000 - 100,000 |
| 3-5 | 100,000 - 200,000 |
| 5+ | 200,000+ |

### Performance

- More assets = more random draws per path
- Antithetic variates help reduce variance
- Use seed for reproducibility

### Common Pitfalls

- Don't confuse best-of-call with call-on-best
- Correlation must form valid PSD matrix
- Negative correlation is limited by number of assets

---

## See Also

- [Basket Options Guide](basket_options.md)
- [Spread Options Guide](spread_options.md)
- [Multi-Asset Tutorial](../../tutorials/pricing/multi_asset_options.ipynb)

---

*QuantStrata User Guide | January 2026*
