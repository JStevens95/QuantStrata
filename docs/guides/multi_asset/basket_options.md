# Basket Options User Guide

This guide explains how to price basket options in QuantStrata.

---

## Overview

A basket option is an option on a weighted portfolio of assets:

- **Call**: max(Σ wᵢSᵢ(T) - K, 0)
- **Put**: max(K - Σ wᵢSᵢ(T), 0)

**Use cases:**
- Index options (S&P 500, FTSE)
- Sector ETF options
- Multi-currency portfolios
- Commodity baskets

---

## Quick Start

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.instruments.multi_asset import MultiAssetBasketEuropeanOption
from src.pricers.multi_asset import MultiAssetBasketEuropeanOptionMcPricer
from src.models.numeric.monte_carlo.multi_asset import CorrelationMatrix

# Create market IDs for underlyings
def make_id(name: str) -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="SPOT", name=name)

# Define the basket call instrument
basket_call = MultiAssetBasketEuropeanOption(
    option_type="call",
    underlyings=(make_id("AAPL"), make_id("GOOGL"), make_id("MSFT")),
    weights=(0.4, 0.35, 0.25),
    strike=100.0,
    expiry=1.0,
    notional=1.0,
)

# Define market data
spots = np.array([100.0, 100.0, 100.0])
r = 0.05
dividends = np.array([0.02, 0.02, 0.02])
volatilities = np.array([0.2, 0.25, 0.3])
correlation = CorrelationMatrix.from_flat(0.5, n=3)

# Create pricer and price
pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=100000, seed=42)
price, std_error = pricer.price_with_std_error(
    basket_call, spots, r, dividends, volatilities, correlation
)

print(f"Basket Call Price: {price:.4f} ± {std_error:.4f}")
```

---

## Instrument Definition

The `MultiAssetBasketEuropeanOption` class defines the basket option contract:

```python
from src.instruments.multi_asset import MultiAssetBasketEuropeanOption

# Using the constructor directly
basket_call = MultiAssetBasketEuropeanOption(
    option_type="call",
    underlyings=(make_id("A"), make_id("B")),
    weights=(0.5, 0.5),
    strike=100.0,
    expiry=1.0,
    notional=1_000_000,
)

# Using the convenience constructor
basket_put = MultiAssetBasketEuropeanOption.from_lists(
    option_type="put",
    underlyings=[make_id("A"), make_id("B"), make_id("C")],
    weights=[0.4, 0.35, 0.25],
    strike=100.0,
    expiry=0.5,
)

# Properties
print(f"Number of assets: {basket_call.n_assets}")
print(f"Option type: {basket_call.option_type}")
```

---

## Pricer Classes

### Monte Carlo Pricer

```python
from src.pricers.multi_asset import MultiAssetBasketEuropeanOptionMcPricer

# Configure the pricer
pricer = MultiAssetBasketEuropeanOptionMcPricer(
    n_paths=200_000,  # Number of Monte Carlo paths
    seed=42,          # Random seed for reproducibility
    antithetic=True,  # Use antithetic variates (default)
)

# Price the option
price = pricer.price(basket_call, spots, r, dividends, volatilities, correlation)

# Get price with standard error
price, std_error = pricer.price_with_std_error(
    basket_call, spots, r, dividends, volatilities, correlation
)

# Run full simulation and get artifact
sim = pricer.run(basket_call, spots, r, dividends, volatilities, correlation)
print(f"Paths simulated: {sim.n_paths_effective}")
print(f"Basket values shape: {sim.basket_values.shape}")
```

---

## Correlation Matrix

### Creating from Flat Correlation

```python
from src.models.numeric.monte_carlo.multi_asset import CorrelationMatrix

# All pairwise correlations = 0.5
corr = CorrelationMatrix.from_flat(rho=0.5, n=4)
```

### Creating from Pairwise Correlations

```python
# Specify individual pairs (missing pairs default to 0)
corr = CorrelationMatrix.from_pairs(
    correlations={(0, 1): 0.6, (0, 2): 0.3, (1, 2): 0.4},
    n=3
)
```

### Creating from Full Matrix

```python
import numpy as np

# Full correlation matrix
corr_matrix = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])
correlation = CorrelationMatrix(corr_matrix)
```

---

## Put-Call Parity

Basket options satisfy put-call parity:

```
Call - Put = e^{-rT} × (Forward_basket - K)
```

```python
# Create both call and put
basket_call = MultiAssetBasketEuropeanOption.from_lists(
    option_type="call",
    underlyings=[make_id("A"), make_id("B")],
    weights=[0.5, 0.5],
    strike=100.0,
    expiry=1.0,
)

basket_put = MultiAssetBasketEuropeanOption.from_lists(
    option_type="put",
    underlyings=[make_id("A"), make_id("B")],
    weights=[0.5, 0.5],
    strike=100.0,
    expiry=1.0,
)

pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=200000, seed=42)
call = pricer.price(basket_call, spots, r, dividends, volatilities, correlation)
put = pricer.price(basket_put, spots, r, dividends, volatilities, correlation)

# Verify parity
weights = np.array(basket_call.weights)
T = basket_call.expiry
forward_basket = (spots * np.exp((r - dividends) * T) * weights).sum()
discount = np.exp(-r * T)
parity = discount * (forward_basket - basket_call.strike)

print(f"Call - Put: {call - put:.4f}")
print(f"Parity value: {parity:.4f}")
```

---

## Correlation Effects

Higher correlation reduces diversification, affecting basket option prices:

```python
import matplotlib.pyplot as plt

correlations = [0.0, 0.3, 0.5, 0.7, 0.9]
prices = []

for rho in correlations:
    corr = CorrelationMatrix.from_flat(rho, n=3)
    basket = MultiAssetBasketEuropeanOption.from_lists(
        option_type="call",
        underlyings=[make_id("A"), make_id("B"), make_id("C")],
        weights=[1/3, 1/3, 1/3],
        strike=100.0,
        expiry=1.0,
    )
    pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=50000, seed=42)
    price = pricer.price(
        basket,
        spots=np.array([100.0, 100.0, 100.0]),
        r=0.05,
        dividends=np.array([0.0, 0.0, 0.0]),
        volatilities=np.array([0.3, 0.3, 0.3]),
        correlation=corr,
    )
    prices.append(price)

plt.plot(correlations, prices, 'b-o')
plt.xlabel('Correlation')
plt.ylabel('Basket Call Price')
plt.title('Basket Call Price vs Correlation')
plt.grid(True)
plt.show()
```

---

## Tips

### Choosing Number of Paths

| Accuracy | Paths | Use Case |
|----------|-------|----------|
| ~5% | 10,000 | Quick estimate |
| ~1-2% | 100,000 | Standard pricing |
| <1% | 500,000+ | Final valuation |

### Weight Conventions

- Weights typically sum to 1 for index-like baskets
- Negative weights are allowed (short positions)
- Weights can be any real numbers

### Performance

- Use `seed` parameter for reproducibility
- Antithetic variates are enabled by default
- For many baskets, consider caching correlation Cholesky

---

## See Also

- [Spread Options Guide](spread_options.md)
- [Rainbow Options Guide](rainbow_options.md)
- [Multi-Asset Tutorial](../../tutorials/pricing/multi_asset_options.ipynb)

---

*QuantStrata User Guide | January 2026*
