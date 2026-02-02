# Spread Options User Guide

This guide explains how to price spread options in QuantStrata.

---

## Overview

A spread option is an option on the difference between two asset prices:

- **Call**: max(S₁(T) - S₂(T) - K, 0)
- **Put**: max(K - (S₁(T) - S₂(T)), 0)

**Use cases:**
- Crack spreads (crude oil vs gasoline)
- Spark spreads (natural gas vs electricity)
- Calendar spreads
- Cross-currency options

---

## Quick Start

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.instruments.multi_asset import MultiAssetSpreadEuropeanOption
from src.pricers.multi_asset import MultiAssetSpreadEuropeanOptionMcPricer

# Create market IDs
def make_id(name: str) -> MarketId:
    return MarketId(asset_class="CMDTY", mkt_type="SPOT", name=name)

# Define the spread call instrument
spread_call = MultiAssetSpreadEuropeanOption(
    option_type="call",
    underlying1=make_id("CL"),   # Crude oil
    underlying2=make_id("HO"),   # Heating oil
    strike=5.0,
    expiry=0.5,
    notional=1.0,
)

# Create pricer and price
pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=100000, seed=42)
price, std_error = pricer.price_with_std_error(
    spread_call,
    spot1=100.0,    # Asset 1 price
    spot2=95.0,     # Asset 2 price
    r=0.05,         # Risk-free rate
    q1=0.02,        # Dividend yield asset 1
    q2=0.01,        # Dividend yield asset 2
    sigma1=0.2,     # Volatility asset 1
    sigma2=0.25,    # Volatility asset 2
    rho=0.6,        # Correlation
)

print(f"Spread Call Price: {price:.4f} ± {std_error:.4f}")
```

---

## Instrument Definition

The `MultiAssetSpreadEuropeanOption` class defines the spread option contract:

```python
from src.instruments.multi_asset import MultiAssetSpreadEuropeanOption

# Spread call
spread_call = MultiAssetSpreadEuropeanOption(
    option_type="call",
    underlying1=make_id("Asset1"),
    underlying2=make_id("Asset2"),
    strike=5.0,
    expiry=0.5,
    notional=1_000_000,
)

# Spread put
spread_put = MultiAssetSpreadEuropeanOption(
    option_type="put",
    underlying1=make_id("Asset1"),
    underlying2=make_id("Asset2"),
    strike=5.0,
    expiry=0.5,
)
```

---

## Pricer Classes

### Monte Carlo Pricer

```python
from src.pricers.multi_asset import MultiAssetSpreadEuropeanOptionMcPricer

# Configure the pricer
pricer = MultiAssetSpreadEuropeanOptionMcPricer(
    n_paths=200_000,
    seed=42,
    antithetic=True,
)

# Price the option
price = pricer.price(
    spread_call,
    spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.01,
    sigma1=0.2, sigma2=0.25, rho=0.6
)

# Get price with standard error
price, std_error = pricer.price_with_std_error(
    spread_call,
    spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.01,
    sigma1=0.2, sigma2=0.25, rho=0.6
)

# Run full simulation
sim = pricer.run(
    spread_call,
    spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.01,
    sigma1=0.2, sigma2=0.25, rho=0.6
)
print(f"Spread values shape: {sim.spread_values.shape}")
```

### Kirk's Approximation

Kirk's approximation provides a fast closed-form estimate:

```python
from src.pricers.multi_asset import MultiAssetSpreadEuropeanOptionKirkPricer

kirk_pricer = MultiAssetSpreadEuropeanOptionKirkPricer()

# Fast analytic approximation
kirk_price = kirk_pricer.price(
    spread_call,
    spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
    sigma1=0.2, sigma2=0.25, rho=0.6
)

print(f"Kirk Call: {kirk_price:.4f}")
```

**When Kirk's works well:**
- Moderate strikes relative to S₂
- Correlation not too close to ±1
- Not too short maturities

**When to use MC instead:**
- Very high/low correlations
- Large strike relative to S₂
- High accuracy required

---

## Exchange Options (Margrabe's Formula)

When K=0, the spread becomes an exchange option with an exact closed-form solution:

```python
from src.instruments.multi_asset import MultiAssetExchangeEuropeanOption
from src.pricers.multi_asset import MultiAssetExchangeEuropeanOptionMargrabePricer

# Exchange option (K=0)
exchange = MultiAssetExchangeEuropeanOption(
    underlying1=make_id("Asset1"),
    underlying2=make_id("Asset2"),
    expiry=1.0,
    notional=1.0,
)

# Exact Margrabe formula
margrabe_pricer = MultiAssetExchangeEuropeanOptionMargrabePricer()
exchange_price = margrabe_pricer.price(
    exchange,
    spot1=100.0, spot2=100.0, r=0.05, q1=0.02, q2=0.02,
    sigma1=0.2, sigma2=0.25, rho=0.5
)

print(f"Exchange Option: {exchange_price:.4f}")

# Verify with MC (spread call with K=0)
spread_zero = MultiAssetSpreadEuropeanOption(
    option_type="call",
    underlying1=make_id("Asset1"),
    underlying2=make_id("Asset2"),
    strike=0.0,
    expiry=1.0,
)
mc_pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=200000, seed=42)
mc_price = mc_pricer.price(
    spread_zero,
    spot1=100.0, spot2=100.0, r=0.05, q1=0.02, q2=0.02,
    sigma1=0.2, sigma2=0.25, rho=0.5
)

print(f"MC Price: {mc_price:.4f}")
```

---

## Comparing Methods

```python
import numpy as np

# Compare Kirk vs MC for different correlations
correlations = [-0.5, 0.0, 0.3, 0.6, 0.9]

print("Correlation | Kirk   | MC     | Diff")
print("-" * 40)

spread = MultiAssetSpreadEuropeanOption(
    option_type="call",
    underlying1=make_id("A"),
    underlying2=make_id("B"),
    strike=5.0,
    expiry=0.5,
)

kirk_pricer = MultiAssetSpreadEuropeanOptionKirkPricer()
mc_pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=100000, seed=42)

for rho in correlations:
    kirk = kirk_pricer.price(
        spread, spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
        sigma1=0.2, sigma2=0.25, rho=rho
    )
    mc = mc_pricer.price(
        spread, spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
        sigma1=0.2, sigma2=0.25, rho=rho
    )
    print(f"{rho:>11.1f} | {kirk:.4f} | {mc:.4f} | {abs(kirk-mc):.4f}")
```

---

## Correlation Effects

Higher correlation means assets move together, reducing spread volatility:

```python
import matplotlib.pyplot as plt

rhos = np.linspace(-0.8, 0.95, 20)
prices = []

exchange = MultiAssetExchangeEuropeanOption(
    underlying1=make_id("A"),
    underlying2=make_id("B"),
    expiry=1.0,
)
margrabe_pricer = MultiAssetExchangeEuropeanOptionMargrabePricer()

for rho in rhos:
    price = margrabe_pricer.price(
        exchange, spot1=100.0, spot2=100.0, r=0.05, q1=0.02, q2=0.02,
        sigma1=0.3, sigma2=0.3, rho=rho
    )
    prices.append(price)

plt.plot(rhos, prices, 'b-')
plt.xlabel('Correlation')
plt.ylabel('Exchange Option Price')
plt.title('Exchange Option Price vs Correlation')
plt.grid(True)
plt.show()
```

---

## Put-Call Parity

Spread options satisfy parity:

```
Call - Put = e^{-rT} × (F₁ - F₂ - K)
```

```python
spread_call = MultiAssetSpreadEuropeanOption(
    option_type="call", underlying1=make_id("A"), underlying2=make_id("B"),
    strike=5.0, expiry=0.5
)
spread_put = MultiAssetSpreadEuropeanOption(
    option_type="put", underlying1=make_id("A"), underlying2=make_id("B"),
    strike=5.0, expiry=0.5
)

pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=200000, seed=42)
call = pricer.price(spread_call, spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
                    sigma1=0.2, sigma2=0.25, rho=0.6)
put = pricer.price(spread_put, spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
                   sigma1=0.2, sigma2=0.25, rho=0.6)

T = 0.5
r = 0.05
F1 = 100.0 * np.exp((r - 0.02) * T)
F2 = 95.0 * np.exp((r - 0.02) * T)
discount = np.exp(-r * T)
parity = discount * (F1 - F2 - 5.0)

print(f"Call - Put: {call - put:.4f}")
print(f"Parity: {parity:.4f}")
```

---

## Tips

### Choosing a Method

| Situation | Recommended Method |
|-----------|-------------------|
| Quick estimate | Kirk's approximation |
| K=0 (exchange) | Margrabe's formula |
| High accuracy | Monte Carlo |
| Extreme parameters | Monte Carlo |

### Common Applications

| Spread Type | Description |
|-------------|-------------|
| Crack spread | Crude oil vs refined products |
| Spark spread | Natural gas vs electricity |
| Crush spread | Soybeans vs soybean oil/meal |
| Calendar spread | Same asset, different dates |

---

## See Also

- [Basket Options Guide](basket_options.md)
- [Rainbow Options Guide](rainbow_options.md)
- [Multi-Asset Tutorial](../../tutorials/pricing/multi_asset_options.ipynb)

---

*QuantStrata User Guide | January 2026*
