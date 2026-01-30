# Equity Documentation

This directory contains technical documentation for equity instruments, market data, and pricing in QuantStrata.

## Contents

| Document | Description |
|----------|-------------|
| [dividends.md](dividends.md) | Continuous and discrete dividend modeling |

## Quick Reference

### Equity Instruments

| Instrument | Module | Description |
|------------|--------|-------------|
| `EquitySpot` | `instruments.equity.linear.spot` | Spot position |
| `EquityForward` | `instruments.equity.linear.forward` | Forward contract |
| `EuropeanEquityVanillaOption` | `instruments.equity.options.vanilla` | European option |
| `AmericanEquityVanillaOption` | `instruments.equity.options.vanilla` | American option |
| `EquityBarrierOption` | `instruments.equity.options.barrier` | Barrier option |
| `EquityDigitalOption` | `instruments.equity.options.digital` | Digital/binary option |
| `EquityAsianOption` | `instruments.equity.options.asian` | Asian option |
| `EquityLookbackOption` | `instruments.equity.options.lookback` | Lookback option |

### Equity Pricers

| Pricer | Model | Options |
|--------|-------|---------|
| `EquityEuropeanVanillaBsmPricer` | Black-Scholes-Merton | European vanilla |
| `EquityEuropeanDigitalBsmPricer` | Black-Scholes-Merton | European digital |
| `EquityEuropeanVanillaMcPricer` | Monte Carlo | European vanilla |
| `EquityEuropeanBarrierMcPricer` | Monte Carlo | Barrier options |
| `EquityEuropeanAsianMcPricer` | Monte Carlo | Asian options |
| `EquityEuropeanLookbackMcPricer` | Monte Carlo | Lookback options |
| `EquityAmericanVanillaFdePricer` | Finite Difference | American vanilla |

### Key Differences from FX

| Aspect | Equity | FX |
|--------|--------|-----|
| **Vol Convention** | Strike-based | Delta-based |
| **Vol Skew** | Negative (put wing up) | Smile (both wings) |
| **Cost-of-Carry** | $b = r - q$ (dividend yield) | $b = r_d - r_f$ (rate diff) |
| **Typical Product** | Single stock, index | Currency pair |

## Usage Example

```python
from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer

# Create option
option = EuropeanEquityVanillaOption(
    underlying="AAPL",
    strike=150.0,
    expiry=1.0,
    option_type="call",
    notional=100,
    currency="USD",
)

# Price
pricer = EquityEuropeanVanillaBsmPricer()
price = pricer.price(option, market)
greeks = pricer.greeks(option, market)
```

## Related Documentation

- [marketdata/synthetic_generators.md](../marketdata/synthetic_generators.md) - Equity data generation
- [marketdata/volatility_surfaces.md](../marketdata/volatility_surfaces.md) - Vol surface conventions
- [mathematics/black_scholes_merton.md](../mathematics/black_scholes_merton.md) - BSM model theory
- [notebooks/equity/](../notebooks/equity/) - Equity pricing notebooks
