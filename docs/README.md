# QuantStrata Documentation

**A professional quantitative finance library for derivatives pricing, risk management, and market data analytics.**

---

## Documentation Structure

### Guides

User-focused documentation for working with QuantStrata.

| Guide | Description |
|-------|-------------|
| [Market Data](guides/market-data/README.md) | Market data architecture, providers, and volatility surfaces |
| [Instruments](guides/instruments/) | Detailed specifications for supported instruments |
| [Equity](guides/equity/README.md) | Equity-specific features including dividend models |

### Reference

Mathematical foundations and model specifications.

| Reference | Description |
|-----------|-------------|
| [Black-Scholes-Merton](reference/models/black_scholes_merton.md) | Generalized BSM with cost-of-carry |
| [Black76](reference/models/black76.md) | Forward/futures pricing model |
| [Bachelier](reference/models/bachelier.md) | Normal model for spreads |
| [Local Volatility](reference/models/local_volatility.md) | Dupire local vol calibration |
| [Monte Carlo](reference/models/monte_carlo_methods.md) | Simulation methods and variance reduction |
| [Finite Difference](reference/models/finite_difference_methods.md) | PDE solvers for American options |
| [Curve Bootstrapping](reference/models/curve_bootstrapping.md) | Interest rate curve construction |
| [Volatility Calibration](reference/models/volatility_calibration.md) | Surface fitting techniques |
| [Heston Model](reference/models/heston_volatility.md) | Stochastic volatility |

### Tutorials

Interactive Jupyter notebooks with worked examples.

| Category | Notebooks |
|----------|-----------|
| **Calibration** | Curve bootstrapping, volatility surface fitting, local vol, Heston |
| **Instruments** | Vanilla, barrier, digital, touch, Asian, lookback, double barrier, forward/futures options |
| **Market Data** | Synthetic data generation for FX, IR, and Equity |
| **Pricing** | FX options, Equity options, IR instruments (FRA, IRS, Caps, Swaptions), Bonds |

### Development

Internal documentation for contributors.

| Document | Description |
|----------|-------------|
| [Roadmap](development/roadmap.md) | Implementation phases and status |
| [Progress](development/progress/) | Phase completion reports |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer

# Define market identifiers
spot_id = MarketId("EQUITY", "SPOT", "AAPL")
vol_id = MarketId("EQUITY", "VOL", "AAPL")
curve_id = MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),))

# Create market snapshot
market = Market(
    asof="2026-01-15",
    quotes={spot_id: Quote(value=150.0)},
    curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
    vols={vol_id: FlatVolSurface(sigma=0.25)},
)

# Create option
option = EuropeanEquityVanillaOption(
    ticker="AAPL",
    option_type="call",
    strike=155.0,
    expiry=0.5,
    notional=100,
    dividend_yield=0.005,
    spot_id=spot_id,
    vol_id=vol_id,
    curve_id=curve_id,
)

# Price
pricer = EquityEuropeanVanillaBsmPricer()
price = pricer.price(option, market)
greeks = pricer.greeks(option, market)
```

---

## Asset Classes

| Asset Class | Status | Models |
|-------------|--------|--------|
| **FX** | Complete | BSM, Black76, Bachelier, Monte Carlo, Finite Difference |
| **Equity** | Complete | BSM, Black76, Bachelier, Monte Carlo, Finite Difference |
| **Rates** | Phase 3 Active | Black76, Bachelier |

## Instruments

| Type | FX | Equity | Rates |
|------|-----|--------|-------|
| Vanilla | Yes | Yes | - |
| Barrier | Yes | Yes | - |
| Digital | Yes | Yes | - |
| Asian | Yes | Yes | - |
| Lookback | Yes | Yes | - |
| Touch | Yes | - | - |
| American | Yes | Yes | - |
| Forward Options | Yes | - | - |
| Futures Options | - | Yes | - |
| Spread Options | Yes | Yes | - |
| FRA | - | - | Yes |
| IRS | - | - | Yes |
| Caps/Floors | - | - | Yes |
| Swaptions | - | - | Yes |
| Bonds (ZC/FR) | - | - | Yes |
| Bond Options | - | - | Yes |

---

## Contributing

See [development documentation](development/) for coding standards and contribution guidelines.
