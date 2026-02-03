# QuantStrata Documentation

**A professional quantitative finance library for derivatives pricing, risk management, calibration, and market data analytics.**

**Current Version:** Phase 5.2 Complete  
**Python:** 3.12+ required  
**Tests:** 1,787 passing

**New here?** See the [Quick Start](QUICKSTART.md) to install and run your first pricing example. Contributors: see [Best Practices](BEST_PRACTICES.md). API reference (Sphinx): build with `pip install -r requirements-docs.txt` then `sphinx-build -b html docs/source docs/build/html`; output in `docs/build/html/`.

---

## Documentation Structure

### Guides

User-focused documentation for working with QuantStrata.

| Category | Guides |
|----------|--------|
| [Backtesting](guides/backtesting/) | Strategy evaluation and performance metrics |
| [Calibration](guides/calibration/) | Unified calibration framework |
| [Models](guides/models/) | BSM, Black76, Bachelier, Heston, Hull-White, SABR, LMM, Merton, Variance Gamma |
| [Instruments](guides/instruments/) | Vanilla, Barrier, Digital, Touch, Asian, Lookback, Bonds, Swaps, Swaptions |
| [Multi-Asset](guides/multi_asset/) | Basket, Spread, Rainbow options |
| [Numerical Methods](guides/numerical_methods/) | LSM, QMC, Importance Sampling |
| [Market Data](guides/market-data/) | Architecture, providers, volatility surfaces |
| [Equity](guides/equity/) | Dividend models |

### Reference

Mathematical foundations, derivations, and model specifications.

| Category | References |
|----------|------------|
| [Calibration](reference/calibration/) | CalibrationEngine architecture, Heston/HW calibration |
| [Analytic Models](reference/models/) | BSM, Black76, Bachelier, SABR derivations |
| [Stochastic Vol](reference/models/) | Heston, Local Volatility (Dupire) |
| [Short Rate](reference/models/) | Hull-White, Black-Karasinski |
| [Forward Rate](reference/models/) | LIBOR Market Model (LMM) |
| [Jump/Lévy](reference/models/) | Merton Jump-Diffusion, Variance Gamma |
| [Numerical](reference/models/) | Monte Carlo, Finite Difference methods |
| [Curves](reference/models/) | Bootstrapping, interpolation |

### Tutorials

Interactive Jupyter notebooks with worked examples.

| Category | Notebooks |
|----------|-----------|
| **Backtesting** | [Introduction](tutorials/backtesting/backtesting_introduction.ipynb) |
| **Calibration** | [Framework](tutorials/calibration/calibration_framework.ipynb), [Curves](tutorials/calibration/calibration_curve_bootstrapping.ipynb), [Vol Surface](tutorials/calibration/calibration_volatility_surface.ipynb), [Local Vol](tutorials/calibration/local_volatility_analysis.ipynb), [Heston](tutorials/calibration/stochastic_vol_heston_analysis.ipynb) |
| **Pricing** | [FX Options](tutorials/pricing/fx_options_pricing.ipynb), [Equity Options](tutorials/pricing/equity_options_pricing.ipynb), [IR Instruments](tutorials/pricing/ir_instruments_pricing.ipynb), [Bonds](tutorials/pricing/bond_pricing.ipynb), [Multi-Asset](tutorials/pricing/multi_asset_options.ipynb) |
| **Models** | [SABR](tutorials/pricing/sabr_model.ipynb), [LMM](tutorials/pricing/lmm_pricing.ipynb), [Jump/Lévy](tutorials/pricing/jump_levy_models.ipynb), [Advanced MC](tutorials/pricing/advanced_mc_methods.ipynb) |
| **Instruments** | [Vanilla](tutorials/instruments/vanilla_options_analysis.ipynb), [Barrier](tutorials/instruments/barrier_options_analysis.ipynb), [Digital](tutorials/instruments/digital_options_analysis.ipynb), [Touch](tutorials/instruments/touch_options_analysis.ipynb), [Asian](tutorials/instruments/asian_options_analysis.ipynb), [Lookback](tutorials/instruments/lookback_options_analysis.ipynb) |
| **Market Data** | [Synthetic Data](tutorials/market-data/synthetic_data_generation.ipynb), [IR Vol Surfaces](tutorials/market-data/ir_volatility_surfaces.ipynb) |

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
# Clone repository
git clone https://github.com/quantstrata/quantstrata.git
cd quantstrata

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Pricing Example

```python
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer

# Market identifiers
spot_id = MarketId("FX", "SPOT", "EURUSD")
vol_id = MarketId("FX", "VOL", "EURUSD")
dom_curve_id = MarketId("IR", "CURVE", "USD.OIS")
for_curve_id = MarketId("IR", "CURVE", "EUR.OIS")

# Market snapshot
market = Market(
    asof="2026-01-27",
    quotes={spot_id: Quote(value=1.08)},
    curves={
        dom_curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05),
        for_curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.03),
    },
    vols={vol_id: FlatVolSurface(sigma=0.10)},
)

# Create option
option = FxVanillaEuropeanOption(
    ccy_pair="EURUSD",
    option_type="call",
    strike=1.10,
    expiry=0.5,
    notional=1_000_000,
    spot_id=spot_id,
    vol_id=vol_id,
    domestic_curve_id=dom_curve_id,
    foreign_curve_id=for_curve_id,
)

# Price and compute Greeks
pricer = FxEuropeanVanillaBsmPricer()
price = pricer.price(option, market)
greeks = pricer.greeks(option, market)

print(f"Price: {price:,.2f}")
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Vega: {greeks.vega:.2f}")
```

### Calibration Example

```python
from src.calibration.stochastic_volatility import calibrate_heston_to_vols
from src.calibration.stochastic_volatility.heston import HestonCalibrationConfig
import numpy as np

# Market vol surface
strikes = np.array([90, 95, 100, 105, 110])
expiries = np.array([0.25, 0.5, 1.0])
market_vols = np.array([...])  # Your market data

# Calibrate Heston
result = calibrate_heston_to_vols(
    market_vols=market_vols,
    strikes=strikes,
    expiries=expiries,
    spot=100.0,
    r=0.05,
    q=0.02,
    config=HestonCalibrationConfig(
        enforce_feller=True,
        use_global_optimizer=True,
    ),
)

print(result)  # Shows κ, θ, ξ, V₀, ρ + diagnostics
```

---

## Asset Classes

| Asset Class | Status | Models | Calibration |
|-------------|--------|--------|-------------|
| **FX** | ✅ Complete | BSM, MC, FDE, SABR | SABR, Dupire |
| **Equity** | ✅ Complete | BSM, MC, FDE, Heston | Heston, SABR, Dupire |
| **Rates** | ✅ Complete | Black76, Bachelier, HW, BK, LMM | Hull-White, SABR |
| **Multi-Asset** | ✅ Complete | Correlated MC | - |

## Instruments

| Type | FX | Equity | Rates | Multi-Asset |
|------|:---:|:------:|:-----:|:-----------:|
| Vanilla European | ✅ | ✅ | ✅ | - |
| Vanilla American | ✅ | ✅ | - | - |
| Barrier | ✅ | ✅ | - | - |
| Digital | ✅ | ✅ | - | - |
| Asian | ✅ | ✅ | - | - |
| Lookback | ✅ | ✅ | - | - |
| Touch | ✅ | - | - | - |
| Forward Options | ✅ | - | - | - |
| Futures Options | - | ✅ | - | - |
| FRA | - | - | ✅ | - |
| IRS | - | - | ✅ | - |
| Caps/Floors | - | - | ✅ | - |
| Swaptions | - | - | ✅ | - |
| Bonds (ZC/Fixed) | - | - | ✅ | - |
| Bond Options | - | - | ✅ | - |
| Basket Options | - | - | - | ✅ |
| Spread Options | - | - | - | ✅ |
| Rainbow (Best/Worst-of) | - | - | - | ✅ |

## Models

| Model | Type | Asset Classes | Calibration |
|-------|------|---------------|-------------|
| Black-Scholes-Merton | Analytic | FX, Equity | N/A |
| Black-76 | Analytic | IR | N/A |
| Bachelier | Analytic | IR | N/A |
| Heston | Stochastic Vol | Equity | ✅ To vol surface |
| SABR | Stochastic Vol | FX, IR | ✅ To smile |
| Local Volatility | Dupire | FX, Equity | ✅ Dupire extraction |
| Hull-White | Short Rate | IR | ✅ To swaptions/caps |
| Black-Karasinski | Short Rate | IR | - |
| LMM | Forward Rate | IR | - |
| Merton | Jump-Diffusion | Equity | - |
| Variance Gamma | Lévy | Equity | - |

## Numerical Methods

| Method | Description | Use Cases |
|--------|-------------|-----------|
| Monte Carlo | Path simulation | Exotic options, multi-asset |
| Finite Difference | PDE discretization | American options, barriers |
| Longstaff-Schwartz | Regression for American | American/Bermudan pricing |
| Quasi-Monte Carlo | Low-discrepancy sequences | Faster MC convergence |
| Importance Sampling | Variance reduction | Rare event pricing |
| FFT Pricing | Carr-Madan integration | Heston calibration |

---

## Project Structure

```
quantstrata/
├── src/
│   ├── calibration/       # Unified calibration framework
│   ├── instruments/       # Trade definitions (FX, Equity, IR, Multi-Asset)
│   ├── models/            # Pricing models (analytic, numeric, stochastic)
│   ├── pricers/           # Asset-class specific pricers
│   ├── marketdata/        # Market data infrastructure
│   ├── portfolio/         # Portfolio management
│   ├── risk/              # Risk metrics & attribution
│   └── orchestrator/      # Pipeline execution
├── tests/                 # 1,787 unit tests
├── docs/                  # Documentation
│   ├── guides/            # User guides
│   ├── reference/         # Technical references
│   ├── tutorials/         # Jupyter notebooks
│   └── development/       # Roadmap & progress
└── examples/              # Example scripts
```

---

## Contributing

See [development documentation](development/) for coding standards and contribution guidelines.

---

*QuantStrata | Professional Quantitative Finance Library | Phase 5.1*
