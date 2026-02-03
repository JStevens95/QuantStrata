# QuantStrata User Guides

This section contains user-focused documentation for working with QuantStrata.

---

## Advanced Analytics & Reporting

Guides for front-office risk reports and publication-quality visualisation.

| Guide | Description |
|-------|-------------|
| [Advanced Analytics & Reporting](analytics/advanced_analytics_reporting.md) | VaR summary, RiskReport, Greeks surface, PnL by scenario, styling and export |

---

## Streaming and Live Data

Guides for streaming market data and live/paper trading.

| Guide | Description |
|-------|-------------|
| [Streaming and Live Data](streaming/streaming_and_live_data.md) | Replay stream, paper adapter, StreamingEngine; same strategy as backtesting; paper vs live |

---

## Performance

Guides for performance and scalability: backend selection, parallel pricing, and caching.

| Guide | Description |
|-------|-------------|
| [Performance and Scalability](performance/performance_and_scalability.md) | Backend (NumPy/Numba/JAX), parallel portfolio pricing, market and pricer caching |

---

## Risk

Guides for risk management: VaR, Greeks aggregation, and stress testing.

| Guide | Description |
|-------|-------------|
| [Risk Framework](risk/risk_framework.md) | VaR (historical, parametric, MC), Greeks aggregation, stress testing |

---

## Calibration

Guides for calibrating models to market data.

| Guide | Description |
|-------|-------------|
| [Calibration Framework](calibration/calibration_framework.md) | Unified interface for model calibration |

---

## Models

Guides for using pricing models.

| Guide | Description |
|-------|-------------|
| [Black-Scholes-Merton](models/black_scholes_merton.md) | Generalized BSM with cost-of-carry |
| [Black-76](models/black76.md) | Forward/futures pricing model |
| [Bachelier](models/bachelier.md) | Normal model for spreads and rates |
| [Heston](models/heston.md) | Stochastic volatility model |
| [Hull-White](models/hull_white.md) | One-factor short rate model |
| [Black-Karasinski](models/black_karasinski.md) | Log-normal short rate model |
| [LMM](models/lmm.md) | LIBOR Market Model |
| [Local Volatility](models/local_volatility.md) | Dupire local vol model |
| [Merton](models/merton.md) | Jump-diffusion model |
| [Variance Gamma](models/variance_gamma.md) | Lévy process model |

---

## Numerical Methods

Guides for advanced numerical techniques.

| Guide | Description |
|-------|-------------|
| [Longstaff-Schwartz](numerical_methods/lsm.md) | American option pricing via regression |
| [Quasi-Monte Carlo](numerical_methods/qmc.md) | Low-discrepancy sequences for faster convergence |
| [Importance Sampling](numerical_methods/importance_sampling.md) | Variance reduction for rare events |

---

## Instruments

Guides for supported financial instruments.

### Options

| Guide | Description |
|-------|-------------|
| [Vanilla Options](instruments/vanilla_options.md) | European and American vanilla options |
| [Barrier Options](instruments/barrier_options.md) | Knock-in/knock-out barriers |
| [Digital Options](instruments/digital_options.md) | Binary/digital payoffs |
| [Touch Options](instruments/touch_options.md) | One-touch, no-touch options |
| [Asian Options](instruments/asian_options.md) | Average rate/strike options |
| [Lookback Options](instruments/lookback_options.md) | Fixed/floating strike lookbacks |
| [Double Barrier Options](instruments/double_barrier_options.md) | Dual barrier structures |
| [Forward Options](instruments/forward_options.md) | Options on forwards |
| [Futures Options](instruments/future_options.md) | Options on futures |

### Interest Rate

| Guide | Description |
|-------|-------------|
| [Bonds](instruments/bonds.md) | Zero-coupon and fixed-rate bonds |
| [Bond Options](instruments/bond_options.md) | Options on bonds |
| [FRA](instruments/fra.md) | Forward Rate Agreements |
| [IRS](instruments/irs.md) | Interest Rate Swaps |
| [Caps/Floors](instruments/caps_floors.md) | Interest rate caps and floors |
| [Swaptions](instruments/swaptions.md) | Options on swaps |

---

## Multi-Asset Products

Guides for multi-asset derivatives.

| Guide | Description |
|-------|-------------|
| [Basket Options](multi_asset/basket_options.md) | Options on weighted baskets |
| [Spread Options](multi_asset/spread_options.md) | Options on price spreads |
| [Rainbow Options](multi_asset/rainbow_options.md) | Best-of, worst-of options |

---

## Market Data

Guides for market data infrastructure.

| Guide | Description |
|-------|-------------|
| [Architecture](market-data/architecture.md) | Market data system design |
| [Synthetic Generators](market-data/synthetic_generators.md) | Generating test market data |
| [Volatility Surfaces](market-data/volatility_surfaces.md) | Vol surface construction |

---

## Other

| Guide | Description |
|-------|-------------|
| [Dividends](equity/dividends.md) | Equity dividend models |
| [Interfaces](interfaces.md) | Core interfaces and protocols |

---

*See also: [Technical References](../reference/) | [Tutorials](../tutorials/)*
