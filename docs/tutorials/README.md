# QuantStrata Tutorials

Interactive Jupyter notebooks with worked examples covering calibration, pricing, and model usage.

---

## Risk

Risk infrastructure: VaR, Greeks aggregation, and stress testing.

| Tutorial | Description |
|----------|-------------|
| [Risk Introduction](risk/risk_introduction.ipynb) | VaR (historical, parametric, MC), Greeks aggregation, stress scenarios |

---

## Calibration

Learn how to calibrate models to market data.

| Tutorial | Description |
|----------|-------------|
| [Calibration Framework](calibration/calibration_framework.ipynb) | Unified calibration engine, Heston/Hull-White/SABR examples |
| [Curve Bootstrapping](calibration/calibration_curve_bootstrapping.ipynb) | Building discount and forward curves |
| [Volatility Surface](calibration/calibration_volatility_surface.ipynb) | SABR smile fitting |
| [Local Volatility](calibration/local_volatility_analysis.ipynb) | Dupire local vol extraction and analysis |
| [Heston Analysis](calibration/stochastic_vol_heston_analysis.ipynb) | Heston model fitting and diagnostics |

---

## Pricing

Price various derivatives across asset classes.

| Tutorial | Description |
|----------|-------------|
| [FX Options](pricing/fx_options_pricing.ipynb) | FX vanilla, barrier, digital pricing |
| [Equity Options](pricing/equity_options_pricing.ipynb) | Equity option pricing workflows |
| [IR Instruments](pricing/ir_instruments_pricing.ipynb) | FRA, IRS, caps/floors, swaptions |
| [Bond Pricing](pricing/bond_pricing.ipynb) | Zero-coupon and fixed-rate bond pricing |
| [Multi-Asset Options](pricing/multi_asset_options.ipynb) | Basket, spread, rainbow options |

---

## Models

Deep dives into specific pricing models.

| Tutorial | Description |
|----------|-------------|
| [SABR Model](pricing/sabr_model.ipynb) | SABR smile dynamics and calibration |
| [LMM Pricing](pricing/lmm_pricing.ipynb) | LIBOR Market Model simulation and pricing |
| [Jump/Lévy Models](pricing/jump_levy_models.ipynb) | Merton jump-diffusion, Variance Gamma |
| [Advanced MC Methods](pricing/advanced_mc_methods.ipynb) | LSM, QMC, Importance Sampling |

---

## Instruments

Detailed analysis of specific instrument types.

| Tutorial | Description |
|----------|-------------|
| [Vanilla Options](instruments/vanilla_options_analysis.ipynb) | European/American vanilla analysis |
| [Barrier Options](instruments/barrier_options_analysis.ipynb) | Knock-in/knock-out dynamics |
| [Digital Options](instruments/digital_options_analysis.ipynb) | Binary option behavior |
| [Touch Options](instruments/touch_options_analysis.ipynb) | One-touch, no-touch analysis |
| [Asian Options](instruments/asian_options_analysis.ipynb) | Average rate/strike options |
| [Lookback Options](instruments/lookback_options_analysis.ipynb) | Path-dependent lookbacks |
| [Double Barrier Options](instruments/double_barrier_options_analysis.ipynb) | Dual barrier structures |
| [Forward Options](instruments/forward_options.ipynb) | Options on forwards |
| [Futures Options](instruments/futures_options.ipynb) | Options on futures |

---

## Market Data

Working with market data infrastructure.

| Tutorial | Description |
|----------|-------------|
| [Synthetic Data Generation](market-data/synthetic_data_generation.ipynb) | Creating test market data |
| [IR Volatility Surfaces](market-data/ir_volatility_surfaces.ipynb) | Swaption and cap vol surfaces |

---

## Running Tutorials

```bash
# Start Jupyter
cd quantstrata
source .venv/bin/activate
jupyter notebook docs/tutorials/
```

---

*See also: [User Guides](../guides/) | [Technical References](../reference/)*
