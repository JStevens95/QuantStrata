# QuantStrata Tutorials

Interactive Jupyter notebooks with worked examples covering calibration, pricing, and model usage.

---

## Advanced Analytics & Reporting

Front-office risk reports and publication-quality plots.

| Tutorial | Description |
|----------|-------------|
| [Advanced Analytics & Reporting](analytics/advanced_analytics_reporting.ipynb) | VaR summary, RiskReport, Greeks surface, PnL by scenario, figure export |

---

## Streaming and Live Data

Streaming protocol, paper brokerage adapter, and StreamingEngine.

| Tutorial | Description |
|----------|-------------|
| [Streaming and Live Data](streaming/streaming_and_live_data.ipynb) | Replay stream + paper adapter + StreamingEngine; same strategy signature as backtest |

---

## Performance

Performance and scalability: benchmarking, parallel pricing, caching, optional JAX MC.

| Tutorial | Description |
|----------|-------------|
| [Performance and Scalability](performance/performance_and_scalability.ipynb) | Backend selection, parallel portfolio pricing, pricer cache, JAX MC pricer (optional) |

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
| [Exotic Options](pricing/exotic_options.ipynb) | Cliquet, autocallable, range accrual (MC pricing) |

---

## Models

Deep dives into specific pricing and simulation models.

| Tutorial | Description |
|----------|-------------|
| [SABR Model](pricing/sabr_model.ipynb) | SABR smile dynamics and calibration |
| [LMM Pricing](pricing/lmm_pricing.ipynb) | LIBOR Market Model simulation and pricing |
| [Jump/Lévy Models](pricing/jump_levy_models.ipynb) | Merton jump-diffusion, Variance Gamma |
| [Advanced MC Methods](pricing/advanced_mc_methods.ipynb) | LSM, QMC, Importance Sampling |
| [Neural SDE](models/neural_sde_tutorial.ipynb) | Data-driven drift/diffusion, training, path generation (Phase 7.7) |

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

## Machine Learning

ML framework: data preparation, training, evaluation, and inference (Phase 7.1).

| Tutorial | Description |
|----------|-------------|
| [ML Model Lifecycle](machine_learning/ml_model_lifecycle.ipynb) | Config → data → model → training → evaluation → tuning → deployment (generic pipeline) |
| [Hybrid GNN-LSTM](machine_learning/hybrid_gnn_lstm_tutorial.ipynb) | End-to-end Hybrid GNN-LSTM: graph + PnL data, architecture, TrainingManager, evaluation, deployment |
| [ML Production](machine_learning/ml_production.ipynb) | Experiment tracking, model registry, hyperparameter tuning (Phase 7.1.5) |

*See also: [ML Framework Guide](../guides/machine_learning/ml_framework.md) | [ML Framework Reference](../reference/machine_learning/ml_framework.md)*

---

## Deep Hedging

Learn optimal hedging strategies via neural policies (Phase 7.6).

| Tutorial | Description |
|----------|-------------|
| [Deep Hedging Tutorial](deep_hedging/deep_hedging_tutorial.ipynb) | Hedging problem, delta vs deep hedging, training, comparison, **backtesting** (BacktestEngineAdapter / pipeline) |

*See also: [Backtesting Hedging Agents](../guides/deep_hedging/backtesting_hedging_agents.md) | [Deep Hedging Theory](../reference/deep_hedging/theory.md)*

---

## Q-Learning / RL

Deploy and backtest RL agents (Phase 7.2).

| Tutorial | Description |
|----------|-------------|
| [RL Deployment Tutorial](q_learning/rl_deployment_tutorial.ipynb) | Deploy and backtest agents via orchestrator pipelines |

*See also: [Deploying RL Agents](../guides/q_learning/deploying_rl_agents.md) | [RL Framework Reference](../reference/q_learning/rl_framework.md)*

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
