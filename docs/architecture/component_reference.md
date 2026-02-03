# QuantStrata Component Reference

Detailed reference of what each module provides and requires.

---

## 1. marketdata

**Purpose:** Market data infrastructure — quotes, curves, volatility surfaces, scenarios, and providers.

### Structure

```
marketdata/
├── core/                  # Core types and interfaces
│   ├── market.py         # Market snapshot
│   ├── ids.py            # MarketId canonical identifiers
│   ├── interfaces.py     # Curve, VolSurface, Quote protocols
│   ├── dataset.py        # MarketDataset (time series/scenarios)
│   ├── panel.py          # Panel numeric container
│   ├── requests.py       # MarketRequest, TimeseriesRequest, Universe
│   └── types.py          # Type aliases and enums
├── curves/               # Interest rate term structures
│   ├── term_structure.py # FlatZeroRateCurve, ZeroRateCurve
│   ├── factory.py        # Curve factories
│   └── bootstrapper.py   # Curve bootstrapping
├── surfaces/             # Volatility surfaces
│   ├── vol_surface.py    # FlatVolSurface, GridVolSurface, SwaptionVolCube
│   ├── local_vol_surface.py
│   └── fx/               # FX-specific conventions and calibration
├── providers/            # Market data providers
│   ├── interfaces.py     # MarketDataProvider protocol
│   ├── static/           # Static provider (from artifacts)
│   ├── synthetic/        # Synthetic data generation
│   ├── historical/       # Historical provider
│   ├── hybrid/           # Primary + fallback
│   └── streaming/        # Real-time streaming
├── scenarios/            # Scenario generation and shocks
│   ├── interfaces.py     # MarketView, ScenarioShock, ScenarioPack
│   ├── shocks.py         # SpotShock, VolShock, ParallelRateShock, etc.
│   └── runner.py         # Scenario execution
└── cache.py              # Market data caching
```

### Key Exports

| Class/Function | Description |
|---------------|-------------|
| `Market` | Immutable market snapshot (asof, quotes, curves, vols) |
| `MarketId` | Canonical identifier for market data |
| `Quote` | Scalar market quote wrapper |
| `Curve` (protocol) | `df(t)`, `zero_rate(t)`, `forward_rate(t1, t2)` |
| `VolSurface` (protocol) | `implied_vol(expiry, strike)` |
| `MarketDataset` | Time series / scenario container |
| `MarketDataProvider` (protocol) | `get_market()`, `get_timeseries()` |
| `ScenarioShock` | Base for market shocks |
| `SpotShock`, `VolShock`, etc. | Concrete shock implementations |

### Dependencies

- **External:** numpy
- **Internal:** None (foundation module)

### Dependents

- instruments, pricers, portfolio, risk, calibration, backtesting, streaming, orchestrator, ui

---

## 2. instruments

**Purpose:** Financial instrument definitions — trade parameters without pricing logic.

### Structure

```
instruments/
├── core/types.py         # Shared type aliases
├── fx/
│   ├── linear/           # Spot, Forward, Swap
│   └── options/          # Vanilla, Barrier, Digital, Asian, Lookback, etc.
├── equity/
│   ├── linear/           # Spot, Forward
│   └── options/          # Vanilla, Barrier, Digital, Asian, Futures, etc.
├── ir/
│   ├── linear/           # FRA, Swap, Bond
│   └── options/          # Caplet, Cap, Swaption, Bond Options
└── multi_asset/          # Basket, Rainbow, Spread
```

### Key Exports

| Class | Description |
|-------|-------------|
| `FxVanillaEuropeanOption` | FX European vanilla option |
| `FxBarrierOption` | FX barrier option |
| `EquityVanillaEuropeanOption` | Equity European vanilla |
| `IrSwap` | Interest rate swap |
| `Caplet`, `Cap`, `Floor` | IR caps/floors |
| `Swaption` | Interest rate swaption |
| `BasketOption` | Multi-asset basket |

### Design Pattern

- All instruments are `@dataclass(frozen=True, slots=True)`
- Market data references via `MarketId`
- No pricing logic — pure data structures

### Dependencies

- **External:** numpy (minimal)
- **Internal:** `marketdata.core.ids.MarketId`, `models.payoffs.types`

### Dependents

- pricers, portfolio

---

## 3. models

**Purpose:** Mathematical pricing models — pure functions, no market objects.

### Structure

```
models/
├── analytic/
│   ├── black_scholes_merton/  # BSM vanilla, digital, barrier
│   ├── black76/               # Forward options
│   └── bachelier/             # Normal model
├── stochastic_volatility/
│   ├── heston.py              # Heston model
│   └── sabr.py                # SABR model
├── short_rate/
│   ├── hull_white.py          # Hull-White
│   └── black_karasinski.py    # Black-Karasinski
├── jump_diffusion/
│   └── merton.py              # Merton jump-diffusion
├── levy/
│   └── variance_gamma.py      # Variance Gamma
├── dynamics/
│   └── gbm_dynamics.py        # GBM simulator
├── payoffs/
│   ├── base.py                # BasePayoff1D, BasePathPayoff1D
│   ├── factory.py             # PayoffFactory
│   ├── vanilla.py, digital.py, barrier.py, asian.py, lookback.py, touch.py
├── numeric/
│   ├── monte_carlo/           # MC estimation, RNG, control variates, QMC, LSM
│   └── finite_difference/     # Grids, operators, schemes, solvers
└── common/                    # Shared types and validation
```

### Key Exports

| Function/Class | Description |
|----------------|-------------|
| `bsm_vanilla_price()` | BSM vanilla price |
| `bsm_vanilla_greeks()` | BSM Greeks |
| `black76_vanilla_price()` | Black76 price |
| `HestonDynamics` | Heston MC simulator |
| `SabrDynamics` | SABR MC simulator |
| `GbmDynamicsSimulator` | GBM path generator |
| `VanillaPayoff`, `BarrierPayoff`, etc. | Payoff implementations |
| `PayoffFactory` | Instrument → Payoff routing |
| `solve_pde_theta()` | FDE solver |

### Dependencies

- **External:** numpy, scipy
- **Internal:** None (foundation module)

### Dependents

- pricers, calibration

---

## 4. pricers

**Purpose:** Pricing adapters — connect instruments + market to models.

### Structure

```
pricers/
├── registry.py           # PricerRegistry, DefaultPricerRegistry
├── fx/
│   ├── spot.py, forward.py
│   ├── european_bsm.py   # BSM analytic
│   ├── european_b76.py   # Black76
│   ├── european_bsm_mc.py, european_bsm_fde.py
│   └── european_bsm_jax_mc.py  # Optional JAX
├── equity/
│   ├── spot.py, forward.py
│   ├── european_bsm.py, european_b76.py
│   └── european_bsm_mc.py, american_bsm_fde.py
├── ir/
│   ├── swap.py, fra.py, bond.py
│   ├── european_b76.py   # Caps/floors
│   ├── european_hw.py    # Hull-White
│   └── european_hw_mc.py, european_hw_fde.py
└── multi_asset/
    └── basket_european_mc.py, rainbow_european_mc.py
```

### Key Exports

| Class | Description |
|-------|-------------|
| `PricerRegistry` | Type-driven pricer resolution |
| `FxVanillaEuropeanOptionBsmPricer` | FX BSM pricer |
| `FxVanillaEuropeanOptionMcPricer` | FX MC pricer |
| `EquityVanillaEuropeanOptionBsmPricer` | Equity BSM pricer |
| `IrSwapPricer` | IR swap pricer |

### Protocol

```python
class InstrumentPricer(Protocol):
    def price(self, instrument: Any, market: Market) -> float: ...
    def greeks(self, instrument: Any, market: Market) -> Dict[str, float]: ...
```

### Dependencies

- **External:** numpy
- **Internal:** `marketdata`, `instruments`, `models`

### Dependents

- portfolio, risk, ui

---

## 5. portfolio

**Purpose:** Portfolio management — positions, portfolio pricing, aggregation.

### Structure

```
portfolio/
├── core.py       # Position, Portfolio, PortfolioResult, PortfolioTotals
├── portfolio.py  # PortfolioPricer
├── parallel.py   # ParallelPortfolioPricer
└── caching.py    # Result caching
```

### Key Exports

| Class | Description |
|-------|-------------|
| `Position` | Instrument + quantity + metadata |
| `Portfolio` | Collection of positions |
| `PortfolioPricer` | Prices portfolio via registry |
| `ParallelPortfolioPricer` | Parallel pricing wrapper |
| `PortfolioResult` | Per-position and total results |

### Dependencies

- **Internal:** `pricers.registry`, `marketdata.core.market`

### Dependents

- risk, backtesting, streaming, orchestrator, machine_learning

---

## 6. risk

**Purpose:** Risk computation — VaR, sensitivities, scenarios, attribution.

### Structure

```
risk/
├── var/
│   ├── config.py         # VarConfig
│   ├── historical.py     # HistoricalVaR
│   ├── parametric.py     # ParametricVaR
│   ├── mc.py             # MonteCarloVaR
│   └── runner.py         # compute_var()
├── sensitivities/
│   ├── engine.py         # compute_sensitivities()
│   ├── aggregation.py    # Greeks aggregation
│   └── result.py         # SensitivitiesReport
├── scenarios/
│   ├── runner.py         # run_portfolio_scenarios()
│   └── generation.py     # Scenario generation
├── attribution/
│   ├── runner.py         # P&L attribution
│   └── report.py
├── reporting/
│   ├── risk_report.py    # RiskReport
│   ├── var_summary.py    # VarSummary
│   └── scenario_report.py
└── validation/           # Greeks vs scenarios validation
```

### Key Exports

| Function/Class | Description |
|----------------|-------------|
| `compute_var()` | VaR dispatcher |
| `compute_sensitivities()` | Greeks computation |
| `run_portfolio_scenarios()` | Scenario analysis |
| `VarResult` | VaR output |
| `SensitivitiesReport` | Greeks report |
| `RiskReport` | Aggregated risk report |

### Dependencies

- **Internal:** `portfolio`, `marketdata.scenarios`, `pricers`

### Dependents

- orchestrator, core.reporting

---

## 7. calibration

**Purpose:** Model calibration — fit model parameters to market data.

### Structure

```
calibration/
├── core/
│   ├── engine.py         # CalibrationEngine
│   ├── objectives.py     # Objective functions
│   └── optimizers.py     # Optimizer configs
├── volatility_surface/
│   ├── sabr.py           # SABR calibration
│   ├── dupire.py         # Dupire local vol
│   └── quantlib/         # QuantLib implementations
├── stochastic_volatility/
│   └── heston.py         # Heston calibration
└── short_rate/
    └── hull_white.py     # Hull-White calibration
```

### Key Exports

| Function/Class | Description |
|----------------|-------------|
| `CalibrationEngine` | Generic calibration orchestrator |
| `calibrate_sabr_to_smile()` | SABR smile calibration |
| `sabr_implied_vol()` | SABR implied vol |
| `calibrate()` | Convenience function |

### Dependencies

- **External:** scipy.optimize
- **Internal:** `models`, `marketdata.surfaces`

---

## 8. backtesting

**Purpose:** Historical strategy testing with performance metrics.

### Structure

```
backtesting/
├── core/
│   ├── engine.py         # BacktestEngine
│   └── metrics.py        # PerformanceMetrics
├── data/
│   ├── adapter.py        # BacktestDataAdapter
│   └── providers.py      # DictDataProvider, CsvDataProvider
└── attribution/
    └── pnl.py            # P&L attribution
```

### Key Exports

| Class/Function | Description |
|----------------|-------------|
| `BacktestEngine` | Main backtest runner |
| `BacktestConfig` | Configuration |
| `BacktestResult` | Output with portfolio value series |
| `PerformanceMetrics` | Sharpe, Sortino, drawdown, etc. |
| `DictDataProvider` | In-memory data provider |
| `CsvDataProvider` | CSV file provider |

### Strategy Signature

```python
def strategy(market: MarketSnapshot, portfolio: PortfolioState, context: Context) -> Sequence[Order]
```

### Dependencies

- **Internal:** `marketdata.providers`, `portfolio`

---

## 9. streaming

**Purpose:** Live/paper trading with same strategy interface as backtesting.

### Structure

```
streaming/
├── engine.py             # StreamingEngine
├── context.py            # LiveContext
├── portfolio_state.py    # Portfolio state utilities
└── brokerage/
    ├── protocol.py       # BrokerageAdapter protocol
    └── paper.py          # PaperBrokerageAdapter
```

### Key Exports

| Class | Description |
|-------|-------------|
| `StreamingEngine` | Event-driven execution engine |
| `BrokerageAdapter` (protocol) | Order submission interface |
| `PaperBrokerageAdapter` | In-memory paper trading |
| `LiveContext` | Context for streaming strategies |

### Dependencies

- **Internal:** `marketdata.providers.streaming`, `portfolio`

---

## 10. orchestrator

**Purpose:** Workflow coordination — pipelines, steps, artifacts.

### Structure

```
orchestrator/
├── core/
│   ├── pipeline.py       # Pipeline, PipelineRunner
│   ├── step.py           # Step base class
│   ├── context.py        # Context (state, logger, artifacts)
│   ├── registry.py       # PipelineRegistry
│   └── state_keys.py     # StateKeys constants
├── pipelines/
│   ├── pricing/          # price_portfolio pipeline
│   ├── risk/             # run_scenarios pipeline
│   └── marketdata/       # build_timeseries pipeline
├── runtime/
│   ├── entrypoints.py    # run_pipeline_from_config()
│   ├── cli.py            # CLI interface
│   └── discovery.py      # Pipeline discovery
├── artifacts/
│   ├── store.py          # ArtifactStore
│   └── manifest.py       # RunManifest
└── config/
    ├── schemas.py        # RunConfig
    └── loader.py, validate.py
```

### Key Exports

| Class/Function | Description |
|----------------|-------------|
| `Pipeline` | Sequence of steps |
| `PipelineRunner` | Executes pipelines |
| `Context` | Execution context |
| `Step` | Base step abstraction |
| `run_pipeline_from_config()` | Main entry point |
| `ArtifactStore` | Output management |

### Dependencies

- **Internal:** `portfolio`, `risk`, `marketdata`

---

## 11. machine_learning

**Purpose:** ML framework — training, evaluation, inference pipelines.

### Structure

```
machine_learning/
├── core/
│   ├── protocols.py      # Trainable protocol, KerasTrainableAdapter
│   └── types.py          # TrainingConfig, TrainingResult, EvaluationResult
├── pipeline/
│   ├── training.py       # run_training(), TrainingLoop
│   ├── evaluation.py     # evaluate_model()
│   └── inference.py      # save_model(), load_model(), predict()
├── data/
│   ├── types.py          # MLDataset, PricingFeatures, CalibrationFeatures
│   ├── pricing.py        # build_pricing_dataset_from_mc/analytic
│   ├── calibration.py    # build_calibration_dataset
│   └── portfolio.py      # build_gnn_dataset_from_portfolio
├── models/
│   └── gnn_rnn_hybrid/   # GNN-RNN model (attention, fusion, projection)
├── calibration/
│   └── training_manager.py  # Keras-specific training manager
└── utilities/
    ├── trade_graph_builder.py
    └── trade_attribute_encoder.py
```

### Key Exports

| Class/Function | Description |
|----------------|-------------|
| `Trainable` (protocol) | Model interface for training |
| `TrainingConfig` | Training configuration |
| `run_training()` | Generic training loop |
| `evaluate_model()` | Model evaluation |
| `save_model()`, `load_model()`, `predict()` | Inference pipeline |
| `MLDataset` | Feature/target container |
| `build_pricing_dataset_from_mc()` | MC → pricing dataset |

### Dependencies

- **External:** numpy, tensorflow (optional for Keras)
- **Internal:** `portfolio`, `pricers`, `marketdata`

---

## 12. core

**Purpose:** Utilities — math, performance backends, reporting/plotting.

### Structure

```
core/
├── math/
│   ├── normal.py         # Normal distribution utilities
│   └── rates.py          # Rate conversion utilities
├── performance/
│   ├── backend.py        # Backend selection (NumPy/Numba/JAX)
│   ├── fd_kernels.py     # FDE kernels
│   ├── mc_kernels.py     # MC kernels
│   └── jax_kernels.py    # JAX implementations
└── reporting/plots/
    ├── style.py          # Plot styling
    ├── utils.py          # Export utilities
    ├── marketdata/       # Curves, surfaces, scenarios plots
    ├── pricers/          # FDE, MC diagnostic plots
    ├── risk/             # Greeks surface, PnL scenario plots
    └── portfolio/        # Portfolio plots
```

### Key Exports

| Module | Description |
|--------|-------------|
| `core.math.normal` | `std_normal_cdf`, `std_normal_ppf` |
| `core.performance.backend` | `get_backend()`, `set_backend()` |
| `core.reporting.plots.style` | Plot styling |
| `core.reporting.plots.risk` | `plot_greeks_surface()`, `plot_pnl_by_scenario()` |

---

## 13. ui

**Purpose:** Dash UIs for browser-based interaction.

### Structure

```
ui/
├── run.py                # Entry point: python -m src.ui.run <app>
├── _shared/
│   ├── layout.py         # make_app_layout()
│   ├── styles.py         # Style constants
│   └── components.py     # Reusable components
└── apps/
    └── pricing_calculator/
        └── app.py        # FX vanilla pricing calculator
```

### Key Exports

| Function | Description |
|----------|-------------|
| `create_pricing_calculator_app()` | FX pricing calculator Dash app |
| `make_app_layout()` | Standard app shell |
| `input_row()`, `dropdown_row()`, etc. | Reusable form components |

### Dependencies

- **External:** dash
- **Internal:** `marketdata`, `instruments`, `pricers`
