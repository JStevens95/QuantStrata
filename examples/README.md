# QuantStrata Examples

This directory contains comprehensive examples demonstrating the QuantStrata library's capabilities for FX derivatives pricing, risk management, and market data handling.

**These examples are designed to be production-ready and representative of workflows used at quant hedge funds.**

## Directory Structure

```
examples/
├── fundamentals/     # Core market data concepts and building blocks
├── pricing/          # Option pricing methods (BSM, MC, FD)
├── risk/             # Risk management, scenarios, P&L attribution, hedging
├── ml/               # Machine learning & reinforcement learning
├── pipelines/        # Production pipeline examples
├── notebooks/        # Interactive Jupyter notebooks
└── workflows/        # End-to-end application workflows
```

---

## Production-Grade Examples

The following examples demonstrate **hedge fund-quality** implementations:

### Scenario Generation & Risk

| Example | Description | Key Features |
|---------|-------------|--------------|
| `risk/fx_option_scenario_pnl.py` | **FX Option Scenario PnL Analysis** | Full term structure simulation with `FactorModelGenerator`, PCA-based curve/vol dynamics, VaR/ES |
| `pipelines/run_var.py` | **VaR Pipeline** | Historical, parametric, and Monte Carlo VaR methods |
| `fundamentals/07_timeseries_generation.py` | **Time Series Generation Tutorial** | GBM, Heston, OU dynamics with correlation |

### Pricing & Greeks

| Example | Description | Key Features |
|---------|-------------|--------------|
| `pricing/01_fx_vanilla_pricing.py` | **FX Vanilla Pricing** | BSM, Monte Carlo, convergence analysis |
| `pricing/02_exotic_options.py` | **Exotic Options** | Barriers, digitals, touch options |
| `pricing/03_portfolio_pricing.py` | **Portfolio Pricing** | Aggregation, portfolio Greeks |

---

## Two-Tier Scenario Generation

QuantStrata provides two approaches to scenario generation:

### Tier 1: TimeseriesGenerator (Simple)
```python
# For scalar risk factors - quick prototyping
from src.marketdata.scenarios.timeseries import TimeseriesGenerator
```
- Output: `[T, S]` per factor
- Use: Learning, simple VaR, testing

### Tier 2: FactorModelGenerator (Production)
```python
# For full term structures - hedge fund production
from src.marketdata.scenarios.timeseries import FactorModelGenerator
```
- Output: 
  - Curves: `[T, S, n_tenors]`
  - Vol surfaces: `[T, S, n_exp, n_strike]`
- Use: Production VaR, XVA, stress testing

See `docs/reference/marketdata/scenario_generation.md` for full documentation.

---

## Part 1: Fundamentals

Learn the core building blocks of the library.

| Script | Description |
|--------|-------------|
| `01_market_ids_and_quotes.py` | MarketId system, Quote objects, Market snapshots |
| `02_curves_and_term_structures.py` | Discount curves, zero rates, forward rates |
| `03_volatility_surfaces.py` | Vol surfaces, implied volatility, smile |
| `04_timeseries_datasets.py` | MarketDataset, Panels, multi-day data |
| `05_market_snapshots.py` | Extracting snapshots for pricing |
| `06_scenario_shocks.py` | SpotShock, VolShock, ParallelRateShock |
| `07_timeseries_generation.py` | **Comprehensive time series generation tutorial** |

**Start here** if you're new to the library.

---

## Part 2: Pricing

Learn to price FX options using different methods.

| Script | Description |
|--------|-------------|
| `01_fx_vanilla_pricing.py` | BSM, Monte Carlo, Finite Difference comparison |
| `02_exotic_options.py` | Barriers, Asians, Lookbacks, Touch options |
| `03_portfolio_pricing.py` | Portfolio aggregation, Greeks |

**Key concepts:** Pricing methods, convergence analysis, Greeks computation.

---

## Part 3: Risk Management

Learn to manage risk with scenarios and sensitivities.

| Script | Description |
|--------|-------------|
| `01_scenario_analysis.py` | Scenario shocks, stress testing, P&L |
| `02_sensitivities_computation.py` | Greeks, bump-and-reprice, engine |
| `fx_option_scenario_pnl.py` | **Production scenario PnL with full term structures** |

**Key concepts:** Scenario P&L, VaR, ES, Greeks validation.

---

## Part 4: Machine Learning & RL

ML and RL applications for quantitative finance.

| Script | Description |
|--------|-------------|
| `01_hedging_environment.py` | RL hedging environment (Gymnasium interface) |
| `02_rl_hedging_agent.py` | Train RL agent for option hedging |
| `03_model_validation.py` | BSM vs Monte Carlo vs Finite Difference comparison |

**Key concepts:** RL for finance, policy gradient, model validation.

---

## Part 5: Pipelines

Production pipeline examples using the orchestrator framework.

| Script | Description |
|--------|-------------|
| `run_var.py` | Value-at-Risk computation pipeline |
| `run_calibrate_heston.py` | Heston model calibration pipeline |
| `run_calibrate_sabr.py` | SABR model calibration pipeline |
| `run_build_curves.py` | Curve construction pipeline |
| `run_build_vol_surface.py` | Vol surface building pipeline |
| `run_portfolio_from_config.py` | Portfolio pricing from config |
| `run_train_gnn_pricer.py` | ML-based pricer training |
| `run_train_neural_sde.py` | Neural SDE training |

---

## Part 5: Showcase

Publication-quality visualizations.

| Script | Description |
|--------|-------------|
| `01_european_vanilla_pricing.py` | Method comparison with professional plots |
| `02_exotic_options_gallery.py` | Visual gallery of exotic payoffs |
| `03_advanced_models.py` | Heston, Local Vol dynamics |

**Use these** for presentations or learning visually.

---

## Running the Examples

From the repository root:

```bash
# Set PYTHONPATH and run
cd /path/to/QuantStrata
PYTHONPATH=. python examples/fundamentals/01_market_ids_and_quotes.py

# Or for risk examples
PYTHONPATH=. python examples/risk/fx_option_scenario_pnl.py

# With optional plotting
PYTHONPATH=. python examples/risk/fx_option_scenario_pnl.py --plot
```

### Dependencies

Most examples require:
- numpy
- matplotlib
- scipy

The QuantStrata library itself (`src/`).

---

## Learning Path

### For Beginners

1. Start with `fundamentals/01_market_ids_and_quotes.py`
2. Work through all fundamentals in order
3. Move to `pricing/01_fx_vanilla_pricing.py`
4. Explore exotic options and portfolios

### For Practitioners / Hedge Fund Context

1. Skim fundamentals for API reference
2. Study `fundamentals/07_timeseries_generation.py` for scenario concepts
3. Review `risk/03_pnl_attribution.py` and `risk/04_delta_hedging.py`
4. Deep dive into `risk/fx_option_scenario_pnl.py` for production patterns
5. Explore `ml/02_rl_hedging_agent.py` for RL applications
6. Review `pipelines/run_var.py` for orchestrator usage

### For Interviewers / Reviewers

1. Check `showcase/` for visual demonstrations
2. Review `risk/fx_option_scenario_pnl.py` for production-quality code
3. Examine `docs/reference/marketdata/scenario_generation.md` for architecture
4. Review test coverage in `tests/`

---

## Hedge Fund Workflow Example

The `risk/fx_option_scenario_pnl.py` example demonstrates a complete hedge fund workflow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HEDGE FUND SCENARIO ANALYSIS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. DEFINE RISK FACTORS                                                  │
│     ├── FX.SPOT.EURUSD (GBM)                                            │
│     ├── IR.CURVE.USD (3 PCA factors: level, slope, curvature)           │
│     ├── IR.CURVE.EUR (3 PCA factors)                                    │
│     └── FX.VOL.EURUSD (3 factors: ATM, skew, smile)                     │
│                                                                          │
│  2. BUILD CORRELATION MATRIX                                             │
│     └── 10×10 matrix capturing cross-asset dependencies                 │
│                                                                          │
│  3. GENERATE SCENARIOS (FactorModelGenerator)                            │
│     ├── Spot paths: [T+1, n_scenarios]                                  │
│     ├── Curve paths: [T+1, n_scenarios, n_tenors]                       │
│     └── Vol paths: [T+1, n_scenarios, n_exp, n_strike]                  │
│                                                                          │
│  4. FULL REVALUATION                                                     │
│     └── Price option at each scenario using BSM                         │
│                                                                          │
│  5. COMPUTE RISK METRICS                                                 │
│     ├── VaR (95%, 99%)                                                  │
│     ├── Expected Shortfall                                              │
│     └── PnL distribution statistics                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Code Style

All examples follow consistent patterns:

- **Docstrings**: Clear description at top of each file
- **Sections**: Numbered sections with headers
- **Print output**: Explanatory text and results
- **Plots**: Professional matplotlib visualizations (optional `--plot` flag)
- **Type hints**: Full type annotations throughout
- **Error handling**: Graceful fallbacks for missing dependencies

---

## Related Documentation

- `docs/reference/marketdata/scenario_generation.md` - Scenario generation architecture
- `docs/mathematics/` - Mathematical derivations
- `docs/notebooks/` - Interactive Jupyter notebooks
- `tests/unit/marketdata/scenarios/` - Comprehensive test suite

---

*QuantStrata Examples | Production-Grade Quant Library*
