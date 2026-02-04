# Pipeline Examples

This folder contains **runnable example scripts** for each built-in orchestrator pipeline. Each script showcases how to configure and run a single pipeline programmatically.

## Purpose

- **Learn**: See exactly what config each pipeline expects and what it produces
- **Test**: Run pipelines from the command line without writing config files
- **Template**: Copy and adapt for your own configs or automation
- **Reference**: Understand the inputs, outputs, and parameters of each pipeline

## Quick Start

```bash
# From the repository root, run any example:
python examples/pipelines/run_build_curves.py
python examples/pipelines/run_build_vol_surface.py
python examples/pipelines/run_compute_greeks.py
# ... etc
```

## Available Pipelines

### Market Data Pipelines

| Script | Pipeline Name | Description |
|--------|---------------|-------------|
| `run_build_curves.py` | `marketdata.build_curves` | Bootstrap yield curves from rate quotes (deposits, swaps) |
| `run_build_vol_surface.py` | `marketdata.build_vol_surface` | Build implied vol surface from option quotes |

### Portfolio Pipelines

| Script | Pipeline Name | Description |
|--------|---------------|-------------|
| `run_portfolio_from_config.py` | `portfolio.build_from_config` | Build portfolio from position specifications |

### Risk Pipelines

| Script | Pipeline Name | Description |
|--------|---------------|-------------|
| `run_compute_greeks.py` | `risk.compute_sensitivities` | Compute portfolio Greeks (delta, gamma, vega, theta, rho) |
| `run_var.py` | `risk.compute_var` | Compute Value-at-Risk (Historical, Parametric, Monte Carlo) |

### Calibration Pipelines

| Script | Pipeline Name | Description |
|--------|---------------|-------------|
| `run_calibrate_sabr.py` | `calibration.volatility_surface` | Calibrate SABR model to vol quotes |
| `run_calibrate_heston.py` | `calibration.stochastic_vol` | Calibrate Heston model to option prices |

### Backtest Pipelines

| Script | Pipeline Name | Description |
|--------|---------------|-------------|
| `run_backtest_strategy.py` | `backtest.run_strategy` | Run strategy backtest with performance analytics |

## Script Structure

Each example script follows a consistent pattern:

```python
#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: [pipeline.name]
===============================================================================

[What this pipeline does]
[When to use this pipeline]
[Key concepts explained]
[How to run]

===============================================================================
"""

# IMPORTS
# ...

# CONFIGURATION
def build_config() -> RunConfig:
    """Build and validate the pipeline configuration."""
    # ... detailed configuration with comments ...

# MAIN EXECUTION
def main() -> None:
    """
    Execute the pipeline and display results.
    
    Steps:
    1. Build configuration
    2. Execute pipeline
    3. Extract results from context
    4. Display formatted output
    """
    # ... execution and display logic ...

if __name__ == "__main__":
    main()
```

## What Each Script Teaches

### `run_build_curves.py`
- How to specify rate quotes (deposits, FRAs, swaps)
- Bootstrapping methodology
- Interpolation and extrapolation settings
- Reading zero rates and discount factors

### `run_build_vol_surface.py`
- Delta vs strike quote conventions
- Smile and term structure concepts
- Arbitrage validation
- Surface interpolation

### `run_portfolio_from_config.py`
- Position specification format
- Instrument types and parameters
- Long/short positions
- Strategy construction (spreads, straddles)

### `run_compute_greeks.py`
- Greek definitions and interpretations
- Bump-and-reprice methodology
- Aggregation by underlying/currency
- Risk exposure analysis

### `run_var.py`
- VaR methodologies compared
- Historical simulation vs parametric
- Monte Carlo VaR
- Expected Shortfall (CVaR)

### `run_calibrate_sabr.py`
- SABR model parameters
- Smile dynamics
- Calibration objective functions
- Quality metrics

### `run_calibrate_heston.py`
- Heston model explained
- Feller condition
- Parameter interpretation
- Global vs local optimization

### `run_backtest_strategy.py`
- Strategy configuration
- Market data simulation
- Execution and transaction costs
- Performance metrics (Sharpe, Sortino, Calmar)

## Config vs Code

- **Code examples** (this folder): Build `RunConfig` in Python so you can see all required fields in one place
- **Config files**: You can also load config from JSON/YAML via `load_run_config(path)`

## Next Steps

- **Combine pipelines**: See `examples/workflows/` for end-to-end workflows that chain multiple pipelines
- **Interactive analysis**: See `examples/notebooks/` for Jupyter notebooks with visualizations
- **Full reference**: See `docs/architecture/orchestrator_pipeline_documentation.md` for all documented pipelines

## Notes

- All examples use synthetic/mock data by default
- Artifacts are saved to `./artifacts/<example_name>/`
- Each script is self-contained and can be run independently
- Comments explain the "why" behind configuration choices
