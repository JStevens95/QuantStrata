# Pipeline Examples

This folder contains **runnable example scripts** for each built-in orchestrator pipeline. Each script showcases how to configure and run a single pipeline programmatically.

## Purpose

- **Learn**: See exactly what config each pipeline expects and what it produces.
- **Test**: Run pipelines from the command line without writing config files.
- **Template**: Copy and adapt for your own configs or automation.

## Available Pipelines

| Script | Pipeline Name | Description |
|--------|----------------|-------------|
| `run_marketdata_build_timeseries.py` | `marketdata.build_timeseries` | Build timeseries dataset and market snapshot from synthetic or static data |
| `run_pricing_price_portfolio.py` | `pricing.price_portfolio` | Price a portfolio against a market snapshot |
| `run_risk_run_scenarios.py` | `risk.run_scenarios` | Run scenario shocks and produce P&L / risk report |

## How to Run

From the repository root:

```bash
# Run a single pipeline example
python examples/pipelines/run_marketdata_build_timeseries.py
python examples/pipelines/run_pricing_price_portfolio.py
python examples/pipelines/run_risk_run_scenarios.py
```

Each script:

1. Builds a `RunConfig` (programmatically or from a small inline dict).
2. Calls `run_pipeline_from_config(cfg)`.
3. Prints a short summary of the result (state keys, key outputs).

## Config vs Code

- **Config file**: You can also load config from JSON/YAML via `load_run_config(path)` (see `src.orchestrator.config.loader`).
- **Code**: These examples build `RunConfig` in Python so you can see all required fields in one place.

## Next Steps

- **Combine pipelines**: See `examples/workflows/` for end-to-end workflows that chain multiple pipelines.
- **Full reference**: See `docs/architecture/orchestrator_pipeline_documentation.md` for all documented pipelines (including those not yet implemented).
