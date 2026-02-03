# End-to-End Workflows

This folder contains **workflow scripts** that combine multiple orchestrator pipelines to run full front-office-style processes. Each workflow runs several pipelines in sequence and passes state (market, portfolio, etc.) from one run to the next.

## Purpose

- **Realistic usage**: Mirror how a quant desk runs market data → pricing → risk in one go.
- **State chaining**: Use `run_pipeline_from_config(..., initial_state=previous_ctx.state)` to feed outputs of one pipeline into the next.
- **Templates**: Adapt these for daily batch jobs, research runs, or integration tests.

## Workflows

| Script | Pipelines Chained | Description |
|--------|-------------------|-------------|
| `options_desk_daily.py` | marketdata.build_timeseries → pricing.price_portfolio → risk.run_scenarios | Full options desk daily: load data, snapshot, price book, run scenarios |
| `pricing_and_var.py` | marketdata.build_timeseries → pricing.price_portfolio → (risk.run_scenarios as proxy for VaR) | Price portfolio then run risk (scenarios as illustrative VaR-style output) |

## How It Works

1. **Run first pipeline** (e.g. `marketdata.build_timeseries`) → get `ctx1` with `dataset`, `market`.
2. **Build or load portfolio** in code (portfolio is not produced by a pipeline in the current set; you inject it).
3. **Run second pipeline** (e.g. `pricing.price_portfolio`) with `initial_state={market: ctx1.state["market"], portfolio: portfolio}`.
4. **Run third pipeline** (e.g. `risk.run_scenarios`) with `initial_state={market: ..., portfolio: ...}` from the same market/portfolio.

Each pipeline run gets its own `run_id` and artifact directory; the workflow script ties them together logically.

## Running

From the repository root:

```bash
python examples/workflows/options_desk_daily.py
python examples/workflows/pricing_and_var.py
```

## More Workflows (Documented)

See **`docs/guides/orchestrator_workflows.md`** for:

- Additional professional workflow ideas (e.g. calibration → pricing, backtest → reporting).
- How to add new workflows once more pipelines (calibration, ML, backtest) are implemented.
