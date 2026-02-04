# End-to-End Workflows

This folder contains **production-style workflow scripts** that demonstrate how to chain multiple components together for realistic front-office use cases.

## Purpose

- **Realistic usage**: Mirror how a quant desk runs daily processes
- **State chaining**: Pass outputs from one step to the next
- **Complete examples**: Full workflows from data to report
- **Templates**: Adapt for production batch jobs

## Available Workflows

| Script | Description | Steps |
|--------|-------------|-------|
| `options_desk_daily.py` | Complete daily options desk workflow | Market Data → Portfolio → Pricing → Scenarios → Report |
| `calibration_to_pricing.py` | Model calibration and exotic pricing | Vol Quotes → SABR Calibration → Validation → Exotic Pricing |

## Workflow Descriptions

### options_desk_daily.py

**Purpose**: Simulates a complete morning process for an FX options trading desk.

**Steps**:
1. **Load Market Data**: Spot quotes, yield curves, vol surfaces
2. **Build Portfolio**: Parse trade blotter into positions
3. **Price Portfolio**: MTM and Greeks for all positions
4. **Run Scenarios**: Stress test under various market moves
5. **Generate Report**: Executive summary for the desk

**Output**:
```
╔══════════════════════════════════════════════════════════════════════╗
║                      OPTIONS DESK DAILY REPORT                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  PORTFOLIO VALUATION                                                 ║
║  Total PV (MTM):              $XXX,XXX                               ║
║                                                                      ║
║  RISK EXPOSURES                                                      ║
║  Delta / Gamma / Vega / Theta                                        ║
║                                                                      ║
║  SCENARIO ANALYSIS                                                   ║
║  Worst/Best case P&L                                                 ║
╚══════════════════════════════════════════════════════════════════════╝
```

### calibration_to_pricing.py

**Purpose**: Demonstrates the workflow from vanilla market quotes to exotic option pricing.

**Steps**:
1. **Load Vol Quotes**: Delta-quoted implied volatilities
2. **Calibrate SABR**: Fit stochastic vol model to market
3. **Validate Calibration**: Compare model vols to market quotes
4. **Price Exotics**: Barrier options, digitals using calibrated model
5. **Model Comparison**: Assess model risk (BSM vs SABR)

**Key Concepts**:
- SABR model parameterization
- Delta-to-strike conversion
- Calibration quality metrics
- Model risk for exotics

## How to Run

From the repository root:

```bash
# Run options desk workflow
python examples/workflows/options_desk_daily.py

# Run calibration workflow
python examples/workflows/calibration_to_pricing.py
```

## Workflow Architecture

### State Management

Workflows pass data between steps using Python objects:

```python
# Step 1 produces market
market, market_ids = step_1_load_market_data()

# Step 2 uses market_ids
portfolio = step_2_build_portfolio(market_ids)

# Step 3 uses both
results = step_3_price_portfolio(market, portfolio)
```

### Configuration

Workflows use inline configuration for clarity:

```python
MARKET_CONFIG = {
    "spots": {"EURUSD": 1.0850},
    "rates": {"USD": 0.05, "EUR": 0.04},
    "vols": {"EURUSD": 0.10},
}
```

In production, you'd load from YAML/JSON files.

### Error Handling

Each step validates its inputs and reports clearly:

```python
print("  [✓] Market data loaded successfully")
```

## Production Considerations

### What Would Change for Production

1. **Market Data**: Connect to live feeds (Bloomberg, Reuters)
2. **Portfolio**: Load from trade capture system
3. **Storage**: Save results to database/file system
4. **Scheduling**: Run via cron/Airflow/Prefect
5. **Alerting**: Send notifications on failures
6. **Logging**: Structured logging for observability

### Adding New Workflows

1. Create new file in `examples/workflows/`
2. Follow the step-by-step pattern
3. Include docstrings and comments
4. Add entry to this README

## Related Resources

- `examples/pipelines/`: Individual pipeline examples
- `examples/notebooks/`: Interactive tutorials
- `docs/architecture/orchestrator_pipeline_documentation.md`: Full pipeline reference
