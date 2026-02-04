# Interactive Jupyter Notebooks

This folder contains **interactive Jupyter notebooks** that demonstrate QuantStrata's capabilities with rich visualizations, explanations, and hands-on exercises.

## Purpose

- **Learn**: Step-by-step tutorials with detailed explanations
- **Visualize**: Charts, plots, and 3D surfaces to understand concepts
- **Experiment**: Modify parameters and see results instantly
- **Reference**: Code patterns for common tasks

## Available Notebooks

| Notebook | Description | Key Topics |
|----------|-------------|------------|
| `01_pricing_and_greeks.ipynb` | Option pricing fundamentals | BSM pricing, Greeks, payoff diagrams, sensitivities |
| `02_volatility_surfaces.ipynb` | Vol surface construction | Smile, term structure, arbitrage, SABR calibration |
| `03_scenario_analysis.ipynb` | Portfolio risk analysis | Stress testing, spot/vol shocks, P&L decomposition |
| `04_multi_pipeline_workflow.ipynb` | End-to-end workflow | Pipeline chaining, market→portfolio→pricing→risk |
| `05_deep_hedging.ipynb` | Machine learning for hedging | Deep hedging, neural networks, risk measures |

## Prerequisites

```bash
# Core requirements
pip install -e .

# Notebook requirements
pip install jupyter matplotlib numpy

# Launch Jupyter
jupyter notebook
```

## How to Use

1. **Start Jupyter**: Run `jupyter notebook` from the repository root
2. **Open a notebook**: Navigate to `examples/notebooks/`
3. **Run cells sequentially**: Use Shift+Enter to execute each cell
4. **Experiment**: Modify parameters in the cells and re-run
5. **Read explanations**: Markdown cells explain concepts and code

## Notebook Structure

Each notebook follows a consistent pattern:

```
1. Introduction
   - What you'll learn
   - Prerequisites
   - Key concepts

2. Setup
   - Imports
   - Configuration
   - Helper functions

3. Main Content
   - Step-by-step tutorial
   - Code with comments
   - Visualizations

4. Key Takeaways
   - Summary of concepts
   - Best practices
   - Next steps

5. Exercises
   - Suggestions for experimentation
```

## Notebook Descriptions

### 01_pricing_and_greeks.ipynb
**Level**: Beginner

Learn the fundamentals of option pricing:
- How to set up market data (spot, curves, vol)
- Creating option instruments
- Computing prices and Greeks
- Interpreting Greek values
- Visualizing payoffs and sensitivities

**Visualizations**:
- Payoff at expiry
- P&L profile
- Delta/gamma/vega/theta vs spot

### 02_volatility_surfaces.ipynb
**Level**: Intermediate

Deep dive into implied volatility:
- Building vol surfaces from quotes
- Understanding the smile and term structure
- Arbitrage constraints (calendar, butterfly)
- SABR model calibration
- 3D surface visualization

**Visualizations**:
- Smile plots by expiry
- Term structure by delta
- 3D surface and contour maps
- SABR parameter sensitivity

### 03_scenario_analysis.ipynb
**Level**: Intermediate

Portfolio risk analysis techniques:
- Building multi-leg portfolios
- Defining spot and vol shocks
- Running scenario analysis
- P&L decomposition
- Risk reporting

**Visualizations**:
- Scenario P&L waterfall
- Spot/vol sensitivity surfaces
- Risk decomposition by position

### 04_multi_pipeline_workflow.ipynb
**Level**: Advanced

End-to-end orchestration:
- Chaining multiple pipelines
- Market data → Portfolio → Pricing → Risk
- State management between steps
- Executive summary generation

**Visualizations**:
- Position-level breakdown
- Greek exposure
- Comprehensive risk dashboard

### 05_deep_hedging.ipynb
**Level**: Advanced

Machine learning for optimal hedging:
- Deep hedging theory and motivation
- Neural network policy learning
- Risk measures (Mean-Variance, CVaR)
- Transaction cost optimization
- Comparison with delta hedging

**Visualizations**:
- P&L distributions (delta vs deep)
- Trading cost comparison
- Training curves
- Risk metric comparison

## Tips for Learning

1. **Run every cell**: Don't skip steps
2. **Read the comments**: Code comments explain the "why"
3. **Experiment**: Change parameters and see what happens
4. **Try exercises**: Each notebook ends with suggestions
5. **Check the source**: Look at imported modules for details

## Troubleshooting

**Import errors**: Ensure QuantStrata is installed (`pip install -e .`)

**Plotting issues**: Some styles may not be available; the code gracefully degrades

**Kernel issues**: Restart the kernel and run from the beginning

## Related Resources

- `examples/pipelines/`: Simpler scripts without visualization
- `examples/workflows/`: Production-style workflows
- `docs/`: API documentation and guides
