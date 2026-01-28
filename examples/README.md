# QuantStrata Examples

This directory contains comprehensive examples demonstrating the QuantStrata library's capabilities for FX derivatives pricing, risk management, and market data handling.

## Directory Structure

```
examples/
├── fundamentals/     # Core market data concepts
├── pricing/          # Option pricing methods
├── risk/             # Risk management and scenarios
└── showcase/         # Visual demonstrations
```

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

**Start here** if you're new to the library.

---

## Part 2: Pricing

Learn to price FX options using different methods.

| Script | Description |
|--------|-------------|
| `01_single_fx_vanilla.py` | BSM, Monte Carlo, Finite Difference comparison |
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

**Key concepts:** Scenario P&L, Greeks validation, risk reports.

---

## Part 4: Showcase

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
# Run a specific example
cd examples/fundamentals
python 01_market_ids_and_quotes.py

# Or from anywhere
python examples/fundamentals/01_market_ids_and_quotes.py
```

### Dependencies

Most examples require only:
- numpy
- matplotlib
- scipy

The QuantStrata library itself (`src/`).

---

## Learning Path

### For Beginners

1. Start with `fundamentals/01_market_ids_and_quotes.py`
2. Work through all fundamentals in order
3. Move to `pricing/01_single_fx_vanilla.py`
4. Explore exotic options and portfolios

### For Practitioners

1. Skim fundamentals for API reference
2. Focus on `pricing/` for method comparison
3. Deep dive into `risk/` for production patterns

### For Interviewers/Reviewers

1. Check `showcase/` for visual demonstrations
2. Review `docs/mathematics/` for theory
3. Examine test coverage in `tests/`

---

## Code Style

All examples follow consistent patterns:

- **Docstrings**: Clear description at top of each file
- **Sections**: Numbered sections with headers
- **Print output**: Explanatory text and results
- **Plots**: Professional matplotlib visualizations
- **Summary**: Key takeaways at the end

---

## Generating Plots

Examples that generate plots save them to the current directory:

```bash
cd examples/fundamentals
python 03_volatility_surfaces.py
# Creates: volatility_surface.png
```

---

## Related Documentation

- `docs/mathematics/` - Detailed mathematical derivations
- `docs/notebooks/` - Interactive Jupyter notebooks
- `docs/PHASE_1_COMPLETE.md` - Phase 1 summary

---

*QuantStrata Examples | Phase 1 Complete*
