# Advanced Analytics & Reporting – Technical Reference

**Modules:** `src.risk.reporting`, `src.core.reporting.plots`

This document describes front-office risk reports (Greeks surfaces, PnL attribution, VaR/CVaR summaries) and publication-quality visualisation (styling, Greeks heatmaps, scenario PnL plots, export).

---

## Overview

Phase 5.6 adds:

- **Front-office risk reports:** VaR/CVaR summary report, combined `RiskReport` (optional Greeks, VaR, attribution, scenario), Greeks surface plotter (2D heatmap).
- **Publication-quality visualisation:** Shared report style (`apply_report_style`), export helper (PNG + PDF), PnL-by-scenario and attribution bar plots; vol surface plots use report style.

---

## VaR/CVaR Summary Report

**Module:** `src.risk.reporting.var_summary`

`VarSummaryReport` wraps `VarResult` with optional factor-level breakdown and consistent export:

```python
from src.risk.var import compute_var, VarConfig
from src.risk.reporting.var_summary import VarSummaryReport, VarBreakdownRow, build_var_summary_report

config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
result = compute_var(config, pnl_series=pnl_series)
# Optional breakdown for parametric (factor contributions)
breakdown = [VarBreakdownRow("spot", 100.0), VarBreakdownRow("vol", 50.0)]  # if available
summary = build_var_summary_report(result, breakdown=breakdown)

summary.to_console()
summary.to_dicts()   # nested dict for JSON
summary.to_csv()
```

---

## Combined Risk Report

**Module:** `src.risk.reporting.risk_report`

`RiskReport` holds optional sections: Greeks summary, VaR summary, attribution report, scenario report. All optional so callers can build partial reports.

```python
from src.risk.reporting.risk_report import RiskReport
from src.risk.sensitivities.aggregation import aggregate_sensitivities
from src.risk.reporting.scenario_report import build_scenario_report
from src.risk.reporting.var_summary import build_var_summary_report

# After computing sensitivities, VaR, scenarios, attribution:
greeks_summary = aggregate_sensitivities(sensitivities_report, include_per_market_id=True)
var_summary = build_var_summary_report(var_result)
scenario_report = build_scenario_report(scenario_runner_result)
# attribution_report from attribute_portfolio_scenarios()

report = RiskReport(
    greeks_summary=greeks_summary,
    var_summary=var_summary,
    attribution_report=attribution_report,
    scenario_report=scenario_report,
)
report.to_console()
report.to_dicts()   # nested dict for JSON
report.to_csv()
```

Instrument-level breakdown is available via `GreeksSummary.per_market_id` and attribution contributions; the report format documents this.

---

## Greeks Surface (2D Heatmap)

**Module:** `src.core.reporting.plots.risk.greeks_surface`

Plot a single greek (e.g. delta, vega) as a function of expiry and strike:

```python
from src.core.reporting.plots.risk import plot_greeks_surface, greek_grid_from_sensitivities

# Option 1: Caller provides grid (e.g. from FD pricer or sweep)
expiries = np.array([0.25, 0.5, 1.0])
strikes = np.array([0.95, 1.0, 1.05])
delta_grid = np.array([[...]])  # shape (len(expiries), len(strikes))
fig = plot_greeks_surface(expiries, strikes, delta_grid, greek_name="delta")

# Option 2: Build grid from SensitivitiesReport when SensitivityKey has expiry/strike
expiries, strikes, z = greek_grid_from_sensitivities(sensitivities_report, "delta")
if expiries.size > 0 and strikes.size > 0:
    fig = plot_greeks_surface(expiries, strikes, z, greek_name="delta")
```

---

## PnL by Scenario and Attribution Bars

**Module:** `src.core.reporting.plots.risk.pnl_scenario`

```python
from src.core.reporting.plots.risk import plot_pnl_by_scenario, plot_attribution_bars

# Bar chart of PnL per scenario (excludes base by default)
fig1 = plot_pnl_by_scenario(scenario_report, title="Stress PnL")

# Bar chart of factor contributions for one scenario
fig2 = plot_attribution_bars(attribution_report, scenario="spot_up_1pct")
```

---

## Report Styling and Export

**Style:** `src.core.reporting.plots.style`

- `apply_report_style(ax)` — grid, spines, tick size for publication-quality axes.
- `get_report_figsize()` — default (9, 5) for report figures.
- `report_rc()` — context manager for report-style rcParams.

**Export:** `src.core.reporting.plots.utils`

- `PlotConfig.save_pdf` — when `save=True`, also save PDF (default True).
- `save_report_figures(figures, out_dir, prefix="report", dpi=160, save_pdf=True)` — batch save with names `report_01_vol_surface.png`, `report_01_vol_surface.pdf`, etc.

```python
from src.core.reporting.plots.utils import PlotConfig, render_fig, save_report_figures

cfg = PlotConfig(save=True, out_dir=Path("outputs/reports"), save_pdf=True)
render_fig(fig, cfg=cfg, filename="var_summary.png")  # saves PNG + PDF

# Batch export
figures = [(fig1, "vol_surface"), (fig2, "greeks_delta")]
save_report_figures(figures, "outputs/reports", prefix="risk")
```

---

## Vol Surface Plots

Vol surface heatmap and smile slices in `src.core.reporting.plots.marketdata.surfaces` now use `apply_report_style(ax)` for consistent publication quality. Use `PlotConfig` and `render_fig` (or `save_report_figures`) to save PNG + PDF.

---

## Conventions

- No pandas: reports use dataclasses and `.to_dicts()` / `.to_csv()` for export.
- Styling is applied per-axes; no global rcParams changes by default (use `report_rc()` context if desired).
- Greeks surface data source is caller-provided; optional `greek_grid_from_sensitivities` when sensitivities have expiry/strike buckets.
