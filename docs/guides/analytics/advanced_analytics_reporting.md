# Advanced Analytics & Reporting – User Guide

This guide shows how to produce front-office risk reports and publication-quality plots: VaR/CVaR summary, combined risk report, Greeks surfaces, PnL by scenario, and figure export.

---

## Quick Start

### 1. VaR summary and combined risk report

After computing VaR (historical, parametric, or MC), wrap the result in a summary report and optionally combine with Greeks and scenario reports:

```python
from pathlib import Path
from src.risk.var import compute_var, VarConfig
from src.risk.reporting.var_summary import build_var_summary_report
from src.risk.reporting.risk_report import RiskReport
from src.risk.sensitivities.aggregation import aggregate_sensitivities
from src.risk.reporting.scenario_report import build_scenario_report

# Assume you have: sensitivities_report, scenario_runner_result, attribution_report
var_config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
var_result = compute_var(var_config, pnl_series=pnl_series)
var_summary = build_var_summary_report(var_result)

greeks_summary = aggregate_sensitivities(sensitivities_report, include_per_market_id=True)
scenario_report = build_scenario_report(scenario_runner_result)

report = RiskReport(
    greeks_summary=greeks_summary,
    var_summary=var_summary,
    attribution_report=attribution_report,  # optional
    scenario_report=scenario_report,
)
print(report.to_console())
# Export: report.to_dicts(), report.to_csv()
```

### 2. Greeks surface (2D heatmap)

Provide a grid of greek values (e.g. from an FD pricer or sensitivity sweep) and plot:

```python
import numpy as np
from src.core.reporting.plots.risk import plot_greeks_surface

expiries = np.array([0.25, 0.5, 1.0])
strikes = np.array([0.95, 1.0, 1.05])
delta_grid = np.array([...])  # shape (3, 3)

fig = plot_greeks_surface(expiries, strikes, delta_grid, greek_name="delta")
# Save with PlotConfig + render_fig, or save_report_figures
```

If your sensitivities report has expiry/strike in each key, use the helper:

```python
from src.core.reporting.plots.risk import greek_grid_from_sensitivities, plot_greeks_surface

expiries, strikes, z = greek_grid_from_sensitivities(sensitivities_report, "delta")
if expiries.size > 0:
    fig = plot_greeks_surface(expiries, strikes, z, greek_name="delta")
```

### 3. PnL by scenario and attribution bars

```python
from src.core.reporting.plots.risk import plot_pnl_by_scenario, plot_attribution_bars

fig1 = plot_pnl_by_scenario(scenario_report, title="Stress PnL")
fig2 = plot_attribution_bars(attribution_report, scenario="spot_up_1pct")
```

### 4. Export figures for reports

```python
from pathlib import Path
from src.core.reporting.plots.utils import PlotConfig, render_fig, save_report_figures

# Single figure: PNG + PDF when save=True
cfg = PlotConfig(save=True, out_dir=Path("outputs/reports"), save_pdf=True)
render_fig(fig, cfg=cfg, filename="greeks_delta.png")

# Batch: report_01_vol_surface.png/pdf, report_02_greeks_delta.png/pdf, ...
figures = [(fig_vol, "vol_surface"), (fig_delta, "greeks_delta"), (fig_pnl, "pnl_by_scenario")]
save_report_figures(figures, Path("outputs/reports"), prefix="risk")
```

---

## Styling

All risk and analytics plots use `apply_report_style(ax)` from `src.core.reporting.plots.style` for consistent grid, spines, and tick size. Vol surface heatmap and smile slices use the same style. For batch figure creation with shared rcParams, use the `report_rc()` context manager.

---

## See also

- [Advanced Analytics & Reporting (reference)](../reference/advanced_analytics_reporting.md) — API and modules.
- [Risk Infrastructure](../reference/risk/risk_infrastructure.md) — VaR, Greeks aggregation, stress testing.
- [Risk Framework](risk/risk_framework.md) — User-facing risk guide.
