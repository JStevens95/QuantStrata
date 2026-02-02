# Risk Framework - User Guide

This guide shows how to use QuantStrata's risk infrastructure: Value-at-Risk (VaR), Greeks aggregation, and stress testing.

---

## Quick Start

### Historical VaR from P&L series

```python
import numpy as np
from src.risk.var import historical_var, VarConfig

# Daily P&L series (e.g. from revaluation or backtest)
pnl_series = np.random.randn(252) * 10_000  # 1 year of daily P&L

config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
result = historical_var(pnl_series, config)

print(f"99% 1-day VaR: {result.var:,.2f}")
print(f"CVaR (expected shortfall): {result.cvar:,.2f}")
```

### Greeks aggregation

```python
from src.risk.sensitivities.engine import compute_sensitivities
from src.risk.sensitivities.aggregation import aggregate_sensitivities

# After computing sensitivities for a portfolio
report = compute_sensitivities(portfolio, market, portfolio_pricer, config=...)
summary = aggregate_sensitivities(report, include_per_market_id=True)

print("Totals by greek:", summary.totals_by_greek)
print("Totals by risk factor:", summary.totals_by_factor)
```

### Stress testing with preset scenarios

```python
from src.risk.scenarios.generation import preset_stress_pack
from src.risk.scenarios.runner import run_portfolio_scenarios

pack = preset_stress_pack(
    "crisis_style",
    spot_id=spot_id,
    vol_id=vol_id,
    domestic_curve_id=domestic_curve_id,
)
result = run_portfolio_scenarios(
    portfolio, base_market, portfolio_pricer,
    list(pack.scenarios.values()),
)
print(result.pnl_by_scenario)
```

---

## Value-at-Risk (VaR)

### Methods

| Method | Input | Use case |
|--------|--------|----------|
| **Historical** | P&L time series | Non-parametric; uses empirical distribution |
| **Parametric** | Sensitivities + factor vols (and optional correlation) | Delta-normal; fast, assumes normal P&L |
| **Monte Carlo** | Portfolio, market, pricer, factor model | Full revaluation; flexible, more expensive |

### VarConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence` | 0.99 | Confidence level (e.g. 99% VaR) |
| `horizon_days` | 1 | VaR horizon; scaling uses sqrt(horizon_days) for i.i.d. |
| `method` | "historical" | "historical", "parametric", or "mc" |

### Facade: compute_var

Use `compute_var` to dispatch by config method:

```python
from src.risk.var import compute_var, VarConfig

# Historical
config = VarConfig(method="historical", confidence=0.99)
result = compute_var(config, pnl_series=pnl_series)

# Parametric (needs sensitivities_report and factor_volatilities)
config = VarConfig(method="parametric", confidence=0.99)
result = compute_var(config, sensitivities_report=report, factor_volatilities=vols)

# Monte Carlo (needs portfolio, market, portfolio_pricer, factor_model)
config = VarConfig(method="mc", confidence=0.99)
result = compute_var(config, portfolio=portfolio, market=market,
                     portfolio_pricer=pricer, factor_model=factor_model, n_paths=5000)
```

---

## Greeks Aggregation

`aggregate_sensitivities` turns a `SensitivitiesReport` (per-row greeks) into a `GreeksSummary`:

- **totals_by_greek:** One total per greek (delta, gamma, vega, rho_domestic, etc.).
- **totals_by_factor:** Totals by risk factor: spot (delta, gamma), vol (vega), rate (rho_*).
- **per_market_id:** Optional list of (market_id, greek, value) for decomposition.

Use for risk reporting and factor decomposition without changing the sensitivities engine.

---

## Stress Testing

### CompositeShock (multi-factor)

Apply several shocks in one scenario:

```python
from src.marketdata.scenarios.shocks import CompositeShock, SpotShock, VolShock, ParallelRateShock

composite = CompositeShock(
    name="crisis_style",
    shocks=[
        SpotShock("s", spot_id, bump=-0.15, bump_mode="relative"),
        VolShock("v", vol_id, bump=0.30, bump_mode="relative"),
        ParallelRateShock("r", curve_id, rate_shift=-0.005),
    ],
)
shocked_market = composite.apply(base_market)
```

### Preset stress packs

`preset_stress_pack(name, spot_id=..., vol_id=..., domestic_curve_id=...)` returns a `ScenarioPack` for:

- **spot_down_10**, **spot_up_10** — spot moves (requires `spot_id`)
- **vol_up_5**, **vol_up_25** — vol moves (requires `vol_id`)
- **rates_up_25bp**, **rates_down_50bp** — curve shifts (requires `domestic_curve_id`)
- **crisis_style** — composite: spot -15%, vol +30%, rates -50bp (requires spot_id, vol_id, domestic_curve_id)

### Historical-based shocks

Build shocks from time series (e.g. worst 1-day move at a percentile):

```python
from src.risk.scenarios.generation import shocks_from_historical_series

series_by_id = {market_id: np.array([...])}  # levels over time
shocks = shocks_from_historical_series(
    series_by_id,
    percentile=5.0,
    use_relative=True,
    horizon=1,
)
# Wrap in CompositeShock for one multi-factor "historical worst" scenario if desired
```

---

## See also

- **Technical reference:** [Risk Infrastructure](../../reference/risk/risk_infrastructure.md)
- **Tutorial:** [Risk Introduction](../../tutorials/risk/risk_introduction.ipynb)
