# Risk Infrastructure - Technical Reference

**Module:** `src.risk`

This document describes the risk infrastructure: Value-at-Risk (VaR), Greeks aggregation, and stress testing (scenario generation). For front-office risk reports (VaR summary, combined RiskReport, Greeks surfaces, PnL-by-scenario plots, styling and export), see [Advanced Analytics & Reporting](../advanced_analytics_reporting.md).

---

## Overview

The risk module provides:

- **Value-at-Risk (VaR):** Historical, Parametric (delta-normal), Monte Carlo; portfolio-level.
- **Greeks Aggregation:** Totals by greek and by risk factor (spot, vol, rate); optional per-market_id breakdown.
- **Stress Testing:** Multi-factor scenarios via `CompositeShock`; preset and historical scenario generation.

---

## Architecture

```
src/risk/
├── var/
│   ├── config.py       # VarConfig, VarResult
│   ├── historical.py   # historical_var
│   ├── parametric.py   # parametric_var
│   ├── mc.py           # mc_var, DiagonalFactorModel
│   └── runner.py       # compute_var (facade)
├── sensitivities/
│   ├── engine.py       # compute_sensitivities
│   ├── result.py       # SensitivitiesReport, SensitivityRow
│   └── aggregation.py  # aggregate_sensitivities, GreeksSummary
├── scenarios/
│   ├── runner.py       # run_portfolio_scenarios
│   └── generation.py    # preset_stress_pack, shocks_from_historical_series
├── attribution/        # Greeks-based PnL attribution
└── reporting/          # ScenarioReport

src/marketdata/scenarios/
└── shocks.py           # SpotShock, VolShock, ParallelRateShock, CompositeShock
```

---

## Value-at-Risk (VaR)

### VarConfig and VarResult

```python
@dataclass(frozen=True, slots=True)
class VarConfig:
    confidence: float = 0.99   # e.g. 99% VaR
    horizon_days: int = 1
    method: str = "historical"  # "historical" | "parametric" | "mc"

@dataclass(frozen=True, slots=True)
class VarResult:
    var: float           # VaR (positive = loss at confidence)
    method: str
    confidence: float
    horizon_days: int
    cvar: Optional[float] = None  # Expected shortfall if computed
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Historical VaR

Input: P&L series (e.g. daily). VaR = negative of (1 - confidence) quantile of P&L; optionally scaled by sqrt(horizon_days).

```python
from src.risk.var import historical_var, VarConfig

config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
result = historical_var(pnl_series, config)
# result.var, result.cvar
```

### Parametric VaR (delta-normal)

Input: `SensitivitiesReport` and factor volatilities (and optional correlation). VaR = z(confidence) * sqrt(Γ' Σ Γ); scaled by sqrt(horizon_days).

```python
from src.risk.var import parametric_var, VarConfig
from src.risk.sensitivities.engine import compute_sensitivities

report = compute_sensitivities(portfolio, market, portfolio_pricer, ...)
factor_volatilities = {key: daily_vol for key in ...}
config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
result = parametric_var(report, factor_volatilities, config)
```

### Monte Carlo VaR

Input: portfolio, market, portfolio_pricer, and a `FactorModel` that provides `sample_shocks(n_paths)` and `build_shocked_market(base_market, shock_row)`.

```python
from src.risk.var import mc_var, DiagonalFactorModel, VarConfig

factor_model = DiagonalFactorModel(
    factor_order=[...],
    factor_volatilities={...},
    shock_builders={key: lambda v: SpotShock(...) for key in ...},
)
config = VarConfig(confidence=0.99, horizon_days=1, method="mc")
result = mc_var(portfolio, market, portfolio_pricer, factor_model, config, n_paths=10_000, seed=42)
```

### Facade: compute_var

Dispatch by `config.method`:

```python
from src.risk.var import compute_var, VarConfig

config = VarConfig(method="historical", ...)
result = compute_var(config, pnl_series=pnl_series)
```

---

## Greeks Aggregation

`aggregate_sensitivities` builds a `GreeksSummary` from a `SensitivitiesReport`:

- **totals_by_greek:** One total per greek (delta, vega, rho_domestic, ...).
- **totals_by_factor:** Totals by risk factor: spot (delta, gamma), vol (vega), rate (rho_*).
- **per_market_id:** Optional list of (market_id, greek, value) for decomposition.

```python
from src.risk.sensitivities.aggregation import aggregate_sensitivities, GreeksSummary

report = compute_sensitivities(portfolio, market, portfolio_pricer, ...)
summary = aggregate_sensitivities(report, include_per_market_id=True)
# summary.totals_by_greek, summary.totals_by_factor, summary.per_market_id
```

---

## Stress Testing

### CompositeShock (multi-factor)

Apply several shocks in sequence to the same base market:

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

Use with `run_portfolio_scenarios(portfolio, base_market, portfolio_pricer, [composite])`.

### Scenario generation

**Preset packs:** Return a `ScenarioPack` for predefined scenarios (spot_down_10, vol_up_5, crisis_style, etc.):

```python
from src.risk.scenarios.generation import preset_stress_pack

pack = preset_stress_pack(
    "crisis_style",
    spot_id=spot_id,
    vol_id=vol_id,
    domestic_curve_id=domestic_curve_id,
)
# pack.scenarios["crisis_style"] is a CompositeShock
```

**Historical-based shocks:** Build shocks from time series (e.g. worst 1-day move at a percentile):

```python
from src.risk.scenarios.generation import shocks_from_historical_series

series_by_id = {market_id: np.array([...])}  # levels over time
shocks = shocks_from_historical_series(series_by_id, percentile=5.0, use_relative=True)
# Wrap in CompositeShock for one multi-factor "historical worst" scenario if desired
```

---

## Conventions

- No pandas: use `np.ndarray` and dataclasses; export via `.to_dicts()` / `.to_csv()` where provided.
- Config/Result pattern: immutable dataclasses with `slots=True` where appropriate.
- Portfolio-level: VaR and aggregation are portfolio-level; scenario runner prices the full portfolio under each scenario.
