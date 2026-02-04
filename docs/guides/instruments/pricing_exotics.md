# How to Price Exotic Structured Products

This guide shows how to price **Cliquet**, **Autocallable**, and **Range Accrual** products using the library’s instruments, market data types, and Monte Carlo pricers.

---

## Prerequisites

- Market inputs: spot (or initial rate), volatility, risk-free rate; for range accrual, Hull-White parameters.
- Instrument parameters: reset/observation dates, barriers, caps/floors, notional, accrual rate.

---

## 1. Cliquet Options

### 1.1 Build the instrument

```python
from datetime import date
from src.instruments.equity.options.cliquet import EquityCliquetOption

# Monthly resets over one year
start = date(2025, 1, 1)
end = date(2026, 1, 1)
reset_dates = [date(2025, m, 1) for m in range(1, 13)]

cliquet = EquityCliquetOption(
    underlying_id="SPY",
    notional=1_000_000,
    start_date=start,
    end_date=end,
    reset_dates=reset_dates,
    local_cap=0.03,    # 3% per period max
    local_floor=-0.01, # -1% per period min
    global_cap=0.20,   # 20% total max
    global_floor=0.0,  # 0% total min (principal protected)
    participation=1.0,
)
```

### 1.2 Prepare market data

```python
from src.pricers.equity.cliquet_gbm_mc import CliquetMarketData

market = CliquetMarketData(
    spot=100.0,
    volatility=0.20,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    valuation_date=start,
)
```

### 1.3 Price and read result

```python
from src.pricers.equity.cliquet_gbm_mc import EquityCliquetGbmMcPricer, CliquetMcConfig

config = CliquetMcConfig(n_paths=50_000, seed=42, compute_greeks=True)
pricer = EquityCliquetGbmMcPricer(config=config)

result = pricer.price(cliquet, market)

print(f"Price: {result.price:,.2f}")
print(f"SE:    {result.standard_error:.4f}")
print(f"Delta: {result.delta}")
print(f"Vega:  {result.vega}")
```

---

## 2. Autocallable Options

### 2.1 Build the instrument

```python
from src.instruments.equity.options.autocallable import EquityAutocallableOption

observation_dates = [date(2025, 4, 1), date(2025, 7, 1), date(2025, 10, 1), date(2026, 1, 1)]

autocall = EquityAutocallableOption(
    underlying_id="SPY",
    notional=1_000_000,
    start_date=date(2025, 1, 1),
    maturity_date=date(2026, 1, 1),
    observation_dates=observation_dates,
    autocall_barrier=1.0,   # 100% of initial
    coupon_barrier=0.8,     # 80% of initial
    put_barrier=0.7,        # 70% of initial
    coupon_rate=0.10,       # 10% p.a.
)
```

### 2.2 Market data and pricing

```python
from src.pricers.equity.autocallable_gbm_mc import (
    EquityAutocallableGbmMcPricer,
    AutocallableMarketData,
)

market = AutocallableMarketData(
    spot=100.0,
    volatility=0.22,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    valuation_date=date(2025, 1, 1),
)

pricer = EquityAutocallableGbmMcPricer(n_paths=50_000, seed=42)
result = pricer.price(autocall, market)

print(f"Price: {result.price:,.2f}")
```

---

## 3. Range Accrual Notes

### 3.1 Build the instrument

```python
from src.instruments.ir.options.range_accrual import IrRangeAccrualNote, ObservationFrequency

note = IrRangeAccrualNote(
    notional=1_000_000,
    start_date=date(2025, 1, 1),
    maturity_date=date(2026, 1, 1),
    range_lower=0.03,   # 3%
    range_upper=0.05,   # 5%
    accrual_rate=0.06,  # 6% when in range
    reference_rate_id="USD3M",
    observation_frequency=ObservationFrequency.DAILY,
)
```

### 3.2 Market data (Hull-White) and pricing

```python
from src.pricers.ir.range_accrual_hw_mc import (
    IrRangeAccrualHwMcPricer,
    RangeAccrualMarketData,
)

market = RangeAccrualMarketData(
    initial_rate=0.04,
    mean_reversion=0.03,
    volatility=0.01,
    long_term_rate=0.04,
    discount_rate=0.05,
    valuation_date=date(2025, 1, 1),
)

pricer = IrRangeAccrualHwMcPricer(n_paths=50_000, seed=42)
result = pricer.price(note, market)

print(f"Price: {result.price:,.2f}")
print(f"Expected accrual fraction: {result.expected_accrual_fraction:.4f}")
print(f"Expected coupon: {result.expected_coupon:,.2f}")
```

---

## 4. Tips

- **Paths:** Increase `n_paths` (e.g. 100k–500k) for tighter standard errors and more stable Greeks.
- **Seeds:** Use a fixed `seed` for reproducibility.
- **Greeks:** Cliquet and autocallable support bump-and-revalue Greeks; range accrual exposes delta/vega where implemented.
- **Dates:** Ensure reset/observation dates are consistent with start/maturity and valuation date.

---

## 5. Example script

A full runnable example that prices all three product types is in:

- `examples/pricing/exotic_structured_products.py`

Run it from the project root (e.g. `python examples/pricing/exotic_structured_products.py`) after installing dependencies.

---

*Reference: [Exotic products (reference)](../../reference/instruments/exotic_products.md).*
