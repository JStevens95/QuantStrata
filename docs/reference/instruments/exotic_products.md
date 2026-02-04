# Exotic Products Reference

Technical specification and API reference for structured exotic products: Cliquet, Autocallable, and Range Accrual.

---

## Overview

The library implements three families of path-dependent exotic products commonly used in equity-linked and interest-rate structured notes:

| Product | Asset Class | Pricer | Payoff Type |
|---------|-------------|--------|-------------|
| **Cliquet** | Equity / FX | GBM Monte Carlo | Capped/floored periodic returns |
| **Autocallable** | Equity | GBM Monte Carlo | Early redemption + coupon + put |
| **Range Accrual** | IR | Hull-White Monte Carlo | Accrual in range |

All require **Monte Carlo** pricing due to path-dependency; no closed-form formulae are used.

---

## 1. Cliquet Options

### 1.1 Definition

A **cliquet** (ratchet) pays on the sum of **capped and floored periodic returns** of the underlying. At each reset date the period return is computed, clamped to a local cap/floor, then accumulated; at maturity the total is clamped to a global cap/floor and multiplied by a participation rate.

**Payoff (conceptual):**

- Local return at reset \(i\):  
  \(R_i = \text{clamp}(S_{t_i}/S_{t_{i-1}} - 1,\; \text{local\_floor},\; \text{local\_cap})\)
- Global return:  
  \(R = \text{clamp}(\sum_i R_i,\; \text{global\_floor},\; \text{global\_cap})\)
- Payoff:  
  \(\text{notional} \times \text{participation} \times \max(R,\; 0)\)

### 1.2 Instrument API

**Module:** `src.instruments.equity.options.cliquet`

| Class | Description |
|-------|-------------|
| `EquityCliquetOption` | Equity cliquet (reset dates, local/global cap/floor, participation) |
| `FxCliquetOption` | FX cliquet (same structure, pair identifier) |

**Key attributes (equity):**

- `underlying_id`, `notional`, `start_date`, `end_date`
- `reset_dates`: list of reset dates
- `local_cap`, `local_floor`, `global_cap`, `global_floor`
- `participation`: multiplier on global return

### 1.3 Payoff

**Module:** `src.models.payoffs.cliquet`

- `CliquetPayoff`: builds from instrument and time grid; evaluates on simulated paths (capped/floored returns, then global clamp).

### 1.4 Pricer

**Module:** `src.pricers.equity.cliquet_gbm_mc`

| Class | Description |
|-------|-------------|
| `EquityCliquetGbmMcPricer` | Prices `EquityCliquetOption` under GBM |
| `FxCliquetGbmMcPricer` | Prices `FxCliquetOption` under GBM |
| `CliquetMarketData` | Market inputs: `spot`, `volatility`, `risk_free_rate`, `dividend_yield`, `valuation_date` |
| `CliquetPricingResult` | `price`, `standard_error`, `delta`, `gamma`, `vega`, `rho`, `theta`, `n_paths`, `elapsed_seconds` |
| `CliquetMcConfig` | `n_paths`, `seed`, `antithetic`, `compute_greeks`, bump sizes |

Greeks are computed by bump-and-revalue (spot for delta/gamma, vol for vega, rate for rho).

---

## 2. Autocallable Options

### 2.1 Definition

An **autocallable** can **terminate early** (autocall) if the underlying is at or above an autocall barrier on an observation date. It typically pays:

- **Autocall:** 100% of notional + coupon if spot ≥ autocall barrier on any observation date (early redemption).
- **Coupon:** Periodic coupon if spot ≥ coupon barrier on observation dates.
- **Maturity:** If not called, at maturity: 100% if spot ≥ put barrier; otherwise (spot/initial − 1) × notional (put payoff).

### 2.2 Instrument API

**Module:** `src.instruments.equity.options.autocallable`

| Class | Description |
|-------|-------------|
| `EquityAutocallableOption` | Observation dates, autocall/coupon/put barriers, coupon rate |

**Key attributes:**

- `underlying_id`, `notional`, `start_date`, `maturity_date`
- `observation_dates`: list of observation dates
- `autocall_barrier`, `coupon_barrier`, `put_barrier` (typically as fractions of initial spot)
- `coupon_rate`: annualised coupon when in range

### 2.3 Payoff

**Module:** `src.models.payoffs.autocallable`

- `AutocallablePayoff`: path-dependent; first observation where spot ≥ autocall barrier triggers early redemption; otherwise coupon accrual and final put payoff.

### 2.4 Pricer

**Module:** `src.pricers.equity.autocallable_gbm_mc`

| Class | Description |
|-------|-------------|
| `EquityAutocallableGbmMcPricer` | Prices `EquityAutocallableOption` under GBM |
| `AutocallableMarketData` | `spot`, `volatility`, `risk_free_rate`, `dividend_yield`, `valuation_date` |
| `AutocallablePricingResult` | `price`, `standard_error`, Greeks, optional autocall/coupon stats |
| `AutocallableMcConfig` | `n_paths`, `seed`, `antithetic`, `compute_greeks` |

---

## 3. Range Accrual Notes

### 3.1 Definition

A **range accrual** note pays interest only for days when a **reference rate** (or index) is within a specified range \([L,\; U]\). The coupon is:

\[
\text{Coupon} = \text{notional} \times \text{accrual\_rate} \times \frac{\text{days in range}}{\text{total days}}
\]

### 3.2 Instrument API

**Module:** `src.instruments.ir.options.range_accrual`

| Class / Enum | Description |
|--------------|-------------|
| `IrRangeAccrualNote` | Notional, start/maturity, range_lower, range_upper, accrual_rate, reference_rate_id, observation_frequency |
| `ObservationFrequency` | `DAILY`, `WEEKLY`, `MONTHLY` |

### 3.3 Payoff

**Module:** `src.models.payoffs.range_accrual`

- `RangeAccrualPayoff`: evaluates on short-rate paths; counts observations in range and computes accrual fraction.

### 3.4 Pricer

**Module:** `src.pricers.ir.range_accrual_hw_mc`

| Class | Description |
|-------|-------------|
| `IrRangeAccrualHwMcPricer` | Prices `IrRangeAccrualNote` under Hull-White one-factor short rate |
| `RangeAccrualMarketData` | `initial_rate`, Hull-White params (`mean_reversion`, `volatility`, `long_term_rate`), `discount_rate`, `valuation_date` |
| `RangeAccrualPricingResult` | `price`, `standard_error`, `delta`, `vega`, `expected_accrual_fraction`, `prob_full_accrual`, `prob_zero_accrual`, `expected_coupon`, `n_paths`, `n_observations` |

---

## 4. Naming Conventions

Pricers follow the pattern: `{product}_{model}_{method}.py`

- **Cliquet:** `cliquet_gbm_mc.py` (GBM, Monte Carlo)
- **Autocallable:** `autocallable_gbm_mc.py` (GBM, Monte Carlo)
- **Range Accrual:** `range_accrual_hw_mc.py` (Hull-White, Monte Carlo)

---

## 5. Dependencies

- **Instruments:** `src.instruments.equity.options.cliquet`, `autocallable`; `src.instruments.ir.options.range_accrual`
- **Payoffs:** `src.models.payoffs.cliquet`, `autocallable`, `range_accrual`
- **Pricers:** `src.pricers.equity.cliquet_gbm_mc`, `autocallable_gbm_mc`; `src.pricers.ir.range_accrual_hw_mc`
- **Numerics:** NumPy; GBM and Hull-White path generation in pricers/models

---

*See also: [Pricing exotics (guide)](../../guides/instruments/pricing_exotics.md), roadmap Phase 7.3.*
