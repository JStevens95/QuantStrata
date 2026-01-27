# Phase 1.1 Implementation Plan: Additional FX Products

**Phase:** Phase 1, Section 1.1  
**Goal:** Implement 4 additional FX path-dependent products  
**Timeline:** Weeks 1-4  
**Status:** In Progress

---

## Overview

This document tracks the implementation of Phase 1.1 from the ROADMAP.md. We will implement:

1. **Asian Options** (average price/rate options)
2. **Lookback Options** (floating strike)
3. **Double Barrier Options**
4. **FX Touch Options** (one-touch, no-touch)

Each product follows the standard implementation pattern:
1. Create Instrument dataclass
2. Create Payoff implementation (path-dependent)
3. Create MC Pricer
4. Register in PricerRegistry
5. Write comprehensive unit tests
6. Add technical documentation

---

## Implementation Order

### 1. Asian Options ✅ (In Progress)
**Priority:** High (foundational path-dependent product)  
**Complexity:** Medium

**Components:**
- [ ] Instrument: `EuropeanFxAsianOption` (`src/instruments/fx/options/asian.py`)
- [ ] Payoff: `AsianPayoff` (`src/models/payoffs/asian.py`)
- [ ] Pricer: `FxEuropeanAsianMcPricer` (`src/pricers/fx/asian_mc.py`)
- [ ] Tests: `tests/unit/pricers/fx/test_fx_asian_mc_pricer.py`
- [ ] Tests: `tests/unit/models/payoffs/test_asian_payoff.py`
- [ ] Documentation: `docs/mathematics/asian_options.md`
- [ ] Registry: Update `DefaultPricerRegistry`

**Key Features:**
- Average price Asian (arithmetic mean)
- Average strike Asian (optional, future)
- Discrete monitoring (daily/weekly/monthly)
- MC pricing only (FD requires 2D PDE, deferred)

---

### 2. Lookback Options
**Priority:** Medium  
**Complexity:** Medium

**Components:**
- [ ] Instrument: `EuropeanFxLookbackOption` (`src/instruments/fx/options/lookback.py`)
- [ ] Payoff: `LookbackPayoff` (`src/models/payoffs/lookback.py`)
- [ ] Pricer: `FxEuropeanLookbackMcPricer` (`src/pricers/fx/lookback_mc.py`)
- [ ] Tests: `tests/unit/pricers/fx/test_fx_lookback_mc_pricer.py`
- [ ] Tests: `tests/unit/models/payoffs/test_lookback_payoff.py`
- [ ] Documentation: `docs/mathematics/lookback_options.md`
- [ ] Registry: Update `DefaultPricerRegistry`

**Key Features:**
- Floating strike lookback (min/max over path)
- Fixed strike lookback (optional)
- Discrete monitoring
- MC pricing only

---

### 3. Double Barrier Options
**Priority:** Medium  
**Complexity:** Medium-High

**Components:**
- [ ] Instrument: `EuropeanFxDoubleBarrierOption` (`src/instruments/fx/options/double_barrier.py`)
- [ ] Payoff: `DoubleBarrierPayoff` (`src/models/payoffs/double_barrier.py`)
- [ ] Pricer: `FxEuropeanDoubleBarrierMcPricer` (`src/pricers/fx/double_barrier_mc.py`)
- [ ] Tests: `tests/unit/pricers/fx/test_fx_double_barrier_mc_pricer.py`
- [ ] Tests: `tests/unit/models/payoffs/test_double_barrier_payoff.py`
- [ ] Documentation: `docs/mathematics/double_barrier_options.md`
- [ ] Registry: Update `DefaultPricerRegistry`

**Key Features:**
- Upper and lower barriers
- Knock-out / knock-in styles
- Rebate handling
- MC pricing only (FD with absorbing boundaries deferred)

---

### 4. FX Touch Options
**Priority:** Medium  
**Complexity:** Low-Medium

**Components:**
- [ ] Instrument: `EuropeanFxTouchOption` (`src/instruments/fx/options/touch.py`)
- [ ] Payoff: `TouchPayoff` (`src/models/payoffs/touch.py`)
- [ ] Pricer: `FxEuropeanTouchMcPricer` (`src/pricers/fx/touch_mc.py`)
- [ ] Tests: `tests/unit/pricers/fx/test_fx_touch_mc_pricer.py`
- [ ] Tests: `tests/unit/models/payoffs/test_touch_payoff.py`
- [ ] Documentation: `docs/mathematics/touch_options.md`
- [ ] Registry: Update `DefaultPricerRegistry`

**Key Features:**
- One-touch (pays if barrier touched)
- No-touch (pays if barrier NOT touched)
- Binary payout
- MC pricing only

---

## Implementation Standards

### Code Standards
- ✅ Clean, efficient code
- ✅ Intuitive function/variable names
- ✅ Detailed docstrings with type hints
- ✅ Line-by-line comments explaining logic
- ✅ Follow existing patterns (barrier.py, european_mc.py)

### Testing Standards
- ✅ Unit tests for payoffs (vectorization, edge cases)
- ✅ Unit tests for pricers (basic pricing, edge cases)
- ✅ Parity tests where applicable (e.g., Asian vs vanilla limits)
- ✅ Regression tests with known values
- ✅ Test coverage >90%

### Documentation Standards
- ✅ Mathematical derivations
- ✅ Key formulas highlighted
- ✅ Important points to remember
- ✅ Interview talking points
- ✅ Methodology explanations

---

## Progress Tracking

### Asian Options
- [ ] Instrument implementation
- [ ] Payoff implementation
- [ ] Pricer implementation
- [ ] Payoff tests
- [ ] Pricer tests
- [ ] Registry registration
- [ ] Technical documentation
- [ ] Code review & cleanup

### Lookback Options
- [ ] Instrument implementation
- [ ] Payoff implementation
- [ ] Pricer implementation
- [ ] Payoff tests
- [ ] Pricer tests
- [ ] Registry registration
- [ ] Technical documentation
- [ ] Code review & cleanup

### Double Barrier Options
- [ ] Instrument implementation
- [ ] Payoff implementation
- [ ] Pricer implementation
- [ ] Payoff tests
- [ ] Pricer tests
- [ ] Registry registration
- [ ] Technical documentation
- [ ] Code review & cleanup

### FX Touch Options
- [ ] Instrument implementation
- [ ] Payoff implementation
- [ ] Pricer implementation
- [ ] Payoff tests
- [ ] Pricer tests
- [ ] Registry registration
- [ ] Technical documentation
- [ ] Code review & cleanup

---

## Notes

- All products are path-dependent, so MC pricing is required
- FD pricing (2D PDE) is deferred to future phases
- Follow the existing barrier option pattern closely
- Ensure payoff library is the single source of truth
- Maintain interface contracts (no breaking changes)
