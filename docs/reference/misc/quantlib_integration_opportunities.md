# QuantLib Integration Opportunities

**Last Updated:** 2026-01-27  
**Status:** Planning Document

---

## Overview

This document outlines opportunities to enhance QuantStrata with additional QuantLib integrations. QuantLib is the industry-standard open-source library for quantitative finance, providing production-tested implementations that can serve as:

1. **Validation benchmarks** for our native implementations
2. **Alternative backends** for performance-critical paths
3. **Extended functionality** for edge cases
4. **Interview talking points** demonstrating industry tool familiarity

---

## Current QuantLib Integration

### Already Implemented

| Component | Location | Purpose |
|-----------|----------|---------|
| Curve Bootstrapping | `marketdata/curves/bootstrapper.py` | Alternative to native bootstrapper |
| SABR Calibration | `calibration/volatility_surface/quantlib/sabr_ql.py` | Vol smile calibration |
| Dupire Local Vol | `calibration/volatility_surface/quantlib/dupire_ql.py` | Local vol extraction |
| Market Adapters | `marketdata/integration/quantlib/adaptors/` | QL object conversion |
| Yield Curve Adapters | `marketdata/integration/quantlib/adaptors/curves.py` | Term structure conversion |
| Vol Surface Adapters | `marketdata/integration/quantlib/adaptors/vols.py` | Vol surface conversion |

---

## Recommended New Integrations

### Priority 1: High Value (Implement First)

#### 1.1 Heston Model Pricing

**Why:** QuantLib has highly optimized Heston pricing via:
- Semi-analytic (characteristic function)
- FFT methods
- Monte Carlo with variance reduction

**Implementation Plan:**

```python
# src/pricers/equity/european_heston_ql.py

import QuantLib as ql
from src.instruments.equity.options.vanilla import EquityVanillaOption

class HestonQLPricer:
    """
    QuantLib-backed Heston pricer for validation and performance.
    
    Engines:
    - AnalyticHestonEngine: Semi-analytic via characteristic function
    - FdHestonVanillaEngine: Finite difference
    - MCEuropeanHestonEngine: Monte Carlo
    """
    
    def __init__(
        self,
        engine: str = "analytic",  # analytic | fd | mc
        **engine_kwargs,
    ):
        self.engine_type = engine
        self.engine_kwargs = engine_kwargs
    
    def price(
        self,
        instrument: EquityVanillaOption,
        market: Market,
        heston_params: HestonParams,
    ) -> PricingResult:
        # Build QuantLib Heston process
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(market.spot))
        
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(0, ql.NullCalendar(), market.rate, ql.Actual365Fixed())
        )
        flat_div = ql.YieldTermStructureHandle(
            ql.FlatForward(0, ql.NullCalendar(), market.dividend, ql.Actual365Fixed())
        )
        
        heston_process = ql.HestonProcess(
            flat_ts, flat_div, spot_handle,
            heston_params.v0,
            heston_params.kappa,
            heston_params.theta,
            heston_params.sigma,
            heston_params.rho,
        )
        
        # Build engine
        if self.engine_type == "analytic":
            heston_model = ql.HestonModel(heston_process)
            engine = ql.AnalyticHestonEngine(heston_model)
        elif self.engine_type == "fd":
            engine = ql.FdHestonVanillaEngine(heston_model, **self.engine_kwargs)
        # ... etc
        
        # Build option and price
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if instrument.is_call else ql.Option.Put,
            instrument.strike,
        )
        exercise = ql.EuropeanExercise(instrument.expiry_ql)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        
        return PricingResult(
            pv=option.NPV(),
            delta=option.delta(),
            gamma=option.gamma(),
            vega=option.vega(),
            theta=option.theta(),
        )
```

**Files to Create:**
- `src/pricers/equity/european_heston_ql.py`
- `tests/unit/pricers/equity/test_european_heston_ql.py`

**Effort:** 4-6 hours

---

#### 1.2 Hull-White Bond Option Pricing

**Why:** QuantLib has exact semi-analytic pricing for bond options under Hull-White.

**Implementation Plan:**

```python
# src/pricers/ir/bond_option_hw_ql.py

class HullWhiteBondOptionQLPricer:
    """
    QuantLib Hull-White bond option pricer.
    
    Uses Jamshidian decomposition for callable bond pricing.
    """
    
    def price(
        self,
        instrument: BondOption,
        market: Market,
        hw_params: HullWhiteParams,
    ) -> PricingResult:
        # Build Hull-White model
        hw_model = ql.HullWhite(
            market.yield_curve_handle,
            hw_params.a,
            hw_params.sigma,
        )
        
        # Build engine
        engine = ql.TreeCallableFixedRateBondEngine(hw_model, n_steps)
        
        # ...
```

**Files to Create:**
- `src/pricers/ir/bond_option_hw_ql.py`
- `src/pricers/ir/swaption_hw_ql.py`

**Effort:** 4-6 hours

---

### Priority 2: Medium Value

#### 2.1 American Option Pricing (FD)

**Why:** QuantLib has optimized FD grids for American options with various boundary conditions.

```python
# src/pricers/equity/american_bsm_ql.py

class AmericanBSMQLPricer:
    """
    QuantLib FD pricer for American options.
    
    Uses FdBlackScholesVanillaEngine with:
    - Various grid configurations
    - Douglas scheme for stability
    """
    
    def price(self, instrument, market):
        process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle
        )
        
        engine = ql.FdBlackScholesVanillaEngine(
            process,
            tGrid=100,
            xGrid=100,
            dampingSteps=0,
        )
        # ...
```

**Effort:** 2-4 hours

---

#### 2.2 Swaption Pricing (Black/Bachelier)

**Why:** QuantLib has production-grade swaption pricing with:
- Black model
- Bachelier model
- Normal model

```python
# src/pricers/ir/swaption_ql.py

class SwaptionQLPricer:
    """
    QuantLib swaption pricer.
    
    Supports:
    - BlackSwaptionEngine: Lognormal vol
    - BachelierSwaptionEngine: Normal vol
    """
```

**Effort:** 4-6 hours

---

#### 2.3 Cap/Floor Pricing

**Why:** QuantLib caplet stripping and pricing is industry-standard.

```python
# src/pricers/ir/capfloor_ql.py

class CapFloorQLPricer:
    """
    QuantLib cap/floor pricer.
    
    Engines:
    - BlackCapFloorEngine
    - BachelierCapFloorEngine
    - AnalyticCapFloorEngine (Hull-White)
    """
```

**Effort:** 2-4 hours

---

### Priority 3: Lower Value (Nice to Have)

#### 3.1 Barrier Option Pricing

**Why:** QuantLib has analytic barrier formulas and FD engines.

#### 3.2 Asian Option Pricing

**Why:** QuantLib has Turnbull-Wakeman and moment matching.

#### 3.3 CMS Pricing

**Why:** QuantLib has convexity-adjusted CMS pricing.

---

## Integration Architecture

### Recommended Pattern

```python
# Pattern: QuantLib as optional backend

class MyPricer:
    """
    Pricer with native and QuantLib backends.
    """
    
    def __init__(self, backend: str = "native"):
        """
        Args:
            backend: "native" or "quantlib"
        """
        self.backend = backend
    
    def price(self, instrument, market) -> PricingResult:
        if self.backend == "quantlib":
            return self._price_quantlib(instrument, market)
        return self._price_native(instrument, market)
    
    def _price_native(self, instrument, market):
        # Native implementation
        ...
    
    def _price_quantlib(self, instrument, market):
        # QuantLib implementation
        try:
            import QuantLib as ql
        except ImportError:
            raise ImportError("QuantLib not installed. Use backend='native'.")
        ...
```

### Testing Pattern

```python
# tests/parity/test_heston_parity.py

import pytest

quantlib = pytest.importorskip("QuantLib")

class TestHestonParity:
    """Test native vs QuantLib Heston pricing."""
    
    def test_european_call_parity(self):
        # Setup
        instrument = create_test_option()
        market = create_test_market()
        params = create_test_heston_params()
        
        # Price with both
        native_result = NativeHestonPricer().price(instrument, market, params)
        ql_result = HestonQLPricer().price(instrument, market, params)
        
        # Compare
        assert native_result.pv == pytest.approx(ql_result.pv, rel=1e-4)
        assert native_result.delta == pytest.approx(ql_result.delta, rel=1e-3)
```

---

## Installation

```bash
# Install QuantLib Python bindings
pip install QuantLib

# Verify installation
python -c "import QuantLib; print(QuantLib.__version__)"
```

---

## Summary

| Integration | Priority | Effort | Value |
|-------------|----------|--------|-------|
| Heston Pricing | High | 4-6 hrs | Validation, performance |
| Hull-White Bond Options | High | 4-6 hrs | Validation, analytics |
| American FD | Medium | 2-4 hrs | Grid comparison |
| Swaption Pricing | Medium | 4-6 hrs | IR products |
| Cap/Floor Pricing | Medium | 2-4 hrs | IR products |
| Barrier Options | Low | 2-4 hrs | Validation |
| Asian Options | Low | 2-4 hrs | Validation |

**Total Estimated Effort:** 20-30 hours for full integration

---

## Next Steps

1. Start with Heston QuantLib pricer (highest value)
2. Add Hull-White bond option pricer
3. Create parity tests for all integrations
4. Document performance comparison results
