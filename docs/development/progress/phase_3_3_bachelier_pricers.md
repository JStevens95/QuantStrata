# Phase 3.3: Bachelier Pricers - Implementation Progress

**Started:** January 27, 2026  
**Completed:** January 27, 2026  
**Status:** COMPLETE

---

## Overview

Phase 3.3 implements Bachelier (normal) model pricers for swaptions and spread options. The Bachelier model is essential for:
- Negative rate environments (common in EUR, JPY, CHF)
- Spread options (difference between two underlyings can be negative)
- Industry-standard swaption pricing

---

## Implementation Summary

### Swaptions ✅

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/ir/options/swaption.py` | ✅ Complete |
| Pricer | `src/pricers/ir/bachelier.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/ir/test_swaption_bachelier_pricer.py` | ✅ Complete |
| Documentation | `docs/guides/instruments/swaptions.md` | ✅ Complete |

**Classes Implemented:**
- `Swaption` - Market data lookup version
- `SwaptionSimple` - Direct parameter version
- `SwaptionBachelierPricer` - Market data pricer
- `SwaptionBachelierPricerSimple` - Simple pricer

**Features:**
- Payer/receiver swaption types
- Cash/physical settlement styles
- Tenor description (e.g., "1Y5Y")
- ITM/OTM detection

**Greeks:**
- delta, gamma, vega, theta, rho
- vega_bp (vega per 1 basis point)

---

### FX Spread Options ✅

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/fx/options/spread.py` | ✅ Complete |
| Pricer | `src/pricers/fx/spread_bachelier.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/test_spread_bachelier_pricer.py` | ✅ Complete |

**Classes Implemented:**
- `EuropeanFxSpreadOption` - Market data lookup
- `EuropeanFxSpreadOptionSimple` - Direct parameters
- `FxSpreadOptionBachelierPricer` - Market data pricer
- `FxSpreadOptionBachelierPricerSimple` - Simple pricer

---

### Equity Spread Options ✅

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/equity/options/spread.py` | ✅ Complete |
| Pricer | `src/pricers/equity/spread_bachelier.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/test_spread_bachelier_pricer.py` | ✅ Complete |

**Classes Implemented:**
- `EuropeanEquitySpreadOption` - Market data lookup
- `EuropeanEquitySpreadOptionSimple` - Direct parameters
- `EquitySpreadOptionBachelierPricer` - Market data pricer
- `EquitySpreadOptionBachelierPricerSimple` - Simple pricer

---

## Test Summary

**Total Tests:** 45  
**Passing:** 45

| Test Category | Count |
|---------------|-------|
| Swaption Validation | 6 |
| Swaption Pricing | 5 |
| Swaption Put-Call Parity | 1 |
| Swaption Greeks | 6 |
| Swaption FD Validation | 2 |
| Swaption Market Data | 1 |
| Swaption Edge Cases | 3 |
| FX Spread Tests | 11 |
| Equity Spread Tests | 7 |
| General Edge Cases | 3 |

---

## Tutorial Notebook

Created comprehensive IR instruments pricing tutorial:
- `docs/tutorials/pricing/ir_instruments_pricing.ipynb`

**Contents:**
1. FRA pricing and PV analysis
2. IRS pricing and DV01
3. Caps and Floors with Black76
4. Swaptions with Bachelier
5. Model comparison (Black76 vs Bachelier)

---

## Mathematical Framework

### Bachelier Model

The Bachelier model assumes normal dynamics:
$$dF = \sigma \, dW$$

**Swaption Pricing:**
$$\text{Payer} = A \times N \times [(F - K) \cdot N(d) + \sigma\sqrt{T} \cdot n(d)]$$

**Spread Option Pricing:**
$$\text{Call} = N \times DF \times [(F - K) \cdot N(d) + \sigma\sqrt{T} \cdot n(d)]$$

Where:
- $d = (F - K) / (\sigma\sqrt{T})$
- $\sigma$ = normal (absolute) volatility

---

## Files Created/Modified

**New Files:**
- `src/instruments/ir/options/swaption.py`
- `src/instruments/fx/options/spread.py`
- `src/instruments/equity/options/spread.py`
- `src/pricers/ir/bachelier.py`
- `src/pricers/fx/spread_bachelier.py`
- `src/pricers/equity/spread_bachelier.py`
- `tests/unit/pricers/ir/test_swaption_bachelier_pricer.py`
- `tests/unit/pricers/test_spread_bachelier_pricer.py`
- `docs/guides/instruments/swaptions.md`
- `docs/tutorials/pricing/ir_instruments_pricing.ipynb`

**Modified Files:**
- `src/instruments/ir/options/__init__.py` - Added swaption exports
- `src/pricers/ir/__init__.py` - Added Bachelier pricer exports

---

## Change Log

| Date | Change |
|------|--------|
| 2026-01-27 | Phase 3.3 started |
| 2026-01-27 | Swaption instrument and pricer implemented |
| 2026-01-27 | FX and Equity spread options implemented |
| 2026-01-27 | 45 unit tests passing |
| 2026-01-27 | Tutorial notebook created |
| 2026-01-27 | Swaptions guide created |
| 2026-01-27 | **Phase 3.3 COMPLETE** |
