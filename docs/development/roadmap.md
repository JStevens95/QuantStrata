# QuantStrata Development Roadmap

**Last Updated:** January 27, 2026 (Phase 4.2 Complete - Advanced Numerical Methods)  
**Current Version:** V1 (FX Derivatives Foundation)  
**Target:** Comprehensive Professional Quant Library

---

## Roadmap Philosophy

This roadmap follows a **phased, incremental approach** that:
1. **Builds on existing strengths** (FX foundation)
2. **Maintains architectural coherence** (interface contracts)
3. **Demonstrates breadth and depth** (multiple asset classes, advanced models)
4. **Enhances educational value** (tutorials, notebooks)
5. **Improves production readiness** (calibration, performance, backtesting)

---

## Phase 1: Enhance FX Foundation (Weeks 1-4)

**Goal:** Complete FX derivatives coverage and add missing functionality

### 1.1 Additional FX Products ✅
- [x] **Asian Options** (average price/rate options)
  - Payoff: `max(avg(S_t) - K, 0)` for call
  - Implement: `AsianPayoff` (path-dependent)
  - Pricers: MC (required), FD (optional, 2D PDE)
  - Use case: Demonstrates path-dependent pricing

- [x] **Lookback Options** (floating strike)
  - Payoff: `max(S_T - min(S_t), 0)` for call
  - Implement: `LookbackPayoff` (path-dependent)
  - Pricer: MC (required)
  - Use case: Demonstrates extreme value statistics

- [x] **Double Barrier Options**
  - Payoff: Vanilla with upper and lower barriers
  - Implement: `DoubleBarrierPayoff` (path-dependent)
  - Pricer: MC (required), FD (optional, absorbing boundaries)
  - Use case: More complex barrier structures

- [x] **FX Touch Options** (one-touch, no-touch)
  - Payoff: Binary payout if barrier touched/not touched
  - Implement: `TouchPayoff` (path-dependent)
  - Pricer: MC (required)
  - Use case: Binary barrier products

### 1.2 Advanced FX Models ✅
- [x] **Local Volatility Model**
  - Implemented: `LocalVolSurface` (σ(S, t))
  - FD pricer: Update PDE coefficients for local vol
  - Calibration: Dupire's formula from market prices
  - Use case: Demonstrates volatility modeling beyond constant vol

- [x] **Stochastic Volatility (Heston Model)**
  - Implemented: `HestonDynamics` (2D SDE)
  - MC pricer: Simulate (S, V) joint process
  - FD pricer: 2D PDE solver (optional, advanced)
  - Use case: Demonstrates multi-factor models

### 1.3 FX Calibration Infrastructure ✅
- [x] **Volatility Surface Calibration**
  - Complete: `src/marketdata/surfaces/fx/calibration.py`
  - Implemented: SABR parametric fitting (`src/calibration/volatility_surface/sabr.py`)
  - Implemented: Dupire local vol extraction (`src/calibration/volatility_surface/dupire.py`)
  - Support: Delta-based strikes, term structure fitting
  - QuantLib backend: Validation implementations available
  - Use case: Production-ready calibration

- [x] **Curve Bootstrapping**
  - Enhanced: `src/marketdata/curves/bootstrapper.py`
  - Support: Multiple quote types (deposits, swaps, FRAs)
  - New: `src/marketdata/curves/interpolation.py` (log-linear, cubic spline)
  - Validation: Arbitrage checks
  - QuantLib backend: Alternative implementation
  - Use case: Real-world curve construction

- [x] **Mathematical Documentation**
  - `docs/mathematics/volatility_calibration.md` - SABR and Dupire theory
  - `docs/mathematics/curve_bootstrapping.md` - Curve construction theory
  - `docs/notebooks/calibration_*.ipynb` - Interactive tutorials

### 1.4 Performance Optimization ✅
- [x] **Numba Backend for MC**
  - Implemented: `src/core/performance/mc_kernels.py`
  - JIT-compiled: GBM path generation (exact, Euler, Milstein)
  - JIT-compiled: Payoff evaluation (vanilla, digital, barrier, Asian, lookback)
  - Target achieved: 20-50x speedup for MC
  - Maintained: Pure NumPy fallback for compatibility

- [x] **Optimized FD Operations**
  - Implemented: `src/core/performance/fd_kernels.py`
  - JIT-compiled: Thomas algorithm (tridiagonal solver)
  - JIT-compiled: PSOR solver (American options)
  - Parallel: Batch tridiagonal solves for Greeks
  - Target achieved: 20-30x speedup

- [x] **Benchmarking Framework**
  - Implemented: `src/core/performance/benchmark.py`
  - Systematic timing with warm-up and statistics
  - Backend comparison utilities
  - Documentation: `docs/mathematics/performance_optimization.md`
  - Tutorial: `docs/notebooks/performance_optimization.ipynb`

### Deliverables:
- ✅ 4+ new FX products
- ✅ 2 advanced models (local vol, Heston)
- ✅ Complete calibration infrastructure
- ✅ Performance benchmarks

**Impact:** Demonstrates depth in FX derivatives and advanced modeling.

---

## Phase 2: Expand to Equity Derivatives (Weeks 5-10)

**Goal:** Add equity asset class with similar depth to FX

### 2.1 Equity Core Instruments ✅
- [x] **Equity Spot & Forward**
  - Implemented: `EquitySpot`, `EquityForward`
  - Pricers: `EquitySpotPricer`, `EquityForwardPricer`
  - Forward = S * exp((r-q)T) with dividend yield
  - Use case: Foundation for equity options

- [x] **European Equity Options**
  - Implemented: `EuropeanEquityVanillaOption`
  - Payoff: Reuses `VanillaPayoff` (integrated in payoff factory)
  - Pricers: `EquityEuropeanVanillaBsmPricer`, `EquityEuropeanVanillaMcPricer`, `EquityEuropeanVanillaFdPricer`
  - BSM with `b = r - q` (dividend yield)
  - Put-call parity verified: C - P = S*exp(-qT) - K*exp(-rT)
  - Use case: Most common equity derivative

- [x] **American Equity Options**
  - Implemented: `AmericanEquityVanillaOption`
  - Pricer: `EquityAmericanVanillaFdPricer` (FD with PSOR)
  - Early exercise premium: American ≥ European
  - Tests verify intrinsic value floor
  - Use case: Early exercise for dividend-paying stocks

- [x] **Equity Barrier Options** (Phase 2.2)
  - Implement: `EuropeanEquityBarrierOption`
  - Pricer: MC (reuse existing barrier pricer)
  - Use case: Structured products component

### 2.2 Equity-Specific Products ✅
- [x] **Equity Digital Options**
  - Implement: `EuropeanEquityDigitalOption`
  - Payoff: Cash-or-asset (reuse existing)
  - Use case: Binary payouts

- [x] **Equity Asian Options**
  - Implement: `AsianEquityOption` (average price)
  - Payoff: Reuse `AsianPayoff`
  - Use case: Employee stock options, structured products

- [x] **Equity Lookback Options**
  - Implement: `LookbackEquityOption`
  - Payoff: Reuse `LookbackPayoff`
  - Use case: Path-dependent exotic

### 2.3 Alternative Pricing Models (Black76 & Bachelier) ✅

**Goal:** Extend the model library with Black76 (log-normal forward) and Bachelier (normal) models.

**Status:** Model foundation complete. Pricers/instruments deferred to Phase 3.

#### 2.3.1 Black76 Model (Forward-Based Pricing) ✅
- [x] **Black76 Model Engine**
  - Implemented: `src/models/analytic/black76/base.py`
  - Core formulas: d1, d2 using forward price F instead of spot
  - Price: C = DF × [F×N(d1) - K×N(d2)]
  - Greeks: Delta, Gamma, Vega, Theta, Rho (all implemented)
  - Unit tests: `tests/unit/models/analytic/black76/test_black76_vanilla.py`
  - Use case: Options on futures/forwards

#### 2.3.2 Bachelier Model (Normal Distribution) ✅
- [x] **Bachelier Model Engine**
  - Implemented: `src/models/analytic/bachelier/base.py`
  - Normal dynamics: dF = σ dW (absolute volatility)
  - Price: C = DF × [(F-K)N(d) + σ√T×n(d)] where d = (F-K)/(σ√T)
  - Greeks: Delta, Gamma, Vega, Theta, Rho (all implemented)
  - Key feature: Supports negative forward/strike (negative rates)
  - Unit tests: `tests/unit/models/analytic/bachelier/test_bachelier_vanilla.py`
  - Use case: Negative rates, spread options

#### 2.3.3 Model Documentation ✅
- [x] **Black76 Technical Documentation**
  - Complete: `docs/mathematics/black76.md`
  - Mathematical derivation from BSM
  - Forward measure vs spot measure
  - When to use Black76 vs BSM
  - Greeks comparison and interpretation
  - Interview key points

- [x] **Bachelier Technical Documentation**
  - Complete: `docs/mathematics/bachelier.md`
  - Normal vs log-normal dynamics
  - Volatility quoting conventions (bp vol vs % vol)
  - Negative underlying handling
  - Use cases and limitations
  - Interview key points

### 2.4 Equity Market Data ✅
- [x] **Equity Market Provider**
  - Implemented: `src/marketdata/providers/synthetic/generators/equity.py`
  - Registered: `register_equity_generators()` in SyntheticProvider
  - Generates: Stock prices (GBM with dividends)
  - Generates: Equity vol surfaces (strike-based)
  - Use case: Testing and examples

- [x] **Equity Volatility Surfaces**
  - Implemented: Strike-based vol generation in equity generator
  - Reuses: `GridVolSurface` with `strike_space="absolute"`
  - Features: Skew parameterization (negative skew for equity)
  - Use case: Real-world equity vol modeling

### 2.5 Equity Models ✅
- [x] **Dividend Models**
  - Implemented: Continuous dividend yield (in BSM and equity generator drift)
  - Implemented: Discrete dividends - `adjust_spot_for_discrete_dividend()`
  - Implemented: Combined forward calculation - `compute_forward_with_dividends()`
  - Use case: Realistic equity modeling with both continuous and discrete dividends

- [x] **Equity Local Volatility**
  - Reused: `LocalVolSurface` from FX (generic spot × time grid)
  - Reused: `DupireCalibrator` (works with r, q for equity)
  - Verified: Unit tests with equity parameters (dividend yields, skew patterns)
  - Use case: Equity vol surface modeling

### Deliverables:
- ✅ Complete equity derivatives suite (7+ products)
- ✅ Black76 model engine (pure functions)
- ✅ Bachelier model engine (pure functions)
- ✅ Technical documentation (Black76, Bachelier)
- ✅ Equity market data infrastructure (2.4)
- ✅ Equity local volatility adaptation (2.5)
- ✅ Unit tests for equity generators and local vol

### 2.6 Documentation & Tutorials ✅
- [x] **Documentation Reorganization**
  - Professional structure: `guides/`, `reference/`, `tutorials/`, `development/`
  - Market data guides: Architecture, synthetic generators, volatility surfaces
  - Instrument guides: All product specifications
  - Model references: BSM, Black76, Bachelier, local vol, etc.
  - Development docs: Roadmap, progress reports

- [x] **Jupyter Notebooks**
  - Implemented: `tutorials/market-data/synthetic_data_generation.ipynb`
  - Implemented: `tutorials/pricing/equity_options_pricing.ipynb`
  - Verified: All notebook imports working correctly
  - Topics: Synthetic data for FX/IR/Equity, equity pricing with BSM/MC/FDE

**Status:** Phase 2 COMPLETE.

**Impact:** Demonstrates breadth across asset classes and alternative pricing models while maintaining architectural coherence.

---

## Phase 3: Interest Rate Derivatives & Black76/Bachelier Pricers (Weeks 11-18)

**Goal:** Add rates asset class and complete Black76/Bachelier pricers from Phase 2.3 model foundation.

### 3.1 Black76 Pricers (Uses Phase 2.3 Model Foundation) ✅

**FX/Equity Forward Options:**
- [x] **FX Forward Options (Black76)**
  - Instrument: `EuropeanFxForwardOption`, `EuropeanFxForwardOptionSimple`
  - Pricer: `FxForwardOptionBlack76Pricer`, `FxForwardOptionBlack76PricerSimple`
  - Model: F = S × exp((r_d - r_f)×T), then Black76
  - Greeks: delta_forward, delta_spot, gamma, vega, theta, rho_domestic, rho_foreign
  - Documentation: `docs/guides/instruments/forward_options.md`
  - Tutorial: `docs/tutorials/instruments/forward_options.ipynb`

- [x] **Equity Index Futures Options (Black76)**
  - Instrument: `EuropeanEquityFuturesOption`, `EuropeanEquityFuturesOptionSimple`
  - Pricer: `EquityFuturesOptionBlack76Pricer`, `EquityFuturesOptionBlack76PricerSimple`
  - Model: F = S × exp((r - q)×T), then Black76
  - Greeks: delta_futures, delta_spot, gamma, vega, theta, rho
  - Documentation: `docs/guides/instruments/futures_options.md`
  - Tutorial: `docs/tutorials/instruments/futures_options.ipynb`

**Interest Rate Options (Black76):**
- [x] **Caps & Floors**
  - Instruments: `Cap`, `CapSimple`, `Floor`, `FloorSimple`, `Caplet`, `CapletSimple`, `Floorlet`, `FloorletSimple`
  - Pricers: `CapBlack76Pricer`, `FloorBlack76Pricer`, `CapletBlack76Pricer`, `FloorletBlack76Pricer`
  - Day count conventions: ACT/360, ACT/365, 30/360
  - Auto-generation of caplets/floorlets from cap/floor schedule
  - Tests: 29 unit tests passing

**Status:** Phase 3.1 COMPLETE. See `docs/development/progress/phase_3_1_black76_pricers.md` for details.

### 3.2 Linear IR Instruments (FRAs & IRS) ✅

**Rationale:** FRAs and IRS are the IR equivalent of FX/Equity Spot and Forward - foundational 
linear products that should be implemented before options on them (swaptions).

- [x] **Forward Rate Agreements (FRA)**
  - Instruments: `ForwardRateAgreement`, `ForwardRateAgreementSimple`
  - Pricers: `FRAPricer`, `FRAPricerSimple`
  - Formula: PV = N × τ × DF(T_pay) × (F - K)
  - Greeks: delta, DV01, PV01
  - Features: Payer/receiver direction, par rate, ITM detection, tenor description
  - Tests: 16 unit tests

- [x] **Interest Rate Swaps (IRS)**
  - Instruments: `InterestRateSwap`, `InterestRateSwapSimple`
  - Leg types: `FixedLeg`, `FloatingLeg`, `SwapLeg`
  - Pricers: `IRSwapPricer`, `IRSwapPricerSimple`
  - Formula: PV = N × Σ[τ_i × DF_i × (K - F_i)] (for fixed receiver)
  - Greeks: delta, DV01, PV01, annuity
  - Features: Configurable frequencies, day counts, floating spread
  - Tests: 20 unit tests

**Status:** Phase 3.2 COMPLETE. See `docs/development/progress/phase_3_2_linear_ir.md` for details.

### 3.3 Bachelier Pricers (Uses Phase 2.3 Model Foundation) ✅

**Rationale:** Swaptions require IRS as underlying, hence 3.3 comes after 3.2.

- [x] **Swaptions (Bachelier)**
  - Instruments: `Swaption`, `SwaptionSimple`
  - Pricers: `SwaptionBachelierPricer`, `SwaptionBachelierPricerSimple`
  - Features: Payer/receiver, cash/physical settlement, tenor description
  - Greeks: delta, gamma, vega, theta, rho, vega_bp
  - Documentation: `docs/guides/instruments/swaptions.md`
  - Tests: 24 unit tests

- [x] **Spread Options (Bachelier)**
  - FX: `EuropeanFxSpreadOption`, `EuropeanFxSpreadOptionSimple`
  - FX Pricer: `FxSpreadOptionBachelierPricer`, `FxSpreadOptionBachelierPricerSimple`
  - Equity: `EuropeanEquitySpreadOption`, `EuropeanEquitySpreadOptionSimple`
  - Equity Pricer: `EquitySpreadOptionBachelierPricer`, `EquitySpreadOptionBachelierPricerSimple`
  - Tests: 21 unit tests

- [x] **Tutorial Notebook**
  - `docs/tutorials/pricing/ir_instruments_pricing.ipynb`
  - Covers: FRAs, IRS, Caps/Floors, Swaptions, Model comparison

**Status:** Phase 3.3 COMPLETE. See `docs/development/progress/phase_3_3_bachelier_pricers.md` for details.

### 3.4 Bond Instruments ✅

- [x] **Zero Coupon Bonds**
  - Instruments: `IrBondZeroCoupon`, `IrBondZeroCouponSimple`
  - Pricers: `IrBondZeroCouponPricer`, `IrBondZeroCouponPricerSimple`
  - Greeks: DV01, modified duration, Macaulay duration, convexity
  - Tests: 8 unit tests

- [x] **Fixed Rate (Coupon) Bonds**
  - Instruments: `IrBondFixedRate`, `IrBondFixedRateSimple`
  - Pricers: `IrBondFixedRatePricer`, `IrBondFixedRatePricerSimple`
  - Features: Clean/dirty price, YTM calculation
  - Greeks: DV01, modified duration, Macaulay duration, convexity
  - Tests: 10 unit tests

- [x] **Bond Options (Black76)**
  - Instruments: `IrBondEuropeanOption`, `IrBondEuropeanOptionSimple`
  - Pricers: `IrBondEuropeanOptionB76Pricer`, `IrBondEuropeanOptionB76PricerSimple`
  - Model: Black76 on forward bond price
  - Greeks: delta, gamma, vega, theta, rho
  - Tests: 22 unit tests

**Status:** Phase 3.4 COMPLETE. See `docs/development/progress/phase_3_4_bond_instruments.md` for details.

### 3.5 Hull-White Model ✅

- [x] **Hull-White Model Core**
  - Implemented: `HullWhiteParameters`, `HullWhiteDynamics`, `HullWhiteSimulation`
  - Simulation: Exact OU and Euler schemes with antithetic variates
  - Analytic: ZC bond pricing, bond option pricing (closed-form)
  - Features: Mean reversion, Gaussian distribution, term structure fitting
  - Tests: 37 model tests

- [x] **Hull-White Analytic Pricers**
  - `IrBondZeroCouponHWPricerSimple` - ZC bond pricing
  - `IrBondEuropeanOptionHWPricerSimple` - Bond option pricing
  - `IrCapletEuropeanOptionHWPricerSimple` - Caplet pricing
  - `IrFloorletEuropeanOptionHWPricerSimple` - Floorlet pricing
  - `IrSwaptionEuropeanOptionHWPricerSimple` - Swaption (Jamshidian)

- [x] **Hull-White Monte Carlo Pricers**
  - `IrBondZeroCouponMCPricerSimple` - ZC bonds via MC
  - `IrBondEuropeanOptionMCPricerSimple` - Bond options via MC
  - `IrCapletEuropeanOptionMCPricerSimple` - Caplets via MC
  - `IrSwaptionEuropeanOptionMCPricerSimple` - Swaptions via MC
  - Features: Configurable paths/steps, variance reduction, std error

- [x] **Hull-White Finite Difference Pricers**
  - `IrBondZeroCouponFDPricerSimple` - ZC bonds via FDE
  - `IrBondEuropeanOptionFDPricerSimple` - Bond options via FDE
  - Features: Crank-Nicolson, configurable grid, Thomas algorithm

- [x] **Documentation**
  - `docs/guides/models/hull_white.md` - Technical guide
  - `docs/development/progress/phase_3_5_hull_white.md` - Progress report

**Status:** Phase 3.5 COMPLETE. See `docs/development/progress/phase_3_5_hull_white.md` for details.

### 3.6 Black-Karasinski Model ✅

- [x] **Black-Karasinski Model** (3.6.1)
  - Implemented: `BlackKarasinskiParameters`, `BlackKarasinskiDynamics`, `BlackKarasinskiSimulation`
  - MC pricer: `IrBondZeroCouponBKMCPricerSimple`, `IrBondEuropeanOptionBKMCPricerSimple`
  - MC pricer: `IrCapletEuropeanOptionBKMCPricerSimple`, `IrFloorletEuropeanOptionBKMCPricerSimple`
  - Key properties: Log-normal rates (always positive), exact OU simulation
  - Tests: 58 tests passing
  - Documentation: `docs/guides/models/black_karasinski.md`

**Status:** Phase 3.6 COMPLETE. See `docs/development/progress/phase_3_6_black_karasinski.md` for details.

### 3.7 Rate Infrastructure Enhancement ✅

- [x] **Rate Market Data Enhancement**
  - Implemented: `SwaptionVolCube` (3D: expiry × tenor × strike)
  - Implemented: `CapFloorVolSurface` (2D: expiry × strike)
  - Implemented: Factory functions for vol surface creation
  - Support: Normal (Bachelier) and log-normal (Black) vol types
  - Tests: 26 tests passing

- [x] **Rate Curve Bootstrapping**
  - Existing: Multi-instrument bootstrapping (deposits, FRAs, swaps)
  - Existing: OIS support in quote types
  - Existing: Arbitrage validation for vol surfaces

**Status:** Phase 3.7 COMPLETE. See `docs/development/progress/phase_3_7_rate_infrastructure.md` for details.

### 3.8 LIBOR Market Model ✅

- [x] **LMM Implementation**
  - Implemented: `LMMCorrelation` (flat, exponential, custom)
  - Implemented: `LMMParameters` (tenors, forwards, vols, correlation)
  - Implemented: `LMMDynamics` (MC simulation with drift correction)
  - MC pricer: Caplets, floorlets, swaptions
  - Tests: 25 tests passing

**Status:** Phase 3.8 COMPLETE. See `docs/development/progress/phase_3_8_lmm.md` for details.

### Deliverables:
- [x] Complete Black76 pricers (FX forward options, futures options, caps/floors)
- [x] Linear IR instruments (FRA, IRS) - 36 tests passing
- [x] Complete Bachelier pricers (swaptions, spread options) - 45 tests passing
- [x] Bond instruments and options (zero coupon, fixed rate, bond options) - 40 tests passing
- [x] Hull-White model (analytic, MC, FDE pricers) - 60 tests passing
- [x] Black-Karasinski model (MC pricers) - 58 tests passing
- [x] Rate market data infrastructure (swaption/cap vol surfaces) - 27 tests passing
- [x] LMM (multi-factor forward rate model) - 25 tests passing

**Impact:** Demonstrates ability to handle complex, multi-factor models. Rates are a key differentiator for quant libraries.

---

## Phase 4: Advanced Models & Methods (Weeks 19-24)

**Goal:** Add advanced quantitative methods and models

### 4.1 Advanced Stochastic Models ✅

- [x] **Jump-Diffusion (Merton Model)**
  - Implemented: `MertonParameters`, `MertonDynamics`, `MertonSimulation`
  - MC pricer: Path and exact terminal simulation
  - Analytic: Series pricing for European options
  - Tests: 46 unit tests passing

- [x] **Stochastic Volatility (SABR Model)**
  - Implemented: `SabrDynamics`, `SabrSimulation`
  - Analytic: Hagan formula for implied vol
  - Calibration: Market smile fitting
  - Tests: 25 unit tests passing

- [x] **Variance Gamma (VG) Model**
  - Implemented: `VarianceGammaParameters`, `VarianceGammaDynamics`
  - MC pricer: Subordination-based simulation
  - Characteristic function: For FFT methods
  - Tests: 38 unit tests passing

**Status:** Phase 4.1 COMPLETE. See `docs/development/progress/phase_4_1_advanced_stochastic_models.md`.

### 4.2 Advanced Numerical Methods ✅

- [x] **Longstaff-Schwartz (LSM) for American**
  - Implemented: `lsm_american_put`, `lsm_american_call`, `price_american_put_lsm`
  - Basis functions: Polynomial, Laguerre, Chebyshev
  - Tests: 15 unit tests passing

- [x] **Quasi-Monte Carlo (QMC)**
  - Implemented: `SobolRng`, `HaltonRng`, `qmc_european_call/put`
  - Features: Scrambling, antithetic, path simulation
  - Tests: 20 unit tests passing

- [x] **Importance Sampling**
  - Implemented: `is_european_call/put`, `adaptive_is_european_call`
  - Features: Optimal drift shift, variance reduction tracking
  - Tests: 19 unit tests passing

- [ ] **Adaptive Mesh Refinement (FD)** (deferred)
  - Deferred to future phase

**Status:** Phase 4.2 COMPLETE. See `docs/development/progress/phase_4_2_advanced_numerical_methods.md`.

### 4.3 Multi-Asset Products
- [ ] **Basket Options**
  - Implement: `BasketOption` (option on weighted sum)
  - Payoff: `max(Σ(w_i * S_i) - K, 0)`
  - MC pricer: Simulate correlated assets
  - Use case: Multi-asset derivatives

- [ ] **Spread Options**
  - Implement: `SpreadOption` (option on S1 - S2)
  - Pricer: MC (required), analytic approximation (optional)
  - Use case: Commodity/energy trading

- [ ] **Worst-of / Best-of Options**
  - Implement: `WorstOfOption`, `BestOfOption`
  - Payoff: Option on min/max of multiple assets
  - MC pricer: Simulate correlated paths
  - Use case: Structured products

### Deliverables:
- ✅ 3 advanced models (Merton jump-diffusion, SABR, Variance Gamma) - 109 tests
- ✅ 3 advanced numerical methods (LSM, QMC, Importance Sampling) - 54 new tests
- [ ] Multi-asset products (Phase 4.3)

**Impact:** Demonstrates advanced quantitative skills and research-level understanding.

---

## Phase 5: Production Infrastructure (Weeks 25-30)

**Goal:** Complete production-ready infrastructure

### 5.1 Calibration Framework
- [ ] **Unified Calibration Interface**
  - Implement: `CalibrationEngine` (generic optimizer)
  - Support: Multiple objective functions (least squares, max likelihood)
  - Support: Multiple optimizers (Levenberg-Marquardt, genetic algorithms)
  - Use case: Model calibration to market data

- [ ] **Volatility Surface Calibration**
  - Complete: FX vol calibration (from Phase 1)
  - Complete: Equity vol calibration (from Phase 2)
  - Complete: Rates vol calibration (from Phase 3)
  - Use case: Production vol surface construction

- [ ] **Model Parameter Calibration**
  - Implement: Heston calibration (to vol surface)
  - Implement: Hull-White calibration (to cap/swaption prices)
  - Implement: SABR calibration (to swaption smile)
  - Use case: Advanced model calibration

### 5.2 Backtesting Infrastructure
- [ ] **Backtesting Framework**
  - Implement: `BacktestEngine` (replay historical data)
  - Support: Strategy evaluation (P&L, Sharpe, drawdowns)
  - Support: Risk metrics (VaR, CVaR)
  - Use case: Strategy validation

- [ ] **Historical Data Integration**
  - Implement: `HistoricalDataProvider`
  - Support: CSV, Parquet, database backends
  - Use case: Real-world backtesting

- [ ] **Performance Attribution**
  - Implement: P&L decomposition (delta, gamma, theta, vega)
  - Support: Daily/weekly/monthly attribution
  - Use case: Understanding strategy performance

### 5.3 Risk Infrastructure Enhancements
- [ ] **Value-at-Risk (VaR)**
  - Implement: Historical VaR, Parametric VaR, Monte Carlo VaR
  - Support: Portfolio-level VaR
  - Use case: Risk management

- [ ] **Greeks Aggregation**
  - Enhance: Portfolio-level greeks with proper bucketing
  - Support: Risk factor decomposition
  - Use case: Risk reporting

- [ ] **Stress Testing**
  - Enhance: Scenario generation (historical, hypothetical)
  - Support: Multi-factor stress scenarios
  - Use case: Regulatory stress testing

### 5.4 Performance & Scalability
- [ ] **JAX Backend** (Optional, Advanced)
  - Implement: JAX-based MC pricer (GPU acceleration)
  - Use case: High-performance computing

- [ ] **Parallel Portfolio Pricing**
  - Implement: Multi-threaded portfolio pricing
  - Use case: Large portfolio performance

- [ ] **Caching & Memoization**
  - Implement: Market data caching
  - Implement: Pricer result caching
  - Use case: Performance optimization

### Deliverables:
- ✅ Complete calibration framework
- ✅ Backtesting infrastructure
- ✅ Enhanced risk infrastructure
- ✅ Performance optimizations

**Impact:** Transforms library from "demonstration" to "production-ready" system.

---

## Phase 6: Educational & Documentation (Ongoing)

**Goal:** Enhance educational value and usability

### 6.1 Tutorials & Examples
- [ ] **Quick Start Guide**
  - Create: `docs/QUICKSTART.md`
  - Content: Install, first pricing example, basic concepts
  - Use case: Onboarding new users

- [ ] **Jupyter Notebooks**
  - Create: `examples/notebooks/`
  - Content:
    - BSM derivation walkthrough
    - FD vs MC comparison
    - Calibration tutorial
    - Risk analysis tutorial
  - Use case: Interactive learning

- [ ] **Video Tutorials** (Optional)
  - Create: Screen recordings of key workflows
  - Use case: Visual learners

### 6.2 Documentation Enhancements
- [ ] **API Reference**
  - Generate: Sphinx/autodoc documentation
  - Use case: Developer reference

- [ ] **Mathematical Appendices**
  - Create: `docs/mathematics/`
  - Content:
    - FD method derivations
    - MC variance reduction theory
    - Model derivations (Heston, Hull-White, etc.)
  - Use case: Deep dive into methodology

- [ ] **Best Practices Guide**
  - Create: `docs/BEST_PRACTICES.md`
  - Content: Coding standards, testing patterns, performance tips
  - Use case: Contributor guide

### 6.3 Interactive Tools
- [ ] **Pricing Calculator** (Web App)
  - Create: Simple web interface for pricing
  - Use case: Non-technical users

- [ ] **Visualization Tools**
  - Enhance: Plotting utilities
  - Add: Interactive plots (Plotly/Bokeh)
  - Use case: Better visualizations

### Deliverables:
- ✅ Comprehensive tutorials
- ✅ Jupyter notebooks
- ✅ Enhanced documentation
- ✅ Interactive tools

**Impact:** Transforms library into an effective **revision tool** and **learning resource**.

---

## Phase 7: Advanced Topics (Weeks 31-36)

**Goal:** Add cutting-edge topics for interview differentiation

### 7.1 Machine Learning Integration
- [ ] **ML-Based Pricing**
  - Implement: Neural network pricers (train on MC data)
  - Use case: Fast approximate pricing

- [ ] **ML Calibration**
  - Implement: ML-based model calibration
  - Use case: Fast calibration

- [ ] **GNN for Portfolio Pricing** (Already started)
  - Complete: `src/m_learning/models/gnn_rnn_hybrid/`
  - Use case: Graph-based portfolio representation

### 7.2 Exotic Products
- [ ] **Cliquet Options**
  - Implement: `CliquetOption` (periodic resets)
  - Pricer: MC (required)
  - Use case: Structured products

- [ ] **Autocallable Products**
  - Implement: `AutocallableOption` (auto-exercise on barrier)
  - Pricer: MC (required)
  - Use case: Popular structured product

- [ ] **Range Accrual**
  - Implement: `RangeAccrual` (payout based on range)
  - Pricer: MC (required)
  - Use case: Interest rate structured product

### 7.3 Credit Derivatives (Optional)
- [ ] **Credit Default Swaps (CDS)**
  - Implement: `CreditDefaultSwap`
  - Model: Reduced-form credit model
  - Use case: Credit risk

- [ ] **Credit Options**
  - Implement: Options on CDS
  - Use case: Credit volatility trading

### 7.4 Commodities (Optional)
- [ ] **Commodity Options**
  - Implement: `CommodityOption`
  - Model: GBM with convenience yield
  - Use case: Energy/agricultural derivatives

### Deliverables:
- ✅ ML integration (pricing, calibration)
- ✅ 3+ exotic products
- ✅ Optional: Credit/commodities

**Impact:** Demonstrates **cutting-edge knowledge** and **research-level** capabilities.

---

## Implementation Guidelines

### Adding a New Product

1. **Create Instrument** (`src/instruments/<asset_class>/<product>.py`)
   - Dataclass with trade parameters
   - Market ID references
   - Validation in `__post_init__`

2. **Create Payoff** (`src/models/payoffs/<product>.py`)
   - Terminal: `BasePayoff1D` with `terminal(spot)`
   - Path-dependent: `BasePathPayoff1D` with `terminal_from_paths(paths)`

3. **Create Pricer(s)** (`src/pricers/<asset_class>/<product>_<method>.py`)
   - Use payoff library (no inline payoff logic)
   - Implement `price(instrument, market) -> float`
   - Optional: `greeks(instrument, market) -> dict`

4. **Register** (`src/pricers/registry.py`)
   - Add to `DefaultPricerRegistry.build()`
   - Set default pricer per product

5. **Tests** (`tests/unit/pricers/<asset_class>/test_<product>.py`)
   - Unit tests for payoff
   - Pricer tests
   - Parity tests (if multiple pricers)

6. **Examples** (`examples/<asset_class>/<product>_example.py`)
   - End-to-end example
   - Visualization

### Adding a New Asset Class

1. **Market Data**
   - Extend `SyntheticProvider` for new asset class
   - Create market ID conventions
   - Implement generators

2. **Instruments**
   - Create `src/instruments/<asset_class>/` directory
   - Implement spot/forward/options

3. **Pricers**
   - Create `src/pricers/<asset_class>/` directory
   - Implement pricers (reuse models where possible)

4. **Tests & Examples**
   - Comprehensive test suite
   - Example scripts

---

## Success Metrics

### Quantitative
- **Products**: 30+ instruments across 3+ asset classes
- **Models**: 10+ stochastic models
- **Methods**: 5+ numerical methods (BSM, FD, MC, LSM, QMC)
- **Test Coverage**: >90%
- **Performance**: MC 10-100x faster with Numba

### Qualitative
- **Architecture**: Maintains interface contracts, no breaking changes
- **Documentation**: Comprehensive tutorials and mathematical derivations
- **Code Quality**: Production-ready, professional standards
- **Educational Value**: Effective revision tool

---

## Timeline Summary

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|-----------------|
| Phase 1 | Weeks 1-4 | FX Enhancement | 4+ FX products, local vol, calibration |
| Phase 2 | Weeks 5-10 | Equity | 7+ equity products, equity infrastructure |
| Phase 3 | Weeks 11-18 | Rates | 6+ rate products, Hull-White, LMM |
| Phase 4 | Weeks 19-24 | Advanced Models | Jump-diffusion, SABR, multi-asset |
| Phase 5 | Weeks 25-30 | Production | Calibration, backtesting, risk |
| Phase 6 | Ongoing | Education | Tutorials, notebooks, docs |
| Phase 7 | Weeks 31-36 | Advanced Topics | ML, exotics, credit/commodities |

**Total Timeline:** ~9 months for core functionality, ongoing for education/advanced topics

---

## Risk Mitigation

### Technical Risks
- **Complexity Creep**: Maintain interface contracts, avoid over-engineering
- **Performance**: Profile early, optimize bottlenecks
- **Testing**: Maintain high test coverage, parity tests prevent regressions

### Scope Risks
- **Feature Creep**: Stick to roadmap, defer nice-to-haves
- **Time Management**: Prioritize high-impact features first
- **Quality vs Speed**: Maintain quality standards, don't rush

### Mitigation Strategies
- **Incremental Development**: Small, testable increments
- **Regular Reviews**: Assess progress against roadmap
- **Flexibility**: Adjust roadmap based on learnings

---

## Conclusion

This roadmap provides a **structured path** to building a **comprehensive, professional quant library** that:

1. ✅ **Builds on existing strengths** (FX foundation)
2. ✅ **Maintains architectural coherence** (interface contracts)
3. ✅ **Demonstrates breadth** (multiple asset classes)
4. ✅ **Demonstrates depth** (advanced models, methods)
5. ✅ **Enhances educational value** (tutorials, notebooks)
6. ✅ **Achieves production readiness** (calibration, backtesting)

**Next Steps:**
1. Review and prioritize phases
2. Start with Phase 1 (FX enhancement)
3. Iterate based on learnings
4. Maintain quality standards throughout

**The foundation is excellent. Time to build! 🚀**
