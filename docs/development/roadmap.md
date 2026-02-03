# QuantStrata Development Roadmap

**Last Updated:** January 27, 2026 (Phase 5.1 Complete - Calibration Framework)  
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
6. **Prepares for application projects** (streaming/live data for algo trading, advanced analytics, GNN-LSTM pricer, Q-learning agents)

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

### 4.3 Multi-Asset Products ✅

- [x] **Basket Options**
  - Instrument: `MultiAssetBasketEuropeanOption` (option_type: call/put)
  - Pricer: `MultiAssetBasketEuropeanOptionMcPricer`
  - Features: Arbitrary weights, correlated simulation, simulation artifacts
  - Tests: 13 unit tests passing

- [x] **Spread Options**
  - Instrument: `MultiAssetSpreadEuropeanOption` (option_type: call/put)
  - Pricer: `MultiAssetSpreadEuropeanOptionMcPricer`, `MultiAssetSpreadEuropeanOptionKirkPricer`
  - Features: Kirk's approximation (analytic), Monte Carlo
  - Tests: 8 unit tests passing

- [x] **Exchange Options**
  - Instrument: `MultiAssetExchangeEuropeanOption` (spread with K=0)
  - Pricer: `MultiAssetExchangeEuropeanOptionMargrabePricer` (exact closed-form)
  - Tests: 2 unit tests passing

- [x] **Best-of / Worst-of Options**
  - Instruments: `MultiAssetBestOfEuropeanOption`, `MultiAssetWorstOfEuropeanOption`
  - Pricers: `MultiAssetBestOfEuropeanOptionMcPricer`, `MultiAssetWorstOfEuropeanOptionMcPricer`
  - Features: N-asset support, correlation effects, simulation artifacts
  - Tests: 10 unit tests passing

- [x] **Multi-Asset Simulation Infrastructure**
  - Implemented: `CorrelationMatrix`, `MultiAssetGBM` (in `src/models/numeric/monte_carlo/multi_asset.py`)
  - Features: Cholesky correlation, terminal simulation, antithetic variates
  - Tests: Integrated into pricer tests

**Status:** Phase 4.3 COMPLETE. See `docs/development/progress/phase_4_3_multi_asset_products.md`.

**Architecture Notes:**
- Instruments follow FX naming convention: `MultiAsset{Product}EuropeanOption`
- Pricers follow FX naming convention: `MultiAsset{Product}EuropeanOption{Method}Pricer`
- All instruments use `option_type: "call" | "put"` field (consistent with `FxVanillaEuropeanOption`)
- Files: `src/instruments/multi_asset/`, `src/pricers/multi_asset/`

### Deliverables:
- ✅ 3 advanced models (Merton jump-diffusion, SABR, Variance Gamma) - 109 tests
- ✅ 3 advanced numerical methods (LSM, QMC, Importance Sampling) - 54 new tests
- ✅ 5 multi-asset products (Basket, Spread, Exchange, Best-of, Worst-of) - 41 tests

**Impact:** Demonstrates advanced quantitative skills and research-level understanding.

---

## Phase 5: Production Infrastructure (Weeks 25-30)

**Goal:** Complete production-ready infrastructure

### 5.1 Calibration Framework ✅

- [x] **Unified Calibration Interface**
  - Implemented: `CalibrationEngine` (generic optimizer orchestration)
  - Implemented: `WeightedLeastSquares`, `PenalizedObjective`, `MaxLikelihood` objectives
  - Implemented: `LBFGSBConfig`, `DifferentialEvolutionConfig`, `LevenbergMarquardtConfig` optimizers
  - Tests: 16 unit tests for core engine

- [x] **Volatility Surface Calibration**
  - Existing: FX smile calibration (ATM/RR/BF → GridVolSurface)
  - Existing: SABR calibration to smile data
  - Existing: Dupire local vol extraction

- [x] **Model Parameter Calibration**
  - Implemented: Heston calibration to vol surface (FFT pricing + optimization)
  - Implemented: Hull-White calibration to swaptions and caps
  - Implemented: SABR calibration to swaption smile (normal/lognormal)
  - Tests: 27 unit tests for model calibration

**Status:** Phase 5.1 COMPLETE.

**Architecture:**
- Core: `src/calibration/core/` (engine, objectives, optimizers)
- Heston: `src/calibration/stochastic_volatility/heston.py`
- Hull-White: `src/calibration/short_rate/hull_white.py`
- SABR IR: Extended `src/calibration/volatility_surface/sabr.py`

**Documentation:**
- Reference: `docs/reference/calibration/calibration_framework.md`
- Guide: `docs/guides/calibration/calibration_framework.md`
- Tutorial: `docs/tutorials/calibration/calibration_framework.ipynb`
- Progress: `docs/development/progress/phase_5_1_calibration_framework.md`

### 5.2 Backtesting Infrastructure ✅

**Status:** Complete (January 2026)

- [x] **Backtesting Framework**
  - Implemented: `BacktestEngine` with portfolio tracking
  - Supported: Strategy evaluation (Sharpe, Sortino, Calmar, max drawdown)
  - Supported: Transaction costs and slippage modeling
  - Use case: Strategy validation

- [x] **Historical Data Integration**
  - Implemented: `HistoricalDataProvider` base class
  - Implemented: `DictDataProvider` (in-memory)
  - Implemented: `CsvDataProvider` (wide/long formats)
  - Use case: Real-world backtesting

- [x] **Performance Attribution**
  - Implemented: `attribute_pnl_to_greeks()` for decomposition
  - Implemented: `PnLAttribution` for time series tracking
  - Supported: Daily/weekly/monthly aggregation
  - Use case: Understanding strategy performance

**Components:**
- `src/backtesting/core/engine.py` - BacktestEngine, BacktestResult
- `src/backtesting/core/metrics.py` - PerformanceMetrics, Sharpe, Sortino, etc.
- `src/backtesting/data/providers.py` - Data provider implementations
- `src/backtesting/attribution/pnl.py` - P&L attribution

**Tests:** 65 unit tests in `tests/unit/backtesting/`

**Documentation:**
- Reference: `docs/reference/backtesting/backtesting_framework.md`
- Guide: `docs/guides/backtesting/backtesting_framework.md`
- Tutorial: `docs/tutorials/backtesting/backtesting_introduction.ipynb`
- Progress: `docs/development/progress/phase_5_2_backtesting.md`

### 5.3 Risk Infrastructure Enhancements ✅
- [x] **Value-at-Risk (VaR)**
  - Implemented: Historical VaR, Parametric VaR, Monte Carlo VaR
  - Support: Portfolio-level VaR
  - Use case: Risk management

- [x] **Greeks Aggregation**
  - Implemented: Portfolio-level greeks with bucketing (by greek, by risk factor)
  - Support: Risk factor decomposition (GreeksSummary, aggregate_sensitivities)
  - Use case: Risk reporting

- [x] **Stress Testing**
  - Implemented: Scenario generation (preset packs, historical-based shocks)
  - Support: Multi-factor stress scenarios (CompositeShock)
  - Use case: Regulatory stress testing

**Status:** Phase 5.3 COMPLETE.

**Components:**
- `src/risk/var/` — VarConfig, VarResult, historical_var, parametric_var, mc_var, compute_var, DiagonalFactorModel
- `src/risk/sensitivities/aggregation.py` — aggregate_sensitivities, GreeksSummary
- `src/risk/scenarios/generation.py` — preset_stress_pack, shocks_from_historical_series, composite_from_preset
- `src/marketdata/scenarios/shocks.py` — CompositeShock

**Documentation:**
- Reference: `docs/reference/risk/risk_infrastructure.md`
- Guide: `docs/guides/risk/risk_framework.md`
- Tutorial: `docs/tutorials/risk/risk_introduction.ipynb`

### 5.4 Performance & Scalability ✅
- [x] **JAX Backend** (Optional, Advanced)
  - Implemented: JAX backend in `src/core/performance/backend.py` (Backend.JAX, jax_available, get_jax_version)
  - Implemented: JAX kernels in `src/core/performance/jax_kernels.py` (GBM path/terminal, vanilla/digital payoff)
  - Implemented: JAX MC pricer for FX vanilla in `src/pricers/fx/european_bsm_jax_mc.py` (pricer_id="jax_mc")
  - Use case: High-performance computing (CPU/GPU when jaxlib with CUDA/ROCm)

- [x] **Parallel Portfolio Pricing**
  - Implemented: `src/portfolio/parallel.py` — ParallelPortfolioPricer (ThreadPoolExecutor)
  - Use case: Large portfolio performance

- [x] **Caching & Memoization**
  - Implemented: Market data cache in `src/marketdata/cache.py` — CachingMarketDataProvider
  - Implemented: Pricer result cache in `src/portfolio/caching.py` — CachingPortfolioPricer
  - Use case: Performance optimization

**Documentation:**
- Reference: `docs/reference/performance_optimisation.md` (JAX, parallel, caching sections)
- Guide: `docs/guides/performance/performance_and_scalability.md`
- Tutorial: `docs/tutorials/performance/performance_and_scalability.ipynb`

### 5.5 Streaming & Live Data (for Algo Trading) ✅
- [x] **Streaming Data Provider**
  - Implemented: `StreamingMarketDataProtocol` in `src/marketdata/providers/streaming/protocol.py` (async `stream()` yielding (timestamp, Market))
  - Implemented: `ReplayStreamProvider` in `src/marketdata/providers/streaming/replay.py` (replay from MarketDataset or list of (timestamp, Market))
  - Use case: Algorithmic trading, paper/live trading; modeled on Alpaca/IBKR patterns; first impl simulated

- [x] **Event-Driven Engine**
  - Implemented: `StreamingEngine` in `src/streaming/engine.py` consuming stream of (timestamp, Market)
  - Reuse: Same strategy signature `(market, portfolio, context) -> orders` as backtesting
  - Support: Paper vs live via injected brokerage adapter
  - Use case: Deploy strategies on streaming data

- [x] **Brokerage Adapter Interface**
  - Implemented: `BrokerageAdapter` protocol in `src/streaming/brokerage/protocol.py` (submit_order, cancel_order, get_positions)
  - Implemented: `PaperBrokerageAdapter` in `src/streaming/brokerage/paper.py` (in-memory execution simulation; apply_market for fills)
  - Use case: Connect algo bot to practice/live brokerage accounts; real Alpaca/IBKR adapters implement same protocol (future)

**Documentation:**
- Reference: `docs/reference/streaming_live_data.md`
- Guide: `docs/guides/streaming/streaming_and_live_data.md`
- Tutorial: `docs/tutorials/streaming/streaming_and_live_data.ipynb`

### 5.6 Advanced Analytics & Reporting
- [x] **Front-Office Risk Reports**
  - Implement: Greeks surfaces, PnL attribution reports, VaR/CVaR summaries
  - Support: Portfolio-level and instrument-level breakdowns
  - Use case: Hedge fund risk management reporting
  - Docs: [Advanced Analytics & Reporting (reference)](../reference/advanced_analytics_reporting.md), [guide](../guides/analytics/advanced_analytics_reporting.md), [tutorial](../tutorials/analytics/advanced_analytics_reporting.ipynb)

- [x] **Publication-Quality Visualisation**
  - Enhance: Plotting utilities for advanced analytics (vol surfaces, Greeks heatmaps, scenario fan charts)
  - Support: Consistent styling, export for reports
  - Use case: Option pricing analytic reports, risk dashboards


### Deliverables:
- ✅ Complete calibration framework
- ✅ Backtesting infrastructure
- ✅ Risk infrastructure (VaR, Greeks aggregation, stress testing)
- [x] Streaming & live data (5.5): streaming provider, event-driven engine, brokerage adapter interface
- [x] Advanced analytics & reporting (5.6): front-office risk reports, publication-quality visualisation
- [x] Performance optimizations (5.4)

**Impact:** Transforms library from "demonstration" to "production-ready" system and supports application projects (algo bot, option analytics, GNN-LSTM pricer, Q-learning orchestrator).

---

## Phase 6: Educational & Documentation (Ongoing)

**Goal:** Enhance educational value and usability

### 6.1 Tutorials & Examples
- [x] **Quick Start Guide**
  - Created: [docs/QUICKSTART.md](../QUICKSTART.md) — install, venv, first pricing example; links to full docs and tutorials.
  - Use case: Onboarding new users

- [x] **Jupyter Notebooks**
  - Tutorials live in `docs/tutorials/` (canonical location; no separate `examples/notebooks/`).
  - Content: calibration, pricing (FD/MC, SABR, LMM, etc.), instruments, risk, streaming, analytics. See [docs/tutorials/README.md](../tutorials/README.md).
  - Use case: Interactive learning

### 6.2 Documentation Enhancements
- [x] **API Reference**
  - Sphinx/autodoc in [docs/source/](../source/); build with `sphinx-build -b html docs/source docs/build/html`. See [requirements-docs.txt](../../requirements-docs.txt).
  - Use case: Developer reference

- [x] **Mathematical Appendices**
  - Content lives under [docs/reference/models/](../reference/models/) and is indexed in [docs/reference/README.md](../reference/README.md) (Mathematical appendices section): FD, MC, Heston, Hull-White, SABR, local vol, curve bootstrapping, volatility calibration.
  - Use case: Deep dive into methodology

- [x] **Best Practices Guide**
  - Created: [docs/BEST_PRACTICES.md](../BEST_PRACTICES.md) — coding standards, testing, performance, project conventions.
  - Use case: Contributor guide

### 6.3 Interactive Tools
- [x] **Pricing Calculator** (Dash UI)
  - Dash apps in `src/ui/`; FX vanilla pricing calculator. Run: `python -m src.ui.pricing_calculator`. See [docs/guides/interactive_tools.md](../guides/interactive_tools.md).
  - Use case: Non-technical users

- [x] **Visualization**
  - Publication-quality matplotlib plots in Phase 5.6; interactive Dash UIs (e.g. pricing calculator) in `src/ui/` with optional `requirements-ui.txt`.
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

### ML/RL Framework Design (General as Possible)

For both **machine learning** and **Q-learning / reinforcement learning**, the library aims to provide a **general framework** rather than one-off scripts. The intended pipeline is:

1. **Build specific ML/RL model instance** — User defines or selects a concrete model (e.g. GNN-LSTM pricer, Q-network for hedging) within the library’s model interfaces.
2. **Fetch / prepare input data** — Generic data loading and preparation (e.g. from market data, backtests, or simulation) into the format expected by the training pipeline.
3. **Train model via generic ML / Q-learning training pipeline** — A single, reusable training loop (or small set of loops) that works with any compliant model and data; supports checkpointing, logging, and hyperparameter control.
4. **Standardised model evaluation outputs** — Common evaluation metrics and outputs (e.g. loss curves, validation scores, RL returns) so that any model can be compared and monitored in a consistent way.
5. **Generalised projection / inference pipeline** — A standard way to load a trained model and run inference (e.g. price prediction, action recommendation) so that downstream applications (reports, algo bot, orchestrator) stay model-agnostic.

This keeps the framework **model-agnostic** and **reusable** across ML-based pricing, calibration, GNN-LSTM pricer, and Q-learning agents.

### 7.1 Machine Learning Integration
- [ ] **Generic ML Training Pipeline**
  - Implement: Reusable training loop (data → model → loss → optimizer step); checkpointing, logging
  - Support: Any model that conforms to a minimal trainable interface (forward, loss, optional validation)
  - Use case: Train NN pricers, calibration nets, GNN-LSTM, etc. through one pipeline

- [ ] **Data Preparation for ML**
  - Implement: Fetch/prepare input data from market data, MC paths, or portfolio representation
  - Support: Standardised feature/target format for pricing and calibration tasks
  - Use case: Feed generic training pipeline

- [ ] **Standardised ML Evaluation Outputs**
  - Implement: Common metrics (loss curves, validation error, pricing error vs. benchmark)
  - Support: Logging and serialisation so any ML model reports in a consistent way
  - Use case: Compare and monitor models

- [ ] **Generalised ML Inference Pipeline**
  - Implement: Load trained model → run inference (e.g. price, implied vol) in a model-agnostic way
  - Support: Integration with pricers, reports, and downstream applications
  - Use case: Deploy trained ML pricer or calibration model

- [ ] **ML-Based Pricing**
  - Implement: Neural network pricers (train on MC data) via the generic pipeline above
  - Use case: Fast approximate pricing

- [ ] **ML Calibration**
  - Implement: ML-based model calibration via the generic pipeline above
  - Use case: Fast calibration

- [ ] **Hybrid GNN-LSTM Full Revaluation Pricer** (Partially built)
  - Complete: `src/machine_learning/models/gnn_rnn_hybrid/` (attention, fusion, GNN/RNN layers, projection)
  - Integrate: Trade graph builder, attribute encoder, training manager with portfolio pricing
  - Train/evaluate/deploy via the generic ML pipeline (build instance → data → train → evaluate → inference)
  - Deliverable: Full revaluation pricer using graph + time-series representation of portfolio
  - Use case: Fast portfolio-level pricing and risk

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

### 7.5 Q-Learning & Reinforcement Learning Agents

The same **general framework** applies: build agent instance → fetch/prepare data (e.g. environments) → generic RL training pipeline → standardised evaluation (returns, risk metrics) → generalised inference (action selection in backtest/live).

- [ ] **Generic Q-Learning / RL Training Pipeline**
  - Implement: Reusable RL training loop (environment → agent → reward → update); support for standard algorithms (e.g. DQN, policy gradient)
  - Support: Any agent that conforms to a minimal interface (select action, update from transition/batch)
  - Use case: Train delta-hedging agent, algo-trading agent, or other RL policies through one pipeline

- [ ] **Environment & Data for RL**
  - Implement: Fetch/prepare environments (e.g. trading sim, hedging sim) from library pricers, market data, backtesting
  - Support: Standardised state/action/reward interface so different agents can plug in
  - Use case: Feed generic RL training pipeline

- [ ] **Standardised RL Evaluation Outputs**
  - Implement: Common metrics (returns, Sharpe, drawdown, episode stats) and logging
  - Support: Same evaluation format across agents for comparison and monitoring
  - Use case: Compare and monitor RL agents

- [ ] **Generalised RL Inference / Deployment**
  - Implement: Load trained agent → select actions in a model-agnostic way (backtest or live)
  - Support: Integration with backtesting engine, streaming engine, and orchestrator
  - Use case: Deploy delta-hedging or algo-trading agent

- [ ] **Q-Learning Framework** (stub: `src/q_learning/`)
  - Implement: Agent interface, environment wrapper for trading/hedging
  - Support: Discrete/continuous action spaces (e.g. hedge ratio, trade size)
  - Use case: Delta hedging agent, algo trading agent

- [ ] **RL Orchestrator**
  - Implement: Orchestrator that deploys RL agents (delta hedging, algo trading) using the generic inference pipeline
  - Integrate: With backtesting, streaming engine, and library pricers/risk
  - Use case: Automated hedging, strategy deployment, cutting-edge applications

### Deliverables:
- ✅ ML integration (pricing, calibration)
- ✅ Hybrid GNN-LSTM full revaluation pricer
- ✅ Q-learning / RL agent framework and orchestrator
- ✅ 3+ exotic products
- ✅ Optional: Credit/commodities

**Impact:** Demonstrates **cutting-edge knowledge** and **research-level** capabilities.

---

## Phase 8: Quant Hedge Fund & Execution Extensions

**Goal:** Add library components required for execution, factor risk, vol trading, portfolio optimisation, tail risk, real-time monitoring, and alternative data so that the additional application projects (5–12 below) can build on the library.

*These items are not yet implemented or not yet planned in Phases 1–7.*

### 8.1 Execution & Transaction Cost Analytics (TCA)
- [ ] **Execution Cost Models**
  - Implement: Market impact models (e.g. temporary/permanent), spread models
  - Support: Parameterised by size, volatility, liquidity proxy
  - Use case: Backtesting with realistic execution cost, TCA reporting

- [ ] **Optimal Execution**
  - Implement: Optimal execution framework (e.g. Almgren-Chriss-style or similar)
  - Support: Trade-off between market impact and timing risk; TWAP/VWAP-style targets
  - Use case: Execution & TCA application project, algo bot execution layer

- [ ] **TCA Metrics & Reporting**
  - Implement: Implementation shortfall, slippage vs benchmark, arrival price, volume participation
  - Support: Standardised TCA output for comparison and reporting
  - Use case: Execution & TCA application project

### 8.2 Factor Risk Model & Factor Attribution
- [ ] **Factor Exposure Computation**
  - Implement: Compute portfolio exposures to risk factors (e.g. rates, vol, sector, style)
  - Support: Factor definitions from library risk factors (Greeks, curves) or external factor returns
  - Use case: Factor risk report, portfolio optimisation

- [ ] **Factor Covariance / Factor Returns Interface**
  - Implement: Interface for factor covariance matrix or factor return series
  - Support: Sample or external factor model; integration with risk aggregation
  - Use case: Factor VaR, portfolio optimisation

- [ ] **Factor PnL Attribution**
  - Implement: PnL attribution by factor (exposure × factor return) alongside existing scenario attribution
  - Support: Portfolio-level and instrument-level factor breakdown
  - Use case: Factor risk & attribution application project, risk reports

### 8.3 Volatility Trading & Variance Swap Analytics
- [ ] **Variance Swap / Vol Swap Pricing**
  - Implement: Variance swap pricing (e.g. model-based from Heston/local vol), fair variance strike
  - Support: Integration with existing vol models
  - Use case: Volatility trading application project

- [ ] **Dispersion & Vol-of-Vol Analytics**
  - Implement: Index vs single-name dispersion metrics, vol-of-vol from existing models
  - Support: Relative value and dispersion trading analytics
  - Use case: Volatility trading application project

### 8.4 Portfolio Construction & Optimisation
- [ ] **Portfolio Optimisation API**
  - Implement: Mean-variance optimisation, risk parity, max Sharpe / min variance
  - Support: Constraints (turnover, sector, leverage, bounds); optional Black-Litterman
  - Use case: Portfolio optimisation application project, algo bot rebalance

- [ ] **Covariance / Risk Input for Optimisation**
  - Implement: Portfolio covariance from library (e.g. Greeks + factor cov, or sample)
  - Support: Same risk inputs as VaR and factor model where applicable
  - Use case: Portfolio optimisation, factor-aware optimisation

### 8.5 Tail Risk & Crisis Analytics
- [ ] **CVaR / Expected Shortfall**
  - Implement: Conditional VaR (expected shortfall) alongside VaR
  - Support: Historical, parametric, or simulation-based
  - Use case: Tail risk application project, risk reports

- [ ] **Tail Dependence & Crisis Scenarios**
  - Implement: Tail dependence metrics, crisis-regime scenarios (e.g. correlation breakdown)
  - Support: Stress scenarios that include tail/crisis behaviour
  - Use case: Tail risk & crisis analytics application project

### 8.6 Real-Time Risk & Limit Monitoring
- [ ] **Limit Monitoring & Alerts**
  - Implement: Threshold checks (VaR, Greeks, exposure, PnL), breach detection, alert interface
  - Support: Integration with streaming engine and risk aggregation
  - Use case: Real-time risk dashboard application project

### 8.7 Alternative Data for Alpha
- [ ] **Alternative Data Adapters & Featurisation**
  - Implement: Adapters for alternative data sources (e.g. sentiment, macro); featurisation into standard format
  - Support: Output compatible with generic ML training pipeline (Phase 7.1)
  - Use case: Alternative data alpha application project, ML strategies

### 8.8 Market-Making / Quoting Simulator Components
- [ ] **Spread & Inventory Model**
  - Implement: Spread rule (e.g. around fair value from pricers), simple inventory penalty
  - Support: Use by market-making simulator or RL quoting agent
  - Use case: Options market-making application project

### Deliverables (Phase 8):
- [ ] Execution models, optimal execution, TCA metrics (8.1)
- [ ] Factor exposure, factor cov/returns interface, factor attribution (8.2)
- [ ] Variance swap analytics, dispersion, vol-of-vol (8.3)
- [ ] Portfolio optimisation API, covariance input (8.4)
- [ ] CVaR, tail dependence, crisis scenarios (8.5)
- [ ] Limit monitoring and alerts (8.6)
- [ ] Alternative data adapters and featurisation (8.7)
- [ ] Spread/inventory model for market-making (8.8)

**Impact:** Library supports execution, factor risk, vol trading, portfolio optimisation, tail risk, real-time monitoring, alt data, and market-making application projects.

---

## After Library Completion: Application Projects

Once the core library is complete (Phases 1–8), the following **orchestrator/application projects** build on QuantStrata for production-style use cases. Each can be developed as a separate project (repo or subproject) that depends on the library.

- **Projects 1–4** depend primarily on Phases 1–7 (option analytics, algo bot, GNN-LSTM pricer, Q-learning orchestrator).
- **Projects 5–12** depend additionally on **Phase 8** (Quant Hedge Fund & Execution Extensions) for execution, factor risk, vol trading, portfolio optimisation, tail risk, real-time monitoring, alt data, and market-making components.

### Application Project 1: Option Pricing Analytic Report & Visualisation
- **Goal:** Comprehensive option pricing analytic report and visualisation for hedge fund risk management front office.
- **Scope:** Most advanced analytics and plots: Greeks surfaces, vol surfaces, PnL attribution, VaR/CVaR, stress scenarios, scenario fan charts, portfolio-level risk dashboards.
- **Library dependency:** Pricers, risk (attribution, VaR, stress), market data, calibration. Phase 5.6 (Advanced Analytics & Reporting) and 5.3 (Risk Infrastructure) feed this.
- **Deliverable:** Orchestrator scripts + report generation (e.g. PDF/HTML) and interactive dashboards.

### Application Project 2: Algorithmic Trading Bot
- **Goal:** Algo trading bot connected to practice/live brokerage, with streaming data, strategy deployment, backtesting, and performance evaluation.
- **Scope:** Streaming tick/bar data; connect to practice/live brokerage (paper and live modes); deploy strategies; backtest and evaluate performance; reports and plots; strategies can use ML/Q-learning from the library.
- **Library dependency:** Backtesting (Phase 5.2), streaming & live data (Phase 5.5), brokerage adapter (Phase 5.5), machine_learning/q_learning (Phase 7).
- **Deliverable:** Trading bot application: data feed → StreamingEngine → strategy (incl. ML/RL) → order execution (paper/live) → performance reports and visualisations.

### Application Project 3: Hybrid GNN-LSTM Full Revaluation Pricer
- **Goal:** Production implementation of the Hybrid GNN-LSTM full revaluation pricer (partially built in the library).
- **Scope:** Complete and harden `src/machine_learning/models/gnn_rnn_hybrid/`; integrate with portfolio representation (trade graph, attributes); train and serve as full revaluation pricer; validate vs. library pricers.
- **Library dependency:** machine_learning (Phase 7.1), portfolio, pricers, market data.
- **Deliverable:** Trained GNN-LSTM pricer service/model that can revalue portfolios using graph + time-series representation.

### Application Project 4: Q-Learning Orchestrator Agent
- **Goal:** Q-learning (RL) orchestrator that acts as an agent for delta hedging, algorithmic trading, and other cutting-edge applications.
- **Scope:** RL agent(s) for delta hedging (e.g. minimise PnL variance vs. cost); algo trading agent (e.g. execution, strategy selection); orchestrator that runs agents against live/backtest environments using library pricers and risk.
- **Library dependency:** q_learning (Phase 7.5), backtesting, streaming engine, pricers, risk.
- **Deliverable:** RL agent framework + orchestrator scripts for delta hedging, algo trading, and extensible agent-based use cases.

---

### Application Project 5: Execution & Transaction Cost Analytics (TCA)
- **Goal:** Execution quality analytics and TCA reporting for trading and algo strategies.
- **Scope:** Optimal execution (e.g. Almgren-Chriss), market impact models, implementation shortfall, TCA metrics vs. benchmarks (VWAP, arrival price).
- **Library dependency:** Phase 8.1 (execution cost models, optimal execution, TCA metrics), backtesting (5.2), streaming (5.5).
- **Deliverable:** Execution models + TCA report generation and integration with algo bot / backtesting.

### Application Project 6: Factor Risk Model & Factor Attribution
- **Goal:** Multi-factor risk and factor-based PnL attribution for portfolio and risk reporting.
- **Scope:** Factor exposures, factor covariance/returns, factor PnL attribution (“how much PnL from sector/momentum/vol?”); integration with existing scenario attribution.
- **Library dependency:** Phase 8.2 (factor exposure, factor cov/returns, factor attribution), portfolio, risk (5.3), reporting (5.6).
- **Deliverable:** Factor risk report and factor attribution outputs for risk dashboards and option analytics.

### Application Project 7: Options Market-Making Simulator
- **Goal:** Market-making simulator with inventory risk, bid/ask around fair value, and optional RL quoting agent.
- **Scope:** Spread and inventory model; quoting logic (library pricers for fair value); optional RL agent for quote placement; backtest/simulate market-making PnL.
- **Library dependency:** Phase 8.8 (spread/inventory model), pricers, calibration, risk, q_learning (7.5), backtesting.
- **Deliverable:** Market-making simulator application: fair value → spread/inventory → quotes → optional RL → PnL and risk analytics.

### Application Project 8: Volatility Trading & Variance Swap Analytics
- **Goal:** Volatility trading analytics: variance swap pricing, dispersion, vol-of-vol, and vol surface relative value.
- **Scope:** Variance swap / vol swap pricing; index vs single-name dispersion; vol-of-vol from library models; vol arbitrage analytics.
- **Library dependency:** Phase 8.3 (variance swap, dispersion, vol-of-vol), Heston/SABR/local vol, calibration, risk.
- **Deliverable:** Volatility trading analytics module and reports (variance swap, dispersion, surface analytics).

### Application Project 9: Portfolio Construction & Optimisation
- **Goal:** Portfolio optimisation and construction (mean-variance, risk parity, Black-Litterman) with constraints.
- **Scope:** Mean-variance, risk parity, max Sharpe / min variance; turnover, sector, leverage constraints; optional Black-Litterman views; rebalance workflows.
- **Library dependency:** Phase 8.4 (portfolio optimisation API, covariance input), portfolio, risk (5.3), backtesting.
- **Deliverable:** Portfolio optimisation service and integration with strategy construction / algo bot rebalance.

### Application Project 10: Tail Risk & Crisis Analytics
- **Goal:** Tail risk and crisis-regime analytics beyond standard VaR/stress.
- **Scope:** CVaR/expected shortfall, tail dependence, crisis-regime scenarios, correlation breakdown; reporting and visualisation.
- **Library dependency:** Phase 8.5 (CVaR, tail dependence, crisis scenarios), risk (5.3), stress testing, reporting (5.6).
- **Deliverable:** Tail risk and crisis analytics reports and dashboards (CVaR, tail dependence, crisis scenarios).

### Application Project 11: Real-Time Risk & Intraday Limit Monitoring
- **Goal:** Real-time risk dashboard and intraday limit monitoring.
- **Scope:** Live Greeks, intraday VaR, limit checks (VaR, Greeks, exposure, PnL), breach alerts; optional margin/SIMM-style view.
- **Library dependency:** Phase 8.6 (limit monitoring, alerts), streaming (5.5), pricers, risk (5.3), brokerage/positions.
- **Deliverable:** Real-time risk dashboard: streaming data → library pricers/risk → limits and alerts.

### Application Project 12: Alternative Data → Alpha / ML Pipeline
- **Goal:** Alternative data ingestion, featurisation, and integration with ML/alpha research and backtesting.
- **Scope:** Adapters for alternative data (e.g. sentiment, macro); featurisation into standard format; feed into generic ML pipeline and backtesting for alpha/strategy research.
- **Library dependency:** Phase 8.7 (alt data adapters, featurisation), generic ML pipeline (7.1), backtesting (5.2), algo bot (Project 2).
- **Deliverable:** Alternative data pipeline and integration with ML training and backtesting for alpha strategies.

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
| Phase 5 | Weeks 25-30+ | Production | Calibration, backtesting, risk, **streaming/live (5.5)**, **analytics/reports (5.6)** |
| Phase 6 | Ongoing | Education | Tutorials, notebooks, docs |
| Phase 7 | Weeks 31-36+ | Advanced Topics | ML, **GNN-LSTM pricer**, **Q-learning/RL agents (7.5)**, exotics, credit/commodities |
| Phase 8 | After 7 | **Quant HF & Execution** | Execution/TCA (8.1), factor risk (8.2), vol trading (8.3), portfolio opt (8.4), tail risk (8.5), limit monitoring (8.6), alt data (8.7), market-making (8.8) |
| *After library* | — | **Application Projects 1–12** | Option analytics, Algo bot, GNN-LSTM pricer, Q-learning orchestrator; **5** Execution/TCA, **6** Factor risk, **7** Market-making, **8** Vol trading, **9** Portfolio opt, **10** Tail risk, **11** Real-time risk, **12** Alt data alpha |

**Total Timeline:** ~9 months for core functionality, ongoing for education/advanced topics; **Phase 8** (quant hedge fund & execution extensions) follows Phase 7; **application projects 1–12** (option report, algo bot, GNN-LSTM pricer, Q-learning orchestrator, execution/TCA, factor risk, market-making, vol trading, portfolio opt, tail risk, real-time risk, alt data alpha) follow library completion.

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
2. Complete remaining Phase 5 (risk, streaming 5.5, analytics 5.6, performance) and Phase 6–7
3. Iterate based on learnings
4. Maintain quality standards throughout
5. **After library completion:** Complete Phase 8 (quant hedge fund & execution extensions) then build application projects 1–12 (option analytics, algo bot, GNN-LSTM pricer, Q-learning orchestrator, execution/TCA, factor risk, market-making, vol trading, portfolio opt, tail risk, real-time risk, alt data alpha)

**The foundation is excellent. Time to build! 🚀**
