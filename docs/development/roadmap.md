# QuantStrata Development Roadmap

**Last Updated:** January 27, 2026  
**Progress:** All core phases (1-8) implemented with tests and documentation  
**Current Version:** V1 (FX Derivatives Foundation)  
**Target:** Comprehensive Professional Quant Library

---

## Implementation Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1-6 | Core Foundation (FX, Equity, IR, Risk, Backtesting, Calibration) | ✅ Complete |
| 7.1 | Machine Learning Integration (GNN-LSTM, Training, Production ML) | ✅ Complete |
| 7.2 | Q-Learning / RL Framework (Environments, Runners) | ✅ Complete |
| 7.3 | Exotic Products (Cliquet, Autocallable, Range Accrual) | ✅ Complete |
| 7.4 | Deep Hedging Core | ✅ Complete |
| 7.5 | Neural SDE (Networks, Solvers, Training) | ✅ Complete |
| 7.6 | Deep Hedging Backtesting (Adapters, Environments, Metrics) | ✅ Complete |
| 8.1 | Volatility Trading (Variance Swaps, Dispersion, Vol-of-Vol) | ✅ Complete |
| 8.2 | Portfolio Optimisation (MV, Risk Parity, Black-Litterman) | ✅ Complete |
| App | Application Projects (Dash UIs) | 📋 Planned |

**Detailed checklist:** See `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md`

---

## Implementation Checklist Standard

For every new component, the following deliverables are required:

| Deliverable | Location | Description |
|-------------|----------|-------------|
| **Implementation** | `src/<module>/` | Core code with type hints and docstrings |
| **Unit Tests** | `tests/unit/<module>/` | Comprehensive test coverage (>90%) |
| **Reference Doc** | `docs/reference/<module>/` | Technical specification and API reference |
| **Guide Doc** | `docs/guides/<module>/` | How-to guide with examples |
| **Tutorial Notebook** | `docs/tutorials/<module>/` | Interactive Jupyter notebook (where applicable) |
| **Pipeline Check** | `src/orchestrator/pipelines/` | Assess if orchestrator pipeline needed |
| **Example Script** | `examples/pipelines/` | If pipeline exists, add example script |

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

### 7.1 Machine Learning Integration ✅
- [x] **Generic ML Training Pipeline**
  - Implemented: `pipelines/training.py` (Trainable + NumPy), `training/trainer.py` (TensorFlow-native), `calibration/training_manager.py` (Keras/HybridGnnRnn)
  - Support: Trainable protocol, KerasTrainableAdapter; checkpointing, logging, early stopping
  - Use case: Train NN pricers, calibration nets, GNN-LSTM through one pipeline or model-specific manager

- [x] **Data Preparation for ML**
  - Implemented: `data/dataset.py` (TFDataset, create_pricing_dataset, create_calibration_dataset), `data/pricing/build.py`, `data/calibration/build.py`, `data/portfolio.py`, `data/gnn_rnn_hybrid/build.py`
  - Support: build_pricing_data, build_pricing_dataset_from_mc/from_analytic, build_calibration_dataset, build_gnn_data, build_gnn_dataset_from_portfolio
  - Use case: Feed generic training pipeline with standardised feature/target format

- [x] **Standardised ML Evaluation Outputs**
  - Implemented: `pipelines/evaluation.py`, `evaluation/evaluator.py`, `evaluation/metrics.py`
  - Support: Common metrics (loss curves, validation error, pricing error vs. benchmark), logging
  - Use case: Compare and monitor models in a consistent way

- [x] **Generalised ML Inference Pipeline**
  - Implemented: `pipelines/inference.py` (Trainable + JSON), `inference/model_io.py` (TF/Keras save/load), `inference/predictor.py`
  - Support: Load trained model → run inference in a model-agnostic way; integration with pricers and reports
  - Use case: Deploy trained ML pricer or calibration model

- [x] **ML-Based Pricing**
  - Implemented: `models/pricing/`, `create_pricing_dataset`, `build_pricing_data`, `build_pricing_dataset_from_mc/from_analytic`
  - Use case: Fast approximate pricing (train on MC or analytic data)

- [x] **ML Calibration**
  - Implemented: `create_calibration_dataset`, `data/calibration/build_calibration_dataset`
  - Use case: ML-based model calibration (IV surface → model parameters)

- [x] **Hybrid GNN-LSTM Full Revaluation Pricer**
  - Complete: `src/machine_learning/models/gnn_rnn_hybrid/` (attention, fusion, GNN/RNN layers, projection)
  - Integrated: Trade graph builder, attribute encoder, TrainingManager with portfolio-style data (`build_gnn_data`, FX/synthetic)
  - Train/evaluate/deploy: build instance → data → TrainingManager.run() or generic pipeline → evaluate → inference (save/load weights)
  - Deliverable: Full revaluation pricer using graph + time-series representation of portfolio
  - Use case: Fast portfolio-level pricing and risk

**Status:** Phase 7.1 Core COMPLETE. See `docs/development/progress/phase_7_1_implementation_notes.md` and `docs/development/progress/phase_7_1_machine_learning_integration.md`. Technical reference: `docs/reference/machine_learning/ml_framework.md`. Tutorials: `docs/tutorials/machine_learning/` (ML lifecycle, Hybrid GNN-LSTM).

### 7.1.5 Production ML Infrastructure ✅

**Goal:** Add production-grade ML tooling for experiment tracking, hyperparameter tuning, and model versioning. Note: `machine_learning/` is already production-quality; these additions extend existing capabilities.

- [x] **Experiment Tracking Integration** ✅
  - Implemented: `MLflowTracker`, `WandBTracker`, `InMemoryTracker` with common `ExperimentTracker` protocol
  - Location: `src/machine_learning/core/tracking.py`
  - Support: Log metrics, parameters, artifacts; integration with training pipelines
  - Tests: `tests/unit/machine_learning/core/test_tracking.py`
  - Docs: `docs/reference/machine_learning/production_ml.md`

- [x] **Hyperparameter Tuning Extensions** ✅
  - Implemented: `SearchSpace`, `MedianPruner`, `PercentilePruner`, `TuningResult`, `run_optuna_tuning`
  - Location: `src/machine_learning/tuning/search_space.py`
  - Support: Bayesian optimisation (Optuna), pruning, parallel trials
  - Tests: `tests/unit/machine_learning/tuning/test_search_space.py`
  - Pipeline: `src/orchestrator/pipelines/ml/hyperparameter_tuning.py`
  - Example: `examples/pipelines/run_hyperparameter_tuning.py`

- [x] **Model Registry & Versioning** ✅
  - Implemented: `ModelRegistry`, `ModelArtifact`, `ModelVersion`, `ModelStage`
  - Location: `src/machine_learning/registry/registry.py`
  - Support: Version tracking, metadata, promotion (staging → production)
  - Use case: Production model management and deployment

**Implementation Checklist:**
```
[x] src/machine_learning/core/tracking.py (MLflowTracker, WandBTracker, TrackingProtocol)
[x] src/machine_learning/tuning/search_space.py (SearchSpace, TrialPruner) — if pipelines/tuning.py insufficient
[x] src/machine_learning/registry/registry.py (ModelRegistry, ModelArtifact, ModelVersion)
[x] tests/unit/machine_learning/core/test_tracking.py
[x] tests/unit/machine_learning/tuning/test_search_space.py
[x] tests/unit/machine_learning/registry/test_registry.py
[x] docs/reference/machine_learning/production_ml.md
[ ] docs/guides/machine_learning/experiment_tracking.md
[ ] docs/guides/machine_learning/hyperparameter_tuning.md
[ ] docs/tutorials/machine_learning/ml_production.ipynb
[x] Pipeline: src/orchestrator/pipelines/ml/hyperparameter_tuning.py
[x] Example: examples/pipelines/run_hyperparameter_tuning.py
```

**Status:** ✅ **Complete** (implementation, tests, reference doc, pipeline, example). **Gaps:** Guide docs (experiment_tracking, hyperparameter_tuning), tutorial notebook. Dependencies: Phase 7.1 complete.

### 7.2 Q-Learning & Reinforcement Learning Agents

The same **general framework** as ML applies: build agent instance → fetch/prepare data (e.g. environments) → generic RL training pipeline → standardised evaluation (returns, risk metrics) → generalised inference (action selection in backtest/live).

- [x] **Generic Q-Learning / RL Training Pipeline**
  - Implement: Reusable RL training loop (environment → agent → reward → update); support for standard algorithms (e.g. DQN, policy gradient)
  - Support: Any agent that conforms to a minimal interface (select action, update from transition/batch)
  - Use case: Train delta-hedging agent, algo-trading agent, or other RL policies through one pipeline
  - **Implemented:** `src/q_learning/pipelines/training.py` — `run_training()`, `RLTrainingLoop`; checkpointing, eval episodes.

- [x] **Environment & Data for RL**
  - Implement: Fetch/prepare environments (e.g. trading sim, hedging sim) from library pricers, market data, backtesting
  - Support: Standardised state/action/reward interface so different agents can plug in
  - Use case: Feed generic RL training pipeline
  - **Implemented:** `RLEnvironment` protocol; `BaseEnv` in `src/q_learning/environments/base.py` (minimal env for tests/template). Trading/hedging sims to wrap backtesting/pricers in future.

- [x] **Standardised RL Evaluation Outputs**
  - Implement: Common metrics (returns, Sharpe, drawdown, episode stats) and logging
  - Support: Same evaluation format across agents for comparison and monitoring
  - Use case: Compare and monitor RL agents
  - **Implemented:** `evaluate_agent()` in `pipelines/evaluation.py`; `RLEvaluationResult`; `sharpe_ratio`, `max_drawdown`, `win_rate` in `evaluation/metrics.py`.

- [x] **Generalised RL Inference / Deployment**
  - Implement: Load trained agent → select actions in a model-agnostic way (backtest or live)
  - Support: Integration with backtesting engine, streaming engine, and orchestrator
  - Use case: Deploy delta-hedging or algo-trading agent
  - **Implemented:** `save_agent()`, `load_agent()`, `select_action()` in `pipelines/inference.py`; artifact layout (parameters.json, config.json, metadata.json).

- [x] **Q-Learning Framework** (stub: `src/q_learning/`)
  - Implement: Agent interface, environment wrapper for trading/hedging
  - Support: Discrete/continuous action spaces (e.g. hedge ratio, trade size)
  - Use case: Delta hedging agent, algo trading agent
  - **Implemented:** `RLAgent`, `RLEnvironment` in `core/protocols.py`; `Transition`, `RLTrainingConfig`, `RLTrainingResult`, `RLEvaluationResult` in `core/types.py`; pipelines and BaseEnv as above.

- [x] **RL Agent Deployment & Environments**
  - Implement: Trading and hedging environments that wrap backtesting/streaming infrastructure
  - Implement: Agent runners that execute trained agents in backtest or live contexts
  - **Implemented:** `src/q_learning/environments/trading.py`, `hedging.py`, `streaming.py`; `src/q_learning/runners/base.py`, `backtest.py`, `live.py`; unit tests for environments and runners
  - Use case: Automated hedging, strategy deployment, cutting-edge applications
  - Note: Orchestrator pipelines in `src/orchestrator/pipelines/rl/` not yet added

**RL Deployment Implementation Checklist:**
```
Environments (extend existing q_learning/environments/):
[x] src/q_learning/environments/trading.py (TradingEnvironment wrapping backtesting)
[x] src/q_learning/environments/hedging.py (HedgingEnvironment wrapping pricers)
[x] src/q_learning/environments/streaming.py (StreamingEnvironment for live execution)

Runners (agent execution utilities):
[x] src/q_learning/runners/backtest.py (BacktestRunner - run agent in backtesting framework)
[x] src/q_learning/runners/live.py (LiveRunner - run agent with streaming engine)
[x] src/q_learning/runners/base.py (BaseRunner protocol)

Orchestrator Pipelines (in src/orchestrator/pipelines/):
[ ] src/orchestrator/pipelines/rl/deploy_agent.py (orchestrator-level deployment)
[ ] src/orchestrator/pipelines/rl/backtest_agent.py (orchestrator-level backtesting)

Tests:
[x] tests/unit/q_learning/environments/test_trading.py
[x] tests/unit/q_learning/environments/test_hedging.py (or covered in test_backtest.py)
[x] tests/unit/q_learning/runners/test_backtest.py

Documentation:
[ ] docs/reference/q_learning/environments.md
[ ] docs/reference/q_learning/runners.md
[ ] docs/guides/q_learning/deploying_rl_agents.md
[ ] docs/tutorials/q_learning/rl_deployment_tutorial.ipynb
[ ] Example: examples/pipelines/run_deploy_rl_agent.py
```

**Status:** ✅ Phase 7.2 complete (training, evaluation, inference, protocols, environments, runners, tests). **Gaps:** Orchestrator rl pipelines (`rl/deploy_agent.py`, `rl/backtest_agent.py`), reference docs (environments.md, runners.md), guide (deploying_rl_agents.md), tutorial notebook, example script. See `docs/development/progress/phase_7_2_q_learning.md`. Technical reference: `docs/reference/q_learning/rl_framework.md`. Guide: `docs/guides/q_learning/rl_framework.md`.

### 7.3 Exotic Products ✅

**Goal:** Implement structured products commonly traded by hedge funds and investment banks.

- [x] **Cliquet Options**
  - Instrument: `CliquetOption` (periodic resets with local/global caps and floors)
  - Parameters: reset_dates, local_cap, local_floor, global_cap, global_floor, participation
  - Payoff: `CliquetPayoff` (path-dependent, sum of capped/floored periodic returns)
  - Pricer: `CliquetMcPricer` (MC required for path-dependency)
  - **Implemented:** `src/instruments/equity/options/cliquet.py`, `src/models/payoffs/cliquet.py`, `src/pricers/equity/cliquet_gbm_mc.py`
  - Use case: Equity-linked structured notes, guaranteed return products

- [x] **Autocallable Products**
  - Instrument: `AutocallableOption` (barrier observation dates, coupon, early redemption)
  - Parameters: observation_dates, autocall_barrier, coupon_barrier, put_barrier, coupon_rate
  - Payoff: `AutocallablePayoff` (early termination on barrier breach with coupon)
  - Pricer: `AutocallableMcPricer` (MC required)
  - **Implemented:** `src/instruments/equity/options/autocallable.py`, `src/models/payoffs/autocallable.py`, `src/pricers/equity/autocallable_gbm_mc.py`
  - Use case: Most popular structured product globally (>$100B annual issuance)

- [x] **Range Accrual**
  - Instrument: `RangeAccrualNote` (IR or FX underlying)
  - Parameters: range_lower, range_upper, observation_freq, notional, accrual_rate
  - Payoff: `RangeAccrualPayoff` (accrues on days within range)
  - Pricer: `RangeAccrualHwMcPricer` (Hull-White MC)
  - **Implemented:** `src/instruments/ir/options/range_accrual.py`, `src/models/payoffs/range_accrual.py`, `src/pricers/ir/range_accrual_hw_mc.py`
  - Use case: Yield enhancement, low-vol betting

**Pricer Naming Convention:** `{product}_{model}_{method}.py` — e.g. `cliquet_gbm_mc.py`, `autocallable_gbm_mc.py`, `range_accrual_hw_mc.py`.

**Implementation Checklist:**
```
Implementation & tests:
[x] src/instruments/equity/options/cliquet.py, autocallable.py; src/instruments/ir/options/range_accrual.py
[x] src/models/payoffs/cliquet.py, autocallable.py, range_accrual.py
[x] src/pricers/equity/cliquet_gbm_mc.py, autocallable_gbm_mc.py; src/pricers/ir/range_accrual_hw_mc.py
[x] tests/unit/... (instruments, pricers) as implemented

Documentation:
[ ] docs/reference/instruments/exotic_products.md
[ ] docs/guides/instruments/pricing_exotics.md
[ ] docs/tutorials/pricing/exotic_options.ipynb

Pipeline & example:
[ ] Pipeline: pricing pipeline extension or exotic pricing step
[ ] Example: examples/pricing/exotic_structured_products.py (or 02_exotic_options.py)
```

**Status:** ✅ **Implementation and unit tests complete.** **Gaps:** Reference doc, guide doc, tutorial notebook, pipeline, example script. See `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` (Phase 7.3).

### 7.4 Credit Derivatives (Optional)
- [ ] **Credit Default Swaps (CDS)**
  - Implement: `CreditDefaultSwap`
  - Model: Reduced-form credit model
  - Use case: Credit risk

- [ ] **Credit Options**
  - Implement: Options on CDS
  - Use case: Credit volatility trading

### 7.5 Commodities (Optional)
- [ ] **Commodity Options**
  - Implement: `CommodityOption`
  - Model: GBM with convenience yield
  - Use case: Energy/agricultural derivatives

### Deliverables:
- ✅ ML integration (pricing, calibration)
- ✅ Hybrid GNN-LSTM full revaluation pricer
- ✅ Q-learning / RL agent framework and orchestrator
- ✅ 3+ exotic products
- ✅ Optional: Credit/commodities

**Impact:** Demonstrates **cutting-edge knowledge** and **research-level** capabilities.

---

## Phase 7.6: Deep Hedging & Neural Optimal Control

**Goal:** Implement a deep hedging framework that trains RL agents to learn optimal hedging strategies accounting for transaction costs, discrete rehedging, and market impact — going beyond classical delta hedging.

**Research Foundation:** Bühler et al. (2019) "Deep Hedging", Horvath et al. (2021) "Deep Hedging under Rough Volatility"

### 7.6.1 Hedging Environment
- [x] **HedgingEnv**
  - Implement: RL environment wrapping pricers and market simulation for hedging
  - State: spot price, time to expiry, current position, Greeks (delta, gamma, vega), recent volatility, PnL
  - Action: hedge ratio (continuous) or discrete hedge amounts
  - Reward: risk-adjusted P&L minus transaction costs
  - Support: Configurable transaction cost model (proportional, fixed, market impact)
  - Use case: Train deep hedging agents
  - **Implemented:** `src/deep_hedging/core/protocols.py` (HedgingEnvironment, BaseHedgingEnv), `src/deep_hedging/environments/gbm.py` (GBMHedgingEnv)

- [x] **Transaction Cost Model**
  - Implement: Proportional spread, fixed cost, temporary/permanent market impact
  - Support: Parameterised by asset, size, volatility
  - Use case: Realistic hedging simulation, execution cost awareness
  - **Implemented:** `src/deep_hedging/core/costs.py` (ProportionalCost, FixedCost, MarketImpactCost, CombinedCost)

- [x] **Market Simulation for Hedging**
  - Implement: Simulate underlying paths (GBM, Heston, or learned dynamics) with discrete rehedging
  - Support: Multiple paths per episode for variance reduction
  - Use case: Generate training data for deep hedging
  - **Implemented:** `src/deep_hedging/environments/gbm.py` (reuses GbmDynamicsSimulator), antithetic variates for variance reduction

### 7.6.2 Deep Hedging Agent
- [x] **Hedging Policy Network**
  - Implement: Neural network policy that outputs hedge ratio given state
  - Architecture: MLP or recurrent (LSTM) for path-dependent hedging
  - Support: Continuous action space (hedge ratio as fraction of delta)
  - Use case: Learn non-linear hedging policy
  - **Implemented:** `src/deep_hedging/agents/deep.py` (MLPPolicy, DeepHedgingAgent)

- [x] **Risk-Aware Loss Function**
  - Implement: Loss based on distribution of terminal P&L (CVaR, variance, utility)
  - Support: Configurable risk measure (variance penalty, CVaR, exponential utility)
  - Use case: Train agents with different risk preferences
  - **Implemented:** `src/deep_hedging/core/risk_measures.py` (VarianceRisk, MeanVarianceRisk, CVaRRisk, EntropicRisk)

- [x] **Training Pipeline Integration**
  - Implement: Integrate with existing RL training pipeline (run_training, evaluate_agent)
  - Support: Checkpointing, logging, evaluation against delta-hedging benchmark
  - Use case: End-to-end deep hedging training
  - **Implemented:** `src/deep_hedging/training/trainer.py` (HedgingTrainer, train_deep_hedging)

### 7.6.3 Evaluation & Benchmarking
- [x] **Delta Hedging Benchmark**
  - Implement: Classical delta-hedging agent for comparison
  - Support: With and without transaction costs
  - Use case: Benchmark deep hedging against standard approach
  - **Implemented:** `src/deep_hedging/agents/delta.py` (DeltaHedgingAgent, NoHedgingAgent)

- [x] **Hedging Performance Metrics**
  - Implement: P&L distribution stats (mean, std, Sharpe, max drawdown, CVaR)
  - Support: Cost breakdown (hedging cost vs tracking error)
  - Use case: Compare hedging strategies
  - **Implemented:** `src/deep_hedging/evaluation/evaluator.py` (compute_hedging_metrics, HedgingEvaluator, compare_agents)

- [x] **Backtesting Integration**
  - Implement: Run trained hedging agent in backtesting framework
  - Support: Historical data replay, out-of-sample evaluation
  - **Implemented:** `src/deep_hedging/adapters/backtesting.py` (BacktestEngineAdapter), `src/deep_hedging/adapters/historical_data.py` (HistoricalDataAdapter), `src/deep_hedging/evaluation/backtest_metrics.py` (HedgingBacktestMetrics), multi-asset and historical environments
  - Use case: Validate deep hedging on real data
  - Tests: `tests/unit/deep_hedging/adapters/test_backtesting.py`, `test_historical_data.py`, `evaluation/test_backtest_metrics.py`, `environments/test_multi_asset.py`, `test_historical.py`
  - Pipeline and example script: see `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` (Phase 7.6).

**Note on Deep Hedging Architecture:**
Deep hedging is fundamentally an RL application (agents, environments, training). It's kept as a separate module because:
1. It's a recognised research field with specific terminology (Bühler et al. "Deep Hedging")
2. Contains domain-specific components: transaction costs, risk measures (CVaR, entropic), hedging evaluation metrics
3. Users searching for "deep hedging" expect a dedicated module
The structure mirrors `q_learning/` but with hedging-specific components.

### 7.6.4 Advanced Deep Hedging
- [x] **Multi-Asset Hedging**
  - Implement: Hedge portfolio of options with multiple underlyings
  - **Implemented:** `src/deep_hedging/environments/multi_asset.py` (MultiAssetHedgingEnv)
  - Use case: Portfolio-level deep hedging

- [x] **Model-Agnostic Hedging**
  - Implement: Train hedging agent without assuming specific dynamics; learn from historical data
  - **Implemented:** `src/deep_hedging/environments/historical.py` (HistoricalHedgingEnv), `src/deep_hedging/adapters/historical_data.py`
  - Use case: Robust hedging under model uncertainty

**Implementation Checklist (backtesting & advanced):**
```
[x] src/deep_hedging/adapters/backtesting.py (BacktestEngineAdapter)
[x] src/deep_hedging/adapters/historical_data.py (HistoricalDataAdapter)
[x] src/deep_hedging/evaluation/backtest_metrics.py (HedgingBacktestMetrics)
[x] src/deep_hedging/environments/multi_asset.py (MultiAssetHedgingEnv)
[x] src/deep_hedging/environments/historical.py (HistoricalHedgingEnv)
[x] tests/unit/deep_hedging/adapters/test_backtesting.py, test_historical_data.py
[x] tests/unit/deep_hedging/evaluation/test_backtest_metrics.py
[x] tests/unit/deep_hedging/environments/test_multi_asset.py, test_historical.py
[x] docs/reference/deep_hedging/theory.md
[ ] docs/guides/deep_hedging/backtesting_hedging_agents.md
[ ] docs/guides/deep_hedging/multi_asset_hedging.md
[ ] docs/guides/deep_hedging/model_agnostic_hedging.md
[ ] Update: docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb (backtesting section)
[ ] Pipeline: src/orchestrator/pipelines/deep_hedging/backtest_agent.py
[ ] Example: examples/pipelines/run_backtest_hedging_agent.py
```

**Status:** ✅ **Core and backtesting integration complete.** Environments, agents, training, evaluation, adapters, multi-asset and historical envs implemented. **Gaps:** Guide docs, tutorial backtesting section, pipeline, example script. See `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` (Phase 7.6) and `docs/development/progress/phase_7_6_deep_hedging.md`.

**Documentation:**
- Theory: `docs/reference/deep_hedging/theory.md` (PhD-level technical reference)
- Tutorial: `docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb`

**Dependencies:** Phase 7.2 (RL framework), pricers (Greeks), MC simulation, backtesting.

---

## Phase 7.7: Neural SDE & Generative Market Simulation

**Goal:** Implement neural stochastic differential equations that learn drift and diffusion functions from data, enabling more realistic market simulation than parametric models (GBM, Heston).

**Research Foundation:** Kidger et al. (2021) "Neural SDEs", Gierjatowicz et al. (2020) "Robust pricing and hedging via neural SDEs"

### 7.7.1 Neural SDE Framework
- [x] **Neural Drift and Diffusion**
  - Implement: Neural networks μ_θ(S, t) and σ_θ(S, t) for SDE: dS = μ_θ dt + σ_θ dW
  - **Implemented:** `src/models/neural_sde/networks.py` (NeuralDriftNetwork, NeuralDiffusionNetwork)
  - Use case: Learn realistic price dynamics from data

- [x] **SDE Solver Integration**
  - Implement: Euler-Maruyama and Milstein solvers for neural SDEs
  - **Implemented:** `src/models/neural_sde/solvers.py`
  - Use case: Generate paths from learned dynamics

- [ ] **Adjoint Sensitivity Method**
  - Implement: Memory-efficient backpropagation through SDE solver
  - Support: Gradient computation for long paths
  - Use case: Train neural SDE on long time series

### 7.7.2 Training Neural SDEs
- [ ] **Score Matching / Maximum Likelihood**
  - Implement: Train neural SDE by matching marginal distributions
  - Support: Score matching loss, Wasserstein distance, MMD
  - Use case: Learn dynamics from historical returns

- [ ] **Calibration to Options**
  - Implement: Calibrate neural SDE to match implied volatility surface
  - Support: Joint calibration to spot dynamics and vol surface
  - Use case: Risk-neutral dynamics for pricing

- [x] **Training Pipeline**
  - Implement: Data pipeline (historical returns → training batches), validation, early stopping, checkpointing
  - **Implemented:** `src/models/neural_sde/training/losses.py`, `trainer.py` (NeuralSDETrainer)
  - Use case: End-to-end neural SDE training

### 7.7.3 Integration with Pricing
- [ ] **Neural SDE Monte Carlo Pricer**
  - Implement: MC pricer using learned neural SDE dynamics
  - Support: Any payoff from existing payoff framework
  - Use case: Price options under learned dynamics

- [ ] **Greeks via Automatic Differentiation**
  - Implement: Compute Greeks by differentiating through neural SDE
  - Support: Delta, gamma, vega (w.r.t. initial vol)
  - Use case: Risk management with neural dynamics

### 7.7.4 Generative Scenario Simulation
- [x] **Path Generator**
  - Implement: Generate paths from trained neural SDE dynamics
  - **Implemented:** `src/models/neural_sde/generation/generator.py`
  - Use case: Scenario generation, synthetic paths for pricing/training

- [ ] **Conditional Generation**
  - Implement: Generate paths conditioned on regime (e.g., high vol, crisis)
  - Support: Conditioning on VIX level, macro variables
  - Use case: Stress testing with realistic conditional scenarios

- [ ] **Synthetic Data Augmentation**
  - Implement: Generate synthetic market data for training other models
  - Support: Preserve statistical properties (vol clustering, fat tails)
  - Use case: Data augmentation for ML models

**Implementation Checklist:**
```
Neural SDE Framework (7.7.1):
[x] src/models/neural_sde/networks.py (NeuralDriftNetwork, NeuralDiffusionNetwork)
[x] src/models/neural_sde/solvers.py (EulerMaruyamaSolver, MilsteinSolver)
[ ] src/models/neural_sde/adjoint.py (AdjointSDEMethod)
[x] src/models/neural_sde/dynamics.py (NeuralSDEDynamics)
[x] tests/unit/models/neural_sde/test_networks.py
[x] tests/unit/models/neural_sde/test_solvers.py

Training (7.7.2):
[ ] src/models/neural_sde/training/score_matching.py (ScoreMatchingTrainer)
[ ] src/models/neural_sde/training/calibration.py (NeuralSDECalibrator)
[x] src/models/neural_sde/training/losses.py, trainer.py (NeuralSDETrainer)
[x] tests/unit/models/neural_sde/training/test_*.py (as implemented)

Pricing Integration (7.7.3):
[ ] src/pricers/neural_sde/mc_pricer.py (NeuralSDEMcPricer)
[ ] src/pricers/neural_sde/greeks.py (NeuralSDEGreeksCalculator)
[ ] tests/unit/pricers/neural_sde/test_mc_pricer.py

Generative Simulation (7.7.4):
[x] src/models/neural_sde/generation/generator.py (PathGenerator)
[ ] src/models/neural_sde/generation/conditional.py (ConditionalGenerator)
[ ] src/models/neural_sde/generation/augmentation.py (SyntheticDataAugmenter)
[ ] tests/unit/models/neural_sde/generation/test_conditional.py

Documentation:
[ ] docs/reference/models/neural_sde.md
[ ] docs/guides/models/training_neural_sde.md
[ ] docs/tutorials/models/neural_sde_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/ml/train_neural_sde.py
[ ] Example: examples/pipelines/run_train_neural_sde.py
```

**Status:** ✅ **Implementation complete** (networks, solvers, dynamics, training losses, trainer, path generator). **Gaps:** Adjoint method; score matching / calibration; pricing integration (MC pricer, Greeks); conditional/augmentation generation; reference/guide docs; tutorial notebook; orchestrator pipeline; example script. See `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` (Phase 7.7).

**Dependencies:** MC framework, calibration engine, ML pipelines (7.1).

---

## Phase 7.8: Rough Volatility Models [DEFERRED]

**Status:** 🔲 **DEFERRED** - Research-level topic, not essential for library completion.

**Rationale for Deferral:**
- Rough volatility is cutting-edge research (2016-2018 papers)
- Complex implementation (~25 files, ~6,000 lines)
- Limited practical use outside academic research
- Library already has Heston, SABR, Local Vol for production needs

**If Revisited Later:**
- Research Foundation: Gatheral, Jaisson & Rosenbaum (2018) "Volatility is Rough"
- Key components: Fractional BM, Rough Bergomi, Rough Heston, Calibration
- See archived roadmap notes for full specification

---

### Phase 7 Deliverables Summary:
- ✅ ML integration (pricing, calibration) — 7.1
- ✅ Hybrid GNN-LSTM full revaluation pricer — 7.1
- ✅ Production ML infrastructure (tracking, tuning, registry) — 7.1.5
- ✅ Q-learning / RL agent framework (environments, runners) — 7.2
- ✅ Exotic products (Cliquet, Autocallable, Range Accrual) — 7.3
- 🔲 Credit derivatives — 7.4 (optional, deferred)
- 🔲 Commodities — 7.5 (optional, deferred)
- ✅ Deep hedging framework (core, backtesting, multi-asset, historical) — 7.6
- ✅ Neural SDE (networks, solvers, dynamics, training, generation) — 7.7
- 🔲 Rough volatility — 7.8 (deferred)

**Impact:** Demonstrates **cutting-edge ML/RL capabilities** in modern quantitative finance: deep hedging, neural SDEs, and hybrid GNN-LSTM pricing.

---

## Phase 8: Focused Extensions (Vol Trading & Portfolio Construction)

**Goal:** Add two high-value, focused extensions that naturally complement the existing library.

**Scope Rationale:** Phase 8 was originally much larger (10 sub-phases). After project assessment, the scope has been reduced to two focused areas that:
1. Build directly on existing infrastructure (vol models, portfolio module)
2. Have clear practical value
3. Are achievable without creating maintenance burden

### 8.1 Volatility Trading & Variance Swap Analytics ✅

- [x] **Variance Swap / Vol Swap Pricing** ✅
  - Implemented: `VarianceSwap`, `VarianceSwapPricer`, `VarianceSwapResult`
  - Location: `src/volatility/trading/variance_swap.py`
  - Support: Log-strip replication, discrete monitoring adjustments
  - Tests: `tests/unit/volatility/trading/test_variance_swap.py`
  - Docs: `docs/reference/volatility/vol_trading.md`

- [x] **Dispersion & Vol-of-Vol Analytics** ✅
  - Implemented: `DispersionTrader`, `DispersionAnalysis`, `DispersionConfig`
  - Location: `src/volatility/trading/dispersion.py`
  - Tests: `tests/unit/volatility/trading/test_dispersion.py`

- [x] **Vol-of-Vol Analytics** ✅
  - Implemented: `VolOfVolAnalyzer`, `VolOfVolMetrics`
  - Location: `src/volatility/analytics/vol_of_vol.py`
  - Support: Regime detection, IV/RV analysis
  - Tests: `tests/unit/volatility/analytics/test_vol_of_vol.py`

**Implementation Complete:**
```
Core:
[x] src/volatility/__init__.py
[x] src/volatility/trading/__init__.py
[x] src/volatility/trading/variance_swap.py (VarianceSwap, VarianceSwapPricer)
[x] src/volatility/trading/dispersion.py (DispersionTrader, DispersionAnalysis)
[x] src/volatility/analytics/__init__.py
[x] src/volatility/analytics/vol_of_vol.py (VolOfVolAnalyzer, VolOfVolMetrics)

Tests:
[x] tests/unit/volatility/trading/test_variance_swap.py
[x] tests/unit/volatility/trading/test_dispersion.py
[x] tests/unit/volatility/analytics/test_vol_of_vol.py

Docs:
[x] docs/reference/volatility/vol_trading.md
[ ] tests/unit/pricers/equity/test_variance_swap.py
[ ] tests/unit/analytics/volatility/test_dispersion.py
[ ] tests/unit/analytics/volatility/test_vol_of_vol.py

Documentation:
[ ] docs/reference/instruments/variance_swaps.md
[ ] docs/guides/volatility/vol_trading_analytics.md
[ ] docs/tutorials/volatility/variance_swap_tutorial.ipynb
```

**Status:** ✅ **Complete.** Tests and reference doc in place. Pipeline/example optional; see `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` (Phase 8.1).

**Dependencies:** Vol surface calibration, Heston model, multi-asset infrastructure.

---

### 8.2 Portfolio Construction & Optimisation ✅

- [x] **Portfolio Optimisation API** ✅
  - Implemented: `MeanVarianceOptimizer`, `MVConstraints`, `MVOptimizationResult`
  - Implemented: `RiskParityOptimizer`, `RiskParityResult`
  - Implemented: `BlackLittermanModel`, `BlackLittermanResult`
  - Location: `src/portfolio/optimization/`
  - Support: Efficient frontier, target return/vol, max Sharpe, min variance
  - Tests: `tests/unit/portfolio/optimization/`
  - Docs: `docs/reference/portfolio/optimisation.md`

- [x] **Covariance Estimation** ✅
  - Implemented: `CovarianceEstimator` (sample, EWM, constant correlation)
  - Implemented: `ShrinkageEstimator` (Ledoit-Wolf)
  - Location: `src/portfolio/optimization/covariance.py`
  - Tests: `tests/unit/portfolio/optimization/test_covariance.py`

**Implementation Complete:**
```
Core Optimisation:
[x] src/portfolio/optimization/__init__.py
[x] src/portfolio/optimization/mean_variance.py (MeanVarianceOptimizer, MVConstraints)
[x] src/portfolio/optimization/risk_parity.py (RiskParityOptimizer)
[x] src/portfolio/optimization/black_litterman.py (BlackLittermanModel)

Covariance:
[x] src/portfolio/optimization/covariance.py (CovarianceEstimator, ShrinkageEstimator)

Tests:
[x] tests/unit/portfolio/optimization/test_mean_variance.py
[x] tests/unit/portfolio/optimization/test_risk_parity.py
[x] tests/unit/portfolio/optimization/test_black_litterman.py
[x] tests/unit/portfolio/optimization/test_covariance.py

Documentation:
[x] docs/reference/portfolio/optimisation.md

Pipeline:
[x] src/orchestrator/pipelines/portfolio/optimise_portfolio.py

Example:
[x] examples/pipelines/run_portfolio_optimisation.py

```

**Status:** ✅ **Complete.** Pipeline and example script implemented. See `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` (Phase 8.2).

**Dependencies:** Portfolio module, risk infrastructure.

---

### Phase 8 Deliverables:
- [x] Variance swap pricing, dispersion & vol-of-vol analytics — 8.1
- [x] Mean-variance, risk parity, Black-Litterman optimisation — 8.2
- [x] Covariance estimation with shrinkage — 8.2

**Estimated Effort:** ~2 weeks total

**Impact:** Enables volatility trading strategies and systematic portfolio construction.
- [ ] **CVaR / Expected Shortfall**
  - Implement: Conditional VaR (expected shortfall) alongside VaR
  - Support: Historical, parametric, or simulation-based
  - Use case: Tail risk application project, risk reports

- [ ] **Tail Dependence & Crisis Scenarios**
  - Implement: Tail dependence metrics, crisis-regime scenarios (e.g. correlation breakdown)
  - Support: Stress scenarios that include tail/crisis behaviour
  - Use case: Tail risk & crisis analytics application project

**Implementation Checklist (8.5):**
```
[ ] src/risk/var/cvar.py (CVaR, ExpectedShortfall, CVaRConfig)
[ ] src/risk/tail/dependence.py (TailDependence, CopulaAnalysis)
[ ] src/risk/tail/crisis_scenarios.py (CrisisScenarioGenerator, CorrelationBreakdown)
[ ] tests/unit/risk/var/test_cvar.py
[ ] tests/unit/risk/tail/test_dependence.py
[ ] docs/reference/risk/tail_risk.md
[ ] docs/guides/risk/tail_risk_management.md
[ ] docs/tutorials/risk/tail_risk_tutorial.ipynb
```

### 8.6 Real-Time Risk & Limit Monitoring
- [ ] **Limit Monitoring & Alerts**
  - Implement: Threshold checks (VaR, Greeks, exposure, PnL), breach detection, alert interface
  - Support: Integration with streaming engine and risk aggregation
  - Use case: Real-time risk dashboard application project

**Implementation Checklist (8.6):**
```
[ ] src/risk/monitoring/limits.py (LimitDefinition, LimitMonitor, LimitBreach)
[ ] src/risk/monitoring/alerts.py (AlertEngine, AlertHandler, AlertNotification)
[ ] src/risk/monitoring/dashboard.py (RiskDashboardData, RealTimeRiskAggregator)
[ ] tests/unit/risk/monitoring/test_limits.py
[ ] tests/unit/risk/monitoring/test_alerts.py
[ ] docs/reference/risk/limit_monitoring.md
[ ] docs/guides/risk/real_time_risk.md
[ ] docs/tutorials/risk/limit_monitoring_tutorial.ipynb
```

### 8.7 Alternative Data for Alpha

**What are data connectors/adapters?**
Connectors are a common data engineering pattern that transform external data sources into a standardised internal format:
```
External Source (JSON/CSV/API) → Connector → Standardised Format → ML Pipeline / Backtesting
```

**Why needed:**
- Different data providers have different formats, APIs, schemas
- Connectors isolate this complexity from core library
- Allows plugging in new data sources without changing ML/backtesting pipelines

- [ ] **Alternative Data Connectors**
  - Implement: Connectors for alternative data sources (sentiment, macro, options flow)
  - Pattern: `DataConnector` protocol with `fetch()`, `transform()`, `to_features()`
  - Examples:
    - `SentimentConnector`: Fetches news/social sentiment, normalises to [-1, +1]
    - `MacroConnector`: Fetches economic indicators (GDP, CPI), aligns timestamps
    - `OptionsFlowConnector`: Fetches unusual options activity data
  - Support: Output compatible with ML training pipeline (Phase 7.1)
  - Use case: Alternative data alpha, ML strategy features

- [ ] **Featurisation Pipeline**
  - Implement: Transform raw alternative data into ML-ready features
  - Support: Time alignment, normalisation, missing data handling
  - Use case: Feed features into generic ML training pipeline

**Implementation Checklist (8.7):**
```
Connectors (in marketdata, following existing provider patterns):
[ ] src/marketdata/providers/alternative/base.py (AltDataConnector protocol)
[ ] src/marketdata/providers/alternative/sentiment.py (SentimentConnector)
[ ] src/marketdata/providers/alternative/macro.py (MacroConnector)
[ ] src/marketdata/providers/alternative/options_flow.py (OptionsFlowConnector)

Featurisation (for ML pipeline):
[ ] src/machine_learning/data/alternative/featuriser.py (AltDataFeaturiser)
[ ] src/machine_learning/data/alternative/transforms.py (TimeAligner, Normaliser)

Tests:
[ ] tests/unit/marketdata/providers/alternative/test_connectors.py
[ ] tests/unit/machine_learning/data/alternative/test_featuriser.py

Documentation:
[ ] docs/reference/marketdata/alternative_data.md
[ ] docs/guides/data/alternative_data_pipeline.md
[ ] docs/tutorials/data/alternative_data_tutorial.ipynb
```

### 8.8 Market-Making / Quoting Analytics

**What is market-making?**
Market-making is providing liquidity by continuously quoting bid and ask prices. Options trading desks often act as market-makers.

**Core concepts:**

1. **Fair Value**: Use library pricers to compute theoretical mid-price
2. **Spread Model**: How to set bid-ask spread around fair value
   - Wider spread = more profit per trade but fewer fills
   - Narrower spread = more volume but tighter margins
   - Depends on: volatility, inventory, time of day, order flow
3. **Inventory Model**: Managing position risk from accumulated trades
   - Example: Long 100 deltas → shade asks higher (discourage more buys)
   - Avellaneda-Stoikov model is the classic academic reference

**Workflow:**
```
Market Data → Pricer (fair value) → Spread Model → Bid/Ask Quotes → Fill → Inventory Update → Risk Adjustment
```

- [ ] **Spread Models**
  - Implement: Spread rules for computing bid-ask around fair value
  - Models: Fixed spread, volatility-adjusted, Avellaneda-Stoikov
  - Support: Use pricers for fair value, market data for vol
  - Use case: Automated quoting, market-making analytics

- [ ] **Inventory Management**
  - Implement: Inventory tracking and risk adjustment
  - Support: Inventory penalty in spread calculation, position limits
  - Use case: Manage delta/gamma exposure from market-making

- [ ] **Quoting Simulator**
  - Implement: Simulate market-making P&L with fills and inventory
  - Support: Backtest quoting strategies, compare spread models
  - Use case: Strategy development, risk analysis

**Implementation Checklist (8.8):**
```
Core (in execution module alongside TCA from 8.1):
[ ] src/execution/market_making/spread.py (SpreadModel, FixedSpread, AvellanedaStoikovSpread)
[ ] src/execution/market_making/inventory.py (InventoryTracker, InventoryPenalty)
[ ] src/execution/market_making/quoting.py (QuotingEngine, Quote, QuoteGenerator)

Simulation:
[ ] src/execution/market_making/simulator.py (MarketMakingSimulator)
[ ] src/execution/market_making/backtest.py (MMBacktestEngine, MMBacktestResult)

Tests:
[ ] tests/unit/execution/market_making/test_spread.py
[ ] tests/unit/execution/market_making/test_inventory.py
[ ] tests/unit/execution/market_making/test_simulator.py

Documentation:
[ ] docs/reference/execution/market_making.md
[ ] docs/guides/execution/market_making_strategy.md
[ ] docs/tutorials/execution/market_making_tutorial.ipynb
```

**Note:** 
- Lives in `src/execution/` alongside TCA (8.1) - both are about trade execution
- Can be extended with RL quoting agent (Phase 7.2 integration) for adaptive market-making

### 8.9 XVA Framework (Credit Valuation Adjustments)

**Goal:** Implement counterparty credit risk adjustments (CVA, DVA, FVA) for OTC derivatives pricing.

- [ ] **Exposure Simulation**
  - Implement: Expected Positive Exposure (EPE), Expected Negative Exposure (ENE) profiles
  - Support: Monte Carlo simulation of portfolio value paths
  - Use case: Counterparty risk measurement, collateral optimisation

- [ ] **CVA/DVA Calculation**
  - Implement: Credit Valuation Adjustment (CVA), Debit Valuation Adjustment (DVA)
  - Formula: CVA = (1-R) ∫ EPE(t) × PD(t) dt
  - Support: Calibration to CDS spreads, netting sets
  - Use case: Counterparty credit risk pricing

- [ ] **FVA Calculation**
  - Implement: Funding Valuation Adjustment (FVA)
  - Support: Asymmetric funding costs (borrowing vs lending spreads)
  - Use case: Funding cost in derivative pricing

- [ ] **Collateral Modelling**
  - Implement: Collateral agreement (CSA) modelling, margin period of risk
  - Support: Threshold, minimum transfer amount, independent amounts
  - Use case: Collateralised exposure calculation

**Implementation Checklist (8.9):**
```
[ ] src/xva/exposure/simulator.py (ExposureSimulator, ExposurePath)
[ ] src/xva/exposure/profiles.py (EPE, ENE, PFE, ExposureProfile)
[ ] src/xva/cva/calculator.py (CVACalculator, DVACalculator)
[ ] src/xva/cva/credit_curve.py (CreditCurve, CDSBootstrapper)
[ ] src/xva/fva/calculator.py (FVACalculator, FundingCurve)
[ ] src/xva/collateral/csa.py (CSAModel, CollateralAgreement)
[ ] src/xva/collateral/margin.py (MarginCalculator, MarginPeriodOfRisk)
[ ] src/xva/netting/netting_set.py (NettingSet, CloseoutNetting)
[ ] tests/unit/xva/test_exposure.py
[ ] tests/unit/xva/test_cva.py
[ ] tests/unit/xva/test_fva.py
[ ] tests/unit/xva/test_collateral.py
[ ] docs/reference/xva/xva_framework.md
[ ] docs/guides/xva/computing_xva.md
[ ] docs/tutorials/xva/xva_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/xva/compute_xva.py
[ ] Example: examples/pipelines/run_compute_xva.py
```

**Status:** Not started.

**Dependencies:** MC simulation, pricers, credit curves.

### 8.10 Regulatory Capital (FRTB & SIMM)

**Goal:** Implement regulatory capital calculations for market risk (FRTB) and initial margin (SIMM).

- [ ] **FRTB Standardised Approach (SA)**
  - Implement: Sensitivities-Based Method (SBM) for delta, vega, curvature
  - Support: Risk class buckets (IR, FX, Equity, Commodity, Credit)
  - Support: Correlation scenarios (high, medium, low)
  - Use case: Basel IV market risk capital

- [ ] **FRTB Default Risk Charge (DRC)**
  - Implement: Default risk capital for non-securitised products
  - Support: Jump-to-default sensitivities
  - Use case: Credit default risk capital

- [ ] **ISDA SIMM**
  - Implement: Standard Initial Margin Model (SIMM 2.x)
  - Support: Delta, vega, curvature risk weights and correlations
  - Support: Concentration thresholds
  - Use case: Initial margin for non-cleared OTC derivatives

- [ ] **Capital Reports**
  - Implement: Regulatory report generation (FRTB, SIMM)
  - Support: Risk class breakdown, bucket details
  - Use case: Regulatory compliance reporting

**Implementation Checklist (8.10):**
```
FRTB SA:
[ ] src/regulatory/frtb/sensitivities.py (FRTBSensitivity, DeltaSensitivity, VegaSensitivity, CurvatureSensitivity)
[ ] src/regulatory/frtb/risk_classes.py (RiskClass, Bucket, RiskWeight)
[ ] src/regulatory/frtb/sbm.py (SBMCalculator, CorrelationScenario)
[ ] src/regulatory/frtb/drc.py (DRCCalculator, DefaultRiskCharge)
[ ] src/regulatory/frtb/aggregation.py (FRTBAggregator, TotalCapital)

SIMM:
[ ] src/regulatory/simm/sensitivities.py (SIMMSensitivity, SIMMDelta, SIMMVega)
[ ] src/regulatory/simm/risk_weights.py (SIMMRiskWeights, ConcentrationThreshold)
[ ] src/regulatory/simm/calculator.py (SIMMCalculator, SIMMResult)
[ ] src/regulatory/simm/correlation.py (SIMMCorrelation)

Reports:
[ ] src/regulatory/reports/frtb_report.py (FRTBReport, FRTBReportGenerator)
[ ] src/regulatory/reports/simm_report.py (SIMMReport, SIMMReportGenerator)

Tests:
[ ] tests/unit/regulatory/frtb/test_sbm.py
[ ] tests/unit/regulatory/frtb/test_drc.py
[ ] tests/unit/regulatory/simm/test_calculator.py

Documentation:
[ ] docs/reference/regulatory/frtb.md
[ ] docs/reference/regulatory/simm.md
[ ] docs/guides/regulatory/regulatory_capital.md
[ ] docs/tutorials/regulatory/frtb_simm_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/regulatory/compute_capital.py
[ ] Example: examples/pipelines/run_frtb_capital.py
```

**Status:** Not started.

**Dependencies:** Greeks calculation, risk sensitivities, market data.

### Deliverables (Phase 8):
- [ ] Execution models, optimal execution, TCA metrics (8.1)
- [ ] Factor exposure, factor cov/returns interface, factor attribution (8.2)
- [ ] Variance swap analytics, dispersion, vol-of-vol (8.3)
- [ ] Portfolio optimisation API, covariance input (8.4)
- [ ] CVaR, tail dependence, crisis scenarios (8.5)
- [ ] Limit monitoring and alerts (8.6)
- [ ] Alternative data adapters and featurisation (8.7)
- [ ] Spread/inventory model for market-making (8.8)
- [ ] XVA framework (CVA, DVA, FVA) (8.9)
- [ ] Regulatory capital (FRTB SA, SIMM) (8.10)

**Impact:** Library supports execution, factor risk, vol trading, portfolio optimisation, tail risk, real-time monitoring, alt data, market-making, counterparty risk (XVA), and regulatory capital application projects.

---

## Phase 9: Deployment & Services

**Goal:** Provide production-ready service layer for deploying quantitative capabilities as APIs and real-time services.

### 9.1 REST API Service (FastAPI)

- [ ] **Pricing Service**
  - Implement: FastAPI endpoints for option pricing (BSM, MC, FDE)
  - Support: Batch pricing, async processing for large portfolios
  - Endpoints: `/price`, `/greeks`, `/portfolio/price`
  - Use case: Integration with trading systems, web dashboards

- [ ] **Risk Service**
  - Implement: FastAPI endpoints for risk calculations
  - Support: VaR, Greeks aggregation, scenario analysis
  - Endpoints: `/risk/var`, `/risk/greeks`, `/risk/scenarios`
  - Use case: Risk dashboards, automated monitoring

- [ ] **Calibration Service**
  - Implement: FastAPI endpoints for model calibration
  - Support: SABR, Heston, Hull-White calibration
  - Endpoints: `/calibrate/sabr`, `/calibrate/heston`
  - Use case: Daily calibration jobs, on-demand calibration

### 9.2 Low-Latency Service (gRPC)

- [ ] **gRPC Pricing Service**
  - Implement: Protocol buffer definitions for pricing requests/responses
  - Support: Streaming pricing for real-time applications
  - Use case: High-frequency pricing, algo trading

- [ ] **gRPC Risk Service**
  - Implement: Protocol buffer definitions for risk calculations
  - Support: Streaming risk updates
  - Use case: Real-time risk monitoring

### 9.3 WebSocket Streaming

- [ ] **Quote Streaming Server**
  - Implement: WebSocket server for streaming market data and prices
  - Support: Subscription model for instruments
  - Use case: Real-time dashboards, trading UIs

- [ ] **Risk Streaming Server**
  - Implement: WebSocket server for streaming risk metrics
  - Support: Real-time Greeks, VaR updates
  - Use case: Risk monitoring dashboards

### 9.4 Service Infrastructure

- [ ] **Service Configuration**
  - Implement: ServiceConfig for deployment settings
  - Support: Environment-based configuration, secrets management
  - Use case: Production deployment

- [ ] **Health & Metrics**
  - Implement: Health check endpoints, Prometheus metrics
  - Support: Latency tracking, error rates, throughput
  - Use case: Production monitoring

- [ ] **Authentication & Authorization**
  - Implement: JWT/API key authentication
  - Support: Role-based access control
  - Use case: Secure API access

**Implementation Checklist (Phase 9):**
```
FastAPI Services (9.1):
[ ] src/services/api/pricing.py (PricingRouter, price_option, price_portfolio)
[ ] src/services/api/risk.py (RiskRouter, compute_var, compute_greeks)
[ ] src/services/api/calibration.py (CalibrationRouter, calibrate_sabr)
[ ] src/services/api/models.py (PricingRequest, PricingResponse, RiskRequest)
[ ] src/services/api/app.py (create_app, lifespan)

gRPC Services (9.2):
[ ] src/services/grpc/protos/pricing.proto
[ ] src/services/grpc/protos/risk.proto
[ ] src/services/grpc/pricing_service.py (PricingServicer)
[ ] src/services/grpc/risk_service.py (RiskServicer)
[ ] src/services/grpc/server.py (GRPCServer)

WebSocket (9.3):
[ ] src/services/websocket/quote_server.py (QuoteWebSocketServer)
[ ] src/services/websocket/risk_server.py (RiskWebSocketServer)
[ ] src/services/websocket/handlers.py (WebSocketHandler)

Infrastructure (9.4):
[ ] src/services/config.py (ServiceConfig, load_config)
[ ] src/services/health.py (HealthCheck, ReadinessCheck)
[ ] src/services/metrics.py (MetricsCollector, PrometheusExporter)
[ ] src/services/auth.py (JWTAuth, APIKeyAuth, RoleBasedAccess)

Tests:
[ ] tests/integration/services/test_pricing_api.py
[ ] tests/integration/services/test_risk_api.py
[ ] tests/integration/services/test_grpc.py
[ ] tests/integration/services/test_websocket.py

Documentation:
[ ] docs/reference/services/api_reference.md
[ ] docs/guides/deployment/deploying_services.md
[ ] docs/guides/deployment/docker_kubernetes.md
[ ] docs/tutorials/deployment/pricing_service_tutorial.ipynb

Examples:
[ ] examples/services/run_pricing_server.py
[ ] examples/services/run_risk_server.py
[ ] examples/services/client_example.py
[ ] docker/Dockerfile.pricing
[ ] docker/docker-compose.yml
```

### Deliverables (Phase 9):
- [ ] FastAPI pricing/risk/calibration services (9.1)
- [ ] gRPC low-latency services (9.2)
- [ ] WebSocket streaming servers (9.3)
- [ ] Service infrastructure (config, health, auth) (9.4)
- [ ] Docker deployment configurations
- [ ] Kubernetes manifests (optional)

**Status:** Not started.

**Dependencies:** All core library phases (1-8), especially pricers, risk, calibration.

**Impact:** Enables deployment of quantitative capabilities as production services for integration with trading systems, dashboards, and external applications.

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
- **Library dependency:** Backtesting (Phase 5.2), streaming & live data (Phase 5.5), brokerage adapter (Phase 5.5), machine_learning (Phase 7.1), q_learning (Phase 7.2).
- **Deliverable:** Trading bot application: data feed → StreamingEngine → strategy (incl. ML/RL) → order execution (paper/live) → performance reports and visualisations.

### Application Project 3: Hybrid GNN-LSTM Full Revaluation Pricer
- **Goal:** Production implementation of the Hybrid GNN-LSTM full revaluation pricer (partially built in the library).
- **Scope:** Complete and harden `src/machine_learning/models/gnn_rnn_hybrid/`; integrate with portfolio representation (trade graph, attributes); train and serve as full revaluation pricer; validate vs. library pricers.
- **Library dependency:** machine_learning (Phase 7.1), portfolio, pricers, market data.
- **Deliverable:** Trained GNN-LSTM pricer service/model that can revalue portfolios using graph + time-series representation.

### Application Project 4: Q-Learning Orchestrator Agent
- **Goal:** Q-learning (RL) orchestrator that acts as an agent for delta hedging, algorithmic trading, and other cutting-edge applications.
- **Scope:** RL agent(s) for delta hedging (e.g. minimise PnL variance vs. cost); algo trading agent (e.g. execution, strategy selection); orchestrator that runs agents against live/backtest environments using library pricers and risk.
- **Library dependency:** q_learning (Phase 7.2), backtesting, streaming engine, pricers, risk.
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
- **Library dependency:** Phase 8.8 (spread/inventory model), pricers, calibration, risk, q_learning (7.2), backtesting.
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

## Timeline Summary (Revised Scope)

| Phase | Status | Focus | Key Deliverables |
|-------|--------|-------|-----------------|
| Phase 1 | ✅ Complete | FX Enhancement | 4+ FX products, local vol, calibration |
| Phase 2 | ✅ Complete | Equity | 7+ equity products, equity infrastructure |
| Phase 3 | ✅ Complete | Rates | 6+ rate products, Hull-White, LMM |
| Phase 4 | ✅ Complete | Advanced Models | Jump-diffusion, SABR, multi-asset |
| Phase 5 | ✅ Complete | Production | Calibration, backtesting, risk, streaming, analytics |
| Phase 6 | ✅ Complete | Education | Tutorials, notebooks, docs |
| Phase 7.1 | ✅ Complete | ML Integration | ML pipelines, GNN-LSTM pricer |
| Phase 7.1.5 | ⬜ Pending | Production ML | Tracking, tuning, registry |
| Phase 7.2 | ⚠️ Core Complete | Q-Learning/RL | RL framework (environments pending) |
| Phase 7.3 | ⬜ Pending | Exotics | Cliquet, Autocallable, Range Accrual |
| Phase 7.6 | ⚠️ Core Complete | Deep Hedging | Core complete (backtesting pending) |
| Phase 7.7 | ⬜ Pending | Neural SDE | Neural drift/diffusion, training |
| Phase 8.1 | ⬜ Pending | Vol Trading | Variance swaps, dispersion |
| Phase 8.2 | ⬜ Pending | Portfolio Opt | Mean-variance, risk parity |
| **LIBRARY COMPLETE** | — | — | — |
| Applications | — | Usage Examples | Option Analytics, Algo Bot, etc. |

### Deferred/Removed Items

| Item | Status | Reason |
|------|--------|--------|
| Phase 7.8 Rough Vol | 🔲 Deferred | Research-level, complex, not essential |
| Phase 8 (old scope) | ❌ Removed | XVA, FRTB, TCA, etc. were overambitious |
| Phase 9 Services | ❌ Removed | Infrastructure work, not core quant |

### Recommended Implementation Order

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE A: Complete Core (Priority)                   ~2-3 weeks  │
├──────────────────────────────────────────────────────────────────┤
│  7.1.5  Production ML (tracking, tuning)                         │
│  7.2    RL Environments (trading, hedging)                       │
│  7.3    Exotic Products (Cliquet, Autocallable, Range)           │
│  7.6    Deep Hedging backtesting adapter                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE B: Focused Extensions                         ~2 weeks    │
├──────────────────────────────────────────────────────────────────┤
│  7.7    Neural SDE (optional, research interest)                 │
│  8.1    Vol Trading (variance swaps, dispersion)                 │
│  8.2    Portfolio Optimisation (mean-variance, risk parity)      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  LIBRARY COMPLETE - DECLARE VICTORY                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  APPLICATION PROJECTS (as needed)                                │
├──────────────────────────────────────────────────────────────────┤
│  Option Analytics Dashboard                                      │
│  Algorithmic Trading Bot                                         │
│  Vol Trading Tool                                                │
│  Portfolio Optimiser                                             │
└──────────────────────────────────────────────────────────────────┘
```

**Total Timeline:** ~4-5 weeks for library completion. Application projects built as needed.

---

## Risk Mitigation

### Scope Management (Key Learning)
- **Original roadmap was overambitious** - Phases 8-9 alone would have doubled the codebase
- **Revised scope is achievable** - Focused on high-value additions
- **Application projects are separate** - Don't add to library maintenance burden

### Technical Risks
- **Complexity Creep**: Maintain interface contracts, avoid over-engineering
- **Performance**: Profile early, optimize bottlenecks
- **Testing**: Maintain high test coverage, parity tests prevent regressions

### Quality Standards
- **Every component needs:** Tests, docs, guides, examples
- **No half-finished features** - Complete or defer
- **Maintenance burden considered** - Only add what can be maintained

---

## Conclusion

### What We've Built

QuantStrata is a **comprehensive, professional quant library** with:

| Dimension | Achievement |
|-----------|-------------|
| **Asset Classes** | FX, Equity, IR, Multi-Asset |
| **Pricing Models** | 15+ (BSM, Heston, Hull-White, LMM, etc.) |
| **Numerical Methods** | Analytic, MC, FDE, LSM, QMC |
| **ML/RL** | GNN-LSTM, Deep Hedging, Q-Learning |
| **Infrastructure** | Calibration, Backtesting, Streaming, Risk |
| **Code Quality** | ~87K LOC, ~40K test LOC, clean architecture |

### What Remains (Optional / Gaps)

| Item | Status | Notes |
|------|--------|-------|
| 7.1.5 Production ML | ✅ Done | Gaps: guide docs (experiment_tracking, hyperparameter_tuning), tutorial notebook |
| 7.2 RL Environments & Runners | ✅ Done | Gaps: orchestrator rl pipelines, extra reference/guide docs, tutorial, example script |
| 7.3 Exotic Products | ✅ Done | Gaps: reference/guide docs, pipeline, example script (see IMPLEMENTATION_CHECKLIST_REVIEW) |
| 7.6 Deep Hedging Backtesting | ✅ Done | Gaps: guide docs, pipeline, example script (optional) |
| 7.7 Neural SDE | ✅ Core done | Gaps: adjoint; score matching/calibration; MC pricer/Greeks; conditional/augmentation; docs; pipeline; example |
| 8.1 Vol Trading | ✅ Done | Gaps: optional extra docs/tutorials |
| 8.2 Portfolio Opt | ✅ Done | — |

### Library Completion Criteria

The library is considered **complete** for core scope when:
1. ✅ Phases 1-6 (done)
2. ✅ Phase 7.1 ML Integration (done)
3. ✅ Phase 7.1.5, 7.2, 7.3, 7.6, 7.7 (implementation and tests done; some docs/pipelines/examples deferred)
4. ✅ Phase 8.1, 8.2 (focused extensions done)

**Remaining work** is optional: tutorial notebooks, additional guide docs, orchestrator pipelines for RL/deep hedging/neural SDE, and example scripts where not yet present. See phase implementation checklists and `docs/development/IMPLEMENTATION_CHECKLIST_REVIEW.md` for details.

After that: **Build application projects as needed, not as library features.**

---

**The foundation is excellent. The scope is now manageable. Time to finish!**
