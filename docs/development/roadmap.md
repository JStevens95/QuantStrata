# QuantStrata Development Roadmap

**Last Updated:** January 27, 2026 (Phase 7.2 Core Complete; added Phases 7.6-7.8, 8.9-8.10, Phase 9)  
**Current Version:** V1 (FX Derivatives Foundation)  
**Target:** Comprehensive Professional Quant Library

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

### 7.1.5 Production ML Infrastructure

**Goal:** Add production-grade ML tooling for experiment tracking, hyperparameter tuning, and model versioning. Note: `machine_learning/` is already production-quality; these additions extend existing capabilities.

- [ ] **Experiment Tracking Integration**
  - Implement: `MLflowTracker`, `WandBTracker` with common `TrackingProtocol`
  - Location: `src/machine_learning/core/tracking.py` (extends core utilities)
  - Support: Log metrics, parameters, artifacts; integration with training pipelines
  - Use case: Track experiments, compare runs, reproduce results

- [ ] **Hyperparameter Tuning Extensions**
  - Implement: `SearchSpace`, `TrialPruner`, `TuningResult` (extends existing `pipelines/tuning.py`)
  - Location: `src/machine_learning/tuning/` (new submodule if depth needed)
  - Support: Bayesian optimisation (Optuna), pruning, parallel trials
  - Integration: Works with existing training pipelines
  - Use case: Automated hyperparameter search for ML pricers and calibration
  - Note: `pipelines/tuning.py` already exists; extend if deeper utilities needed

- [ ] **Model Registry & Versioning**
  - Implement: `ModelRegistry`, `ModelArtifact`, `ModelVersion`
  - Location: `src/machine_learning/registry/` or `src/machine_learning/core/registry.py`
  - Support: Version tracking, metadata, promotion (staging → production)
  - Use case: Production model management and deployment

**Implementation Checklist:**
```
[ ] src/machine_learning/core/tracking.py (MLflowTracker, WandBTracker, TrackingProtocol)
[ ] src/machine_learning/tuning/search_space.py (SearchSpace, TrialPruner) — if pipelines/tuning.py insufficient
[ ] src/machine_learning/registry/registry.py (ModelRegistry, ModelArtifact, ModelVersion)
[ ] tests/unit/machine_learning/core/test_tracking.py
[ ] tests/unit/machine_learning/tuning/test_search_space.py
[ ] tests/unit/machine_learning/registry/test_registry.py
[ ] docs/reference/machine_learning/production_ml.md
[ ] docs/guides/machine_learning/experiment_tracking.md
[ ] docs/guides/machine_learning/hyperparameter_tuning.md
[ ] docs/tutorials/machine_learning/ml_production.ipynb
[ ] Pipeline: extends src/machine_learning/pipelines/tuning.py
[ ] Example: examples/pipelines/run_tune_gnn_pricer.py
```

**Status:** Not started. Dependencies: Phase 7.1 complete.

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

- [ ] **RL Agent Deployment & Environments**
  - Implement: Trading and hedging environments that wrap backtesting/streaming infrastructure
  - Implement: Agent runners that execute trained agents in backtest or live contexts
  - Integrate: With backtesting engine, streaming engine, and library pricers/risk
  - Use case: Automated hedging, strategy deployment, cutting-edge applications
  - Note: Orchestrator pipelines go in `src/orchestrator/pipelines/rl/`; q_learning module provides environments and runners

**RL Deployment Implementation Checklist:**
```
Environments (extend existing q_learning/environments/):
[ ] src/q_learning/environments/trading.py (TradingEnvironment wrapping backtesting)
[ ] src/q_learning/environments/hedging.py (HedgingEnvironment wrapping pricers)
[ ] src/q_learning/environments/streaming.py (StreamingEnvironment for live execution)

Runners (agent execution utilities):
[ ] src/q_learning/runners/backtest.py (BacktestRunner - run agent in backtesting framework)
[ ] src/q_learning/runners/live.py (LiveRunner - run agent with streaming engine)
[ ] src/q_learning/runners/base.py (BaseRunner protocol)

Orchestrator Pipelines (in src/orchestrator/pipelines/):
[ ] src/orchestrator/pipelines/rl/deploy_agent.py (orchestrator-level deployment)
[ ] src/orchestrator/pipelines/rl/backtest_agent.py (orchestrator-level backtesting)

Tests:
[ ] tests/unit/q_learning/environments/test_trading.py
[ ] tests/unit/q_learning/environments/test_hedging.py
[ ] tests/unit/q_learning/runners/test_backtest.py

Documentation:
[ ] docs/reference/q_learning/environments.md
[ ] docs/reference/q_learning/runners.md
[ ] docs/guides/q_learning/deploying_rl_agents.md
[ ] docs/tutorials/q_learning/rl_deployment_tutorial.ipynb
[ ] Example: examples/pipelines/run_deploy_rl_agent.py
```

**Status:** Phase 7.2 core complete (training, evaluation, inference, protocols, BaseEnv, metrics). RL Orchestrator not yet implemented. See `docs/development/progress/phase_7_2_q_learning.md`. Technical reference: `docs/reference/q_learning/rl_framework.md`. Guide: `docs/guides/q_learning/rl_framework.md`.

### 7.3 Exotic Products

**Goal:** Implement structured products commonly traded by hedge funds and investment banks.

- [ ] **Cliquet Options**
  - Instrument: `CliquetOption` (periodic resets with local/global caps and floors)
  - Parameters: reset_dates, local_cap, local_floor, global_cap, global_floor, participation
  - Payoff: `CliquetPayoff` (path-dependent, sum of capped/floored periodic returns)
  - Pricer: `CliquetMcPricer` (MC required for path-dependency)
  - Greeks: Delta, gamma, vega (bump-and-reval), rho
  - Use case: Equity-linked structured notes, guaranteed return products

- [ ] **Autocallable Products**
  - Instrument: `AutocallableOption` (barrier observation dates, coupon, early redemption)
  - Parameters: observation_dates, autocall_barrier, coupon_barrier, put_barrier, coupon_rate
  - Payoff: `AutocallablePayoff` (early termination on barrier breach with coupon)
  - Pricer: `AutocallableMcPricer` (MC required)
  - Greeks: Delta, gamma, vega, autocall probability
  - Use case: Most popular structured product globally (>$100B annual issuance)

- [ ] **Range Accrual**
  - Instrument: `RangeAccrualNote` (IR or FX underlying)
  - Parameters: range_lower, range_upper, observation_freq, notional, accrual_rate
  - Payoff: `RangeAccrualPayoff` (accrues on days within range)
  - Pricer: `RangeAccrualMcPricer` (MC required)
  - Greeks: Delta, range sensitivity
  - Use case: Yield enhancement, low-vol betting

**Implementation Checklist:**

**Pricer Naming Convention:** `{product}_{model}_{method}.py`
- Follows existing pattern: `european_bsm_mc.py`, `european_heston_mc.py`
- For exotics: `cliquet_gbm_mc.py`, `autocallable_gbm_mc.py` (allows future model variants)

```
Cliquet Options:
[ ] src/instruments/equity/options/cliquet.py (EquityCliquetOption)
[ ] src/instruments/fx/options/cliquet.py (FxCliquetOption)
[ ] src/models/payoffs/cliquet.py (CliquetPayoff)
[ ] src/pricers/equity/cliquet_gbm_mc.py (EquityCliquetGbmMcPricer)
[ ] src/pricers/fx/cliquet_gbm_mc.py (FxCliquetGbmMcPricer)
[ ] tests/unit/instruments/equity/test_cliquet.py
[ ] tests/unit/pricers/equity/test_cliquet_gbm_mc.py

Autocallable Products:
[ ] src/instruments/equity/options/autocallable.py (EquityAutocallableOption)
[ ] src/models/payoffs/autocallable.py (AutocallablePayoff)
[ ] src/pricers/equity/autocallable_gbm_mc.py (EquityAutocallableGbmMcPricer)
[ ] src/pricers/equity/autocallable_localvol_mc.py (EquityAutocallableLocalvolMcPricer) — optional
[ ] tests/unit/instruments/equity/test_autocallable.py
[ ] tests/unit/pricers/equity/test_autocallable_gbm_mc.py

Range Accrual:
[ ] src/instruments/ir/options/range_accrual.py (IrRangeAccrualNote)
[ ] src/models/payoffs/range_accrual.py (RangeAccrualPayoff)
[ ] src/pricers/ir/range_accrual_hw_mc.py (IrRangeAccrualHwMcPricer) — Hull-White model
[ ] tests/unit/instruments/ir/test_range_accrual.py
[ ] tests/unit/pricers/ir/test_range_accrual_hw_mc.py

Documentation:
[ ] docs/reference/instruments/exotic_products.md
[ ] docs/guides/instruments/pricing_exotics.md
[ ] docs/tutorials/pricing/exotic_options.ipynb
```

**Status:** Not started.

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

- [ ] **Backtesting Integration**
  - Implement: Run trained hedging agent in backtesting framework
  - Support: Historical data replay, out-of-sample evaluation
  - Use case: Validate deep hedging on real data
  - Note: Uses existing `src/backtesting/` infrastructure; deep_hedging provides adapters

**Backtesting Integration Implementation Checklist:**
```
Adapters (bridge deep_hedging agents to backtesting framework):
[ ] src/deep_hedging/adapters/backtesting.py (BacktestEngineAdapter - adapts hedging agent to backtesting.Strategy interface)
[ ] src/deep_hedging/adapters/historical_data.py (HistoricalDataAdapter - prepares historical data for hedging env)

Results (hedging-specific result processing):
[ ] src/deep_hedging/evaluation/backtest_metrics.py (HedgingBacktestMetrics - extends evaluation with backtest-specific metrics)

Tests:
[ ] tests/unit/deep_hedging/adapters/test_backtesting.py
[ ] tests/unit/deep_hedging/evaluation/test_backtest_metrics.py

Documentation:
[ ] docs/guides/deep_hedging/backtesting_hedging_agents.md
[ ] Update: docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb (add backtesting section)

Pipeline:
[ ] src/orchestrator/pipelines/deep_hedging/backtest_agent.py
[ ] Example: examples/pipelines/run_backtest_hedging_agent.py
```

**Note on Deep Hedging Architecture:**
Deep hedging is fundamentally an RL application (agents, environments, training). It's kept as a separate module because:
1. It's a recognised research field with specific terminology (Bühler et al. "Deep Hedging")
2. Contains domain-specific components: transaction costs, risk measures (CVaR, entropic), hedging evaluation metrics
3. Users searching for "deep hedging" expect a dedicated module
The structure mirrors `q_learning/` but with hedging-specific components.

### 7.6.4 Advanced Deep Hedging
- [ ] **Multi-Asset Hedging**
  - Implement: Hedge portfolio of options with multiple underlyings
  - Support: Cross-gamma, correlation hedging
  - Use case: Portfolio-level deep hedging

- [ ] **Model-Agnostic Hedging**
  - Implement: Train hedging agent without assuming specific dynamics
  - Support: Learn from historical data directly (uses historical data adapter from 7.6.3)
  - Use case: Robust hedging under model uncertainty

**Implementation Checklist:**
```
Multi-Asset (extends existing environments/agents):
[ ] src/deep_hedging/environments/multi_asset.py (MultiAssetHedgingEnv)
[ ] src/deep_hedging/agents/multi_asset.py (MultiAssetDeepHedgingAgent)
[ ] tests/unit/deep_hedging/environments/test_multi_asset.py
[ ] tests/unit/deep_hedging/agents/test_multi_asset.py

Model-Agnostic (uses historical adapter):
[ ] src/deep_hedging/environments/historical.py (HistoricalHedgingEnv - wraps historical data)
[ ] tests/unit/deep_hedging/environments/test_historical.py

Documentation:
[ ] docs/guides/deep_hedging/multi_asset_hedging.md
[ ] docs/guides/deep_hedging/model_agnostic_hedging.md
```

**Status:** ✅ **Core complete.** Environments, agents, training, evaluation implemented. Advanced features (multi-asset, backtesting integration, model-agnostic) pending. See `docs/development/progress/phase_7_6_deep_hedging.md`.

**Documentation:**
- Theory: `docs/reference/deep_hedging/theory.md` (PhD-level technical reference)
- Tutorial: `docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb`

**Dependencies:** Phase 7.2 (RL framework), pricers (Greeks), MC simulation, backtesting.

---

## Phase 7.7: Neural SDE & Generative Market Simulation

**Goal:** Implement neural stochastic differential equations that learn drift and diffusion functions from data, enabling more realistic market simulation than parametric models (GBM, Heston).

**Research Foundation:** Kidger et al. (2021) "Neural SDEs", Gierjatowicz et al. (2020) "Robust pricing and hedging via neural SDEs"

### 7.7.1 Neural SDE Framework
- [ ] **Neural Drift and Diffusion**
  - Implement: Neural networks μ_θ(S, t) and σ_θ(S, t) for SDE: dS = μ_θ dt + σ_θ dW
  - Architecture: MLP with positivity constraints for diffusion
  - Support: Time-dependent and state-dependent dynamics
  - Use case: Learn realistic price dynamics from data

- [ ] **SDE Solver Integration**
  - Implement: Euler-Maruyama and Milstein solvers for neural SDEs
  - Support: Batched simulation for efficiency
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

- [ ] **Training Pipeline**
  - Implement: Data pipeline (historical returns → training batches)
  - Support: Validation, early stopping, checkpointing
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
[ ] src/models/neural_sde/networks.py (NeuralDriftNetwork, NeuralDiffusionNetwork)
[ ] src/models/neural_sde/solvers.py (EulerMaruyamaSolver, MilsteinSolver)
[ ] src/models/neural_sde/adjoint.py (AdjointSDEMethod)
[ ] src/models/neural_sde/dynamics.py (NeuralSDEDynamics)
[ ] tests/unit/models/neural_sde/test_networks.py
[ ] tests/unit/models/neural_sde/test_solvers.py

Training (7.7.2):
[ ] src/models/neural_sde/training/score_matching.py (ScoreMatchingTrainer)
[ ] src/models/neural_sde/training/calibration.py (NeuralSDECalibrator)
[ ] src/models/neural_sde/training/pipeline.py (NeuralSDETrainingPipeline)
[ ] tests/unit/models/neural_sde/training/test_score_matching.py

Pricing Integration (7.7.3):
[ ] src/pricers/neural_sde/mc_pricer.py (NeuralSDEMcPricer)
[ ] src/pricers/neural_sde/greeks.py (NeuralSDEGreeksCalculator)
[ ] tests/unit/pricers/neural_sde/test_mc_pricer.py

Generative Simulation (7.7.4):
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

**Status:** Not started. See `docs/development/progress/phase_7_7_neural_sde.md` (to be created).

**Dependencies:** MC framework, calibration engine, ML pipelines (7.1).

---

## Phase 7.8: Rough Volatility Models

**Goal:** Implement rough volatility models (rough Bergomi, rough Heston) that capture the empirically observed roughness of volatility paths (Hurst parameter H ≈ 0.1), providing better fit to short-dated ATM skew and vol term structure.

**Research Foundation:** Gatheral, Jaisson & Rosenbaum (2018) "Volatility is Rough", Bayer, Friz & Gatheral (2016) "Pricing under Rough Volatility"

### 7.8.1 Fractional Brownian Motion
- [ ] **Fractional BM Sampler**
  - Implement: Efficient simulation of fractional Brownian motion with H ∈ (0, 1)
  - Methods: Cholesky decomposition, Hosking method, hybrid schemes
  - Support: Configurable Hurst parameter, batch generation
  - Use case: Foundation for rough vol models

- [ ] **Volterra Process**
  - Implement: Volterra-type integral for rough volatility: V_t = ∫ K(t-s) dW_s
  - Support: Power-law kernel K(t) = t^{H-1/2}
  - Use case: Rough Bergomi variance process

### 7.8.2 Rough Bergomi Model
- [ ] **Rough Bergomi Dynamics**
  - Implement: dS/S = √V dW, V_t = ξ(t) exp(η W^H_t - η²t^{2H}/2)
  - Parameters: Forward variance curve ξ(t), vol-of-vol η, Hurst H
  - Support: Correlation between spot and vol
  - Use case: State-of-the-art rough vol model

- [ ] **Rough Bergomi MC Pricer**
  - Implement: Monte Carlo pricer with rough Bergomi paths
  - Support: European, barrier, and path-dependent options
  - Use case: Price under rough vol dynamics

- [ ] **Hybrid Simulation Scheme**
  - Implement: Efficient simulation combining exact and approximate methods
  - Support: Turbocharging (variance reduction via conditioning)
  - Use case: Fast rough Bergomi simulation

### 7.8.3 Rough Heston Model
- [ ] **Rough Heston Dynamics**
  - Implement: Heston with fractional kernel: V_t = V_0 + ∫ K(t-s)(θ - V_s) ds + ∫ K(t-s) ν√V_s dW_s
  - Parameters: Mean reversion θ, vol-of-vol ν, Hurst H
  - Support: Affine structure for semi-analytic pricing
  - Use case: Rough vol with mean reversion

- [ ] **Characteristic Function (Adams Method)**
  - Implement: Solve fractional Riccati equation for characteristic function
  - Support: Fourier pricing for European options
  - Use case: Fast rough Heston pricing

### 7.8.4 Calibration
- [ ] **Rough Vol Calibration**
  - Implement: Calibrate rough Bergomi / rough Heston to vol surface
  - Support: Fit to ATM skew term structure, smile
  - Use case: Market-consistent rough vol parameters

- [ ] **Hurst Parameter Estimation**
  - Implement: Estimate H from realised volatility time series
  - Methods: Variogram, R/S analysis, wavelets
  - Use case: Empirical validation, model selection

### 7.8.5 Integration
- [ ] **Vol Surface Infrastructure**
  - Implement: Extend vol surface to support rough vol model outputs
  - Support: Rough vol implied vol computation
  - Use case: Consistent vol surface representation

- [ ] **Greeks under Rough Vol**
  - Implement: Delta, gamma, vega under rough vol (via bump-and-reval or AD)
  - Support: Path-wise and likelihood ratio methods
  - Use case: Risk management with rough vol

**Implementation Checklist:**
```
Fractional Brownian Motion (7.8.1):
[ ] src/models/rough_volatility/fbm/sampler.py (FractionalBMSampler)
[ ] src/models/rough_volatility/fbm/cholesky.py (CholeskyFBM)
[ ] src/models/rough_volatility/fbm/hosking.py (HoskingFBM)
[ ] src/models/rough_volatility/volterra.py (VolterraProcess)
[ ] tests/unit/models/rough_volatility/test_fbm.py

Rough Bergomi (7.8.2):
[ ] src/models/rough_volatility/rough_bergomi/dynamics.py (RoughBergomiDynamics)
[ ] src/models/rough_volatility/rough_bergomi/simulator.py (RoughBergomiSimulator)
[ ] src/models/rough_volatility/rough_bergomi/hybrid_scheme.py (HybridSimulationScheme)
[ ] src/pricers/rough_volatility/rough_bergomi_mc.py (RoughBergomiMcPricer)
[ ] tests/unit/models/rough_volatility/test_rough_bergomi.py
[ ] tests/unit/pricers/rough_volatility/test_rough_bergomi_mc.py

Rough Heston (7.8.3):
[ ] src/models/rough_volatility/rough_heston/dynamics.py (RoughHestonDynamics)
[ ] src/models/rough_volatility/rough_heston/char_func.py (RoughHestonCharFunc, AdamsMethod)
[ ] src/models/rough_volatility/rough_heston/fourier_pricer.py (RoughHestonFourierPricer)
[ ] tests/unit/models/rough_volatility/test_rough_heston.py

Calibration (7.8.4):
[ ] src/calibration/rough_volatility/rough_bergomi.py (RoughBergomiCalibrator)
[ ] src/calibration/rough_volatility/rough_heston.py (RoughHestonCalibrator)
[ ] src/calibration/rough_volatility/hurst_estimation.py (HurstEstimator)
[ ] tests/unit/calibration/rough_volatility/test_calibrators.py
[ ] tests/unit/calibration/rough_volatility/test_hurst.py

Integration (7.8.5):
[ ] src/models/rough_volatility/vol_surface.py (RoughVolSurface)
[ ] src/models/rough_volatility/greeks.py (RoughVolGreeks)
[ ] tests/unit/models/rough_volatility/test_integration.py

Documentation:
[ ] docs/reference/models/rough_volatility.md
[ ] docs/guides/models/rough_vol_calibration.md
[ ] docs/tutorials/models/rough_volatility_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/calibration/rough_vol.py
[ ] Example: examples/pipelines/run_calibrate_rough_bergomi.py
```

**Status:** Not started. See `docs/development/progress/phase_7_8_rough_volatility.md` (to be created).

**Dependencies:** MC framework, vol surface infrastructure, calibration engine.

---

### Phase 7 Extended Deliverables:
- ✅ ML integration (pricing, calibration) — 7.1
- ✅ Hybrid GNN-LSTM full revaluation pricer — 7.1
- ✅ Q-learning / RL agent framework — 7.2
- ✅ Deep hedging framework with transaction costs — 7.6
- ✅ Neural SDE for learned market dynamics — 7.7
- ✅ Rough volatility models (rough Bergomi, rough Heston) — 7.8
- ✅ 3+ exotic products — 7.3
- ✅ Optional: Credit/commodities — 7.4/7.5

**Impact:** Demonstrates **cutting-edge research-level** capabilities in modern quantitative finance: deep hedging, neural SDEs, and rough volatility.

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

**Implementation Checklist (8.1):**
```
[ ] src/execution/costs/market_impact.py (TemporaryImpact, PermanentImpact, AlmgrenChrissImpact)
[ ] src/execution/costs/spread.py (SpreadModel, LiquidityAdjustedSpread)
[ ] src/execution/optimal/almgren_chriss.py (AlmgrenChrissSolver, OptimalTrajectory)
[ ] src/execution/optimal/twap_vwap.py (TWAPSchedule, VWAPSchedule)
[ ] src/execution/tca/metrics.py (ImplementationShortfall, ArrivalPrice, Slippage)
[ ] src/execution/tca/report.py (TCAReport, TCAReportGenerator)
[ ] tests/unit/execution/test_market_impact.py
[ ] tests/unit/execution/test_optimal_execution.py
[ ] tests/unit/execution/test_tca_metrics.py
[ ] docs/reference/execution/tca.md
[ ] docs/guides/execution/optimal_execution.md
[ ] docs/tutorials/execution/tca_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/execution/compute_tca.py
[ ] Example: examples/pipelines/run_tca_analysis.py
```

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

**Implementation Checklist (8.2):**
```
[ ] src/risk/factor/exposure.py (FactorExposureCalculator, FactorDefinition)
[ ] src/risk/factor/covariance.py (FactorCovarianceMatrix, FactorReturns)
[ ] src/risk/factor/attribution.py (FactorPnLAttribution, FactorAttributionReport)
[ ] src/risk/factor/factor_var.py (FactorVaR)
[ ] tests/unit/risk/factor/test_exposure.py
[ ] tests/unit/risk/factor/test_attribution.py
[ ] docs/reference/risk/factor_model.md
[ ] docs/guides/risk/factor_risk.md
[ ] docs/tutorials/risk/factor_attribution_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/risk/compute_factor_risk.py
[ ] Example: examples/pipelines/run_factor_attribution.py
```

### 8.3 Volatility Trading & Variance Swap Analytics
- [ ] **Variance Swap / Vol Swap Pricing**
  - Implement: Variance swap pricing (e.g. model-based from Heston/local vol), fair variance strike
  - Support: Integration with existing vol models
  - Use case: Volatility trading application project

- [ ] **Dispersion & Vol-of-Vol Analytics**
  - Implement: Index vs single-name dispersion metrics, vol-of-vol from existing models
  - Support: Relative value and dispersion trading analytics
  - Use case: Volatility trading application project

**Implementation Checklist (8.3):**
```
[ ] src/instruments/equity/options/variance_swap.py (VarianceSwap, VolatilitySwap)
[ ] src/pricers/equity/variance_swap.py (VarianceSwapPricer, VarianceSwapReplicator)
[ ] src/analytics/volatility/dispersion.py (DispersionAnalytics, IndexVsSingleName)
[ ] src/analytics/volatility/vol_of_vol.py (VolOfVolCalculator)
[ ] tests/unit/instruments/equity/test_variance_swap.py
[ ] tests/unit/analytics/volatility/test_dispersion.py
[ ] docs/reference/instruments/variance_swaps.md
[ ] docs/guides/volatility/vol_trading_analytics.md
[ ] docs/tutorials/volatility/variance_swap_tutorial.ipynb
```

### 8.4 Portfolio Construction & Optimisation
- [ ] **Portfolio Optimisation API**
  - Implement: Mean-variance optimisation, risk parity, max Sharpe / min variance
  - Support: Constraints (turnover, sector, leverage, bounds); optional Black-Litterman
  - Use case: Portfolio optimisation application project, algo bot rebalance

- [ ] **Covariance / Risk Input for Optimisation**
  - Implement: Portfolio covariance from library (e.g. Greeks + factor cov, or sample)
  - Support: Same risk inputs as VaR and factor model where applicable
  - Use case: Portfolio optimisation, factor-aware optimisation

**Implementation Checklist (8.4):**
```
[ ] src/portfolio/optimisation/mean_variance.py (MeanVarianceOptimiser)
[ ] src/portfolio/optimisation/risk_parity.py (RiskParityOptimiser)
[ ] src/portfolio/optimisation/black_litterman.py (BlackLittermanOptimiser)
[ ] src/portfolio/optimisation/constraints.py (TurnoverConstraint, SectorConstraint, LeverageConstraint)
[ ] src/portfolio/optimisation/covariance.py (CovarianceEstimator, ShrinkageEstimator)
[ ] tests/unit/portfolio/optimisation/test_mean_variance.py
[ ] tests/unit/portfolio/optimisation/test_risk_parity.py
[ ] docs/reference/portfolio/optimisation.md
[ ] docs/guides/portfolio/portfolio_construction.md
[ ] docs/tutorials/portfolio/portfolio_optimisation_tutorial.ipynb
[ ] Pipeline: src/orchestrator/pipelines/portfolio/optimise_portfolio.py
[ ] Example: examples/pipelines/run_portfolio_optimisation.py
```

### 8.5 Tail Risk & Crisis Analytics
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

## Timeline Summary

| Phase | Status | Focus | Key Deliverables |
|-------|--------|-------|-----------------|
| Phase 1 | ✅ Complete | FX Enhancement | 4+ FX products, local vol, calibration |
| Phase 2 | ✅ Complete | Equity | 7+ equity products, equity infrastructure |
| Phase 3 | ✅ Complete | Rates | 6+ rate products, Hull-White, LMM |
| Phase 4 | ✅ Complete | Advanced Models | Jump-diffusion, SABR, multi-asset |
| Phase 5 | ✅ Complete | Production | Calibration, backtesting, risk, streaming, analytics |
| Phase 6 | ✅ Complete | Education | Tutorials, notebooks, docs |
| Phase 7.1 | ✅ Complete | ML Integration | ML pipelines, GNN-LSTM pricer |
| Phase 7.1.5 | ⬜ Pending | Production ML | MLflow, Optuna, Model Registry |
| Phase 7.2 | ⚠️ Core Complete | Q-Learning/RL | RL framework, RL Orchestrator pending |
| Phase 7.3 | ⬜ Pending | Exotics | Cliquet, Autocallable, Range Accrual |
| Phase 7.4-7.5 | ⬜ Optional | Extensions | Credit derivatives, Commodities |
| Phase 7.6 | ⚠️ Core Complete | Deep Hedging | Environments, agents, backtesting pending |
| Phase 7.7 | ⬜ Pending | Neural SDE | Neural drift/diffusion, SDE training |
| Phase 7.8 | ⬜ Pending | Rough Volatility | Fractional BM, rough Bergomi/Heston |
| Phase 8.1 | ⬜ Pending | Execution/TCA | Market impact, optimal execution |
| Phase 8.2 | ⬜ Pending | Factor Risk | Factor exposure, factor attribution |
| Phase 8.3 | ⬜ Pending | Vol Trading | Variance swaps, dispersion |
| Phase 8.4 | ⬜ Pending | Portfolio Opt | Mean-variance, risk parity |
| Phase 8.5 | ⬜ Pending | Tail Risk | CVaR, crisis scenarios |
| Phase 8.6 | ⬜ Pending | Limit Monitoring | Real-time risk, alerts |
| Phase 8.7 | ⬜ Pending | Alt Data | Data adapters, featurisation |
| Phase 8.8 | ⬜ Pending | Market-Making | Spread/inventory models |
| Phase 8.9 | ⬜ Pending | XVA | CVA, DVA, FVA |
| Phase 8.10 | ⬜ Pending | Regulatory | FRTB SA, SIMM |
| Phase 9 | ⬜ Pending | Services | FastAPI, gRPC, WebSocket |
| *After library* | — | Applications | Projects 1–12 |

### Recommended Implementation Order

```
┌──────────────────────────────────────────────────────────────────┐
│  IMMEDIATE: Complete Core Gaps                                   │
├──────────────────────────────────────────────────────────────────┤
│  7.2   RL Orchestrator (completes RL framework)                  │
│  7.3   Exotic Products (Cliquet, Autocallable, Range Accrual)    │
│  7.6   Deep Hedging Backtesting Integration                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  SHORT-TERM: Production ML & Research Models                     │
├──────────────────────────────────────────────────────────────────┤
│  7.1.5 Production ML (MLflow, Optuna, Registry)                  │
│  7.7   Neural SDE                                                │
│  7.8   Rough Volatility                                          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  MEDIUM-TERM: Quant HF Extensions                                │
├──────────────────────────────────────────────────────────────────┤
│  8.1   Execution & TCA                                           │
│  8.2   Factor Risk                                               │
│  8.4   Portfolio Optimisation                                    │
│  8.5   Tail Risk (CVaR)                                          │
│  8.9   XVA Framework                                             │
│  8.10  Regulatory Capital (FRTB/SIMM)                            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  LONGER-TERM: Services & Remaining Extensions                    │
├──────────────────────────────────────────────────────────────────┤
│  8.3   Vol Trading Analytics                                     │
│  8.6   Real-Time Limit Monitoring                                │
│  8.7   Alternative Data                                          │
│  8.8   Market-Making Simulator                                   │
│  9     Deployment & Services                                     │
└──────────────────────────────────────────────────────────────────┘
```

**Total Timeline:** Library core (Phases 1-7) substantially complete; remaining items add ~6-8 months for Phases 7.x completion, 8.x extensions, and Phase 9 services. Application projects follow library completion.

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

### Current Status

The library has achieved **substantial completion** of Phases 1-7 core functionality:
- ✅ FX, Equity, IR derivatives with multiple pricing methods
- ✅ Advanced models (Heston, SABR, Hull-White, LMM, Merton, VG)
- ✅ Production infrastructure (calibration, backtesting, streaming, risk)
- ✅ ML integration with GNN-LSTM pricer
- ✅ Deep hedging framework (core complete)

### Immediate Next Steps

1. **Complete Core Gaps:**
   - 7.2: RL Orchestrator (deploy RL agents to backtest/live)
   - 7.3: Exotic Products (Cliquet, Autocallable, Range Accrual)
   - 7.6: Deep Hedging backtesting integration

2. **Production ML Enhancement (7.1.5):**
   - Experiment tracking (MLflow/W&B)
   - Hyperparameter tuning (Optuna)
   - Model registry and versioning

3. **Research-Level Models:**
   - 7.7: Neural SDE (learned market dynamics)
   - 7.8: Rough Volatility (rough Bergomi, rough Heston)

### Medium-Term Goals

4. **Quant HF Extensions (Phase 8):**
   - Execution/TCA, Factor Risk, Portfolio Optimisation
   - XVA Framework (8.9), Regulatory Capital (8.10)
   - Vol Trading, Market-Making, Alternative Data

5. **Deployment & Services (Phase 9):**
   - FastAPI pricing/risk services
   - gRPC low-latency services
   - WebSocket streaming

### Long-Term Vision

6. **Application Projects 1-12:**
   - Option Analytics, Algo Bot, GNN-LSTM Pricer
   - Q-Learning Orchestrator, Execution/TCA
   - Factor Risk, Market-Making, Vol Trading
   - Portfolio Optimisation, Tail Risk, Real-Time Risk
   - Alternative Data Alpha

---

**Implementation Standard:** For every new component, deliver:
- Implementation with type hints and docstrings
- Unit tests (>90% coverage)
- Reference documentation
- Guide documentation
- Tutorial notebook (where applicable)
- Pipeline check → example script if pipeline exists

**The foundation is excellent. Time to build!**
