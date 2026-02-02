# Phase 4.1: Advanced Stochastic Models - Progress Report

**Status:** COMPLETE  
**Completed:** January 27, 2026

---

## Overview

Phase 4.1 implements three advanced stochastic models that go beyond Black-Scholes and Heston, providing tools for modeling fat tails, jumps, and complex volatility dynamics.

---

## Deliverables

### 1. Merton Jump-Diffusion Model

**Location:** `src/models/jump_diffusion/`

**Components:**
- `MertonParameters`: Model parameters with validation
- `MertonDynamics`: Monte Carlo simulation (path and terminal)
- `MertonSimulation`: Output container with jump tracking
- `merton_european_call/put`: Analytic series pricing
- `merton_implied_vol`: BS implied vol computation

**Features:**
- GBM + compound Poisson jumps
- Log-normal jump size distribution
- Martingale-preserving drift correction
- Exact terminal simulation for efficiency
- Antithetic variance reduction

**Tests:** 46 unit tests passing

### 2. SABR Stochastic Volatility Model

**Location:** `src/models/stochastic_volatility/sabr.py` (dynamics)  
**Calibration:** `src/calibration/volatility_surface/sabr.py` (analytics)

**Components:**
- `SabrParameters`: Model parameters (α, β, ρ, ν)
- `SabrDynamics`: Monte Carlo simulation
- `SabrSimulation`: Output with vol paths
- `sabr_implied_vol`: Hagan approximation
- `calibrate_sabr_to_smile`: Market calibration

**Features:**
- CEV backbone (β = 0, 0.5, 1)
- Stochastic volatility
- Log-Euler discretization for stability
- Multiple schemes (euler, log_euler, absorbing)

**Tests:** 25 unit tests passing

### 3. Variance Gamma Model

**Location:** `src/models/levy/`

**Components:**
- `VarianceGammaParameters`: Model parameters (θ, σ, ν)
- `VarianceGammaDynamics`: Subordination-based simulation
- `VarianceGammaSimulation`: Output with Gamma time tracking
- `vg_european_call/put`: Monte Carlo pricing
- `vg_characteristic_function`: For FFT methods

**Features:**
- Pure-jump Lévy process
- Time-changed Brownian motion construction
- Controllable skewness and kurtosis
- Exact terminal simulation
- Moment properties (variance, skew, kurtosis)

**Tests:** 38 unit tests passing

---

## Test Summary

**Total Tests:** 109 passing

| Model | Tests |
|-------|-------|
| Merton Jump-Diffusion | 46 |
| SABR Dynamics | 25 |
| Variance Gamma | 38 |

---

## Documentation

### Technical References
- `docs/reference/models/merton_jump_diffusion.md`
- `docs/reference/models/sabr.md`
- `docs/reference/models/variance_gamma.md`

### User Guides
- `docs/guides/models/merton.md`
- `docs/guides/models/variance_gamma.md`

### Tutorials
- `docs/tutorials/pricing/jump_levy_models.ipynb` - Merton and Variance Gamma with visualizations
- `docs/tutorials/pricing/sabr_model.ipynb` - SABR smile modeling, calibration, and Monte Carlo

---

## Architecture

### Directory Structure

```
src/models/
├── jump_diffusion/          # NEW
│   ├── __init__.py
│   └── merton.py            # Merton model
├── levy/                    # NEW
│   ├── __init__.py
│   └── variance_gamma.py    # VG model
└── stochastic_volatility/
    ├── heston.py            # Existing
    └── sabr.py              # NEW (dynamics)
```

### Design Patterns

1. **Consistent Structure**: All models follow the pattern:
   - `Parameters` dataclass (frozen, validated)
   - `Dynamics` class (simulation methods)
   - `Simulation` output container

2. **Monte Carlo Integration**: Uses `NormalRng` from `src/models/numeric/monte_carlo/rng.py`

3. **Analytic Pricing**: Where available (Merton series, SABR Hagan)

---

## Key Design Decisions

### Merton Model

1. **Exact Terminal Simulation**: Efficient for European pricing
2. **Jump Tracking**: `jump_counts` array for analysis
3. **Series Pricing**: 50 terms sufficient for convergence

### SABR Model

1. **Log-Euler Default**: Preserves positivity for β=1
2. **Absorbing Scheme**: For β<1 boundary handling
3. **Calibration Integration**: Re-exports from calibration module

### Variance Gamma Model

1. **Subordination Construction**: Clean implementation
2. **MC Pricing**: More robust than FFT for general use
3. **Gamma Time Tracking**: For path analysis

---

## Model Comparison

| Feature | Merton | SABR | Variance Gamma |
|---------|--------|------|----------------|
| Type | Jump-diffusion | Stochastic vol | Pure jump |
| Parameters | 4 | 4 | 3 |
| Diffusion | Yes | Yes | No |
| Jumps | Poisson | No | Gamma subordination |
| Analytic | Series | Hagan approx | Char function |
| Primary Use | Crash risk | Vol smile | Fat tails |

---

## Usage Examples

### Merton Pricing

```python
from src.models.jump_diffusion import MertonDynamics, MertonParameters

params = MertonParameters(sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2)
dynamics = MertonDynamics(params=params, drift=0.05)
S_T = dynamics.simulate_exact(spot0=100, maturity=1.0, n_paths=100000)
```

### SABR Simulation

```python
from src.models.stochastic_volatility import SabrDynamics, SabrParameters

params = SabrParameters(alpha=0.3, beta=1.0, rho=-0.5, nu=0.4)
dynamics = SabrDynamics(params=params)
sim = dynamics.simulate(forward0=100, maturity=1.0, n_paths=10000, n_steps=100)
```

### Variance Gamma Pricing

```python
from src.models.levy import VarianceGammaDynamics, VarianceGammaParameters

params = VarianceGammaParameters(theta=-0.1, sigma=0.2, nu=0.2)
dynamics = VarianceGammaDynamics(params=params, drift=0.05)
S_T = dynamics.simulate_terminal(spot0=100, maturity=1.0, n_paths=100000)
```

---

## Future Enhancements

1. **FFT Pricing**: Implement Carr-Madan for VG (characteristic function available)
2. **Greeks**: Analytic or finite difference Greeks for all models
3. **Calibration**: Joint calibration to market smiles
4. **Path-Dependent Products**: Asian, barrier options under these models

---

*Document Version: 1.0 | QuantStrata Phase 4.1 | January 2026*
