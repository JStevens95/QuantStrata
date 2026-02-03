# QuantStrata Technical Reference

This section contains mathematical foundations, derivations, and detailed model specifications.

---

## Mathematical appendices

The following reference docs under [models/](models/) and [calibration/](calibration/) serve as mathematical appendices: derivations, PDE/numerical methods, and model specifications.

| Topic | Document | Description |
|-------|----------|-------------|
| Finite difference methods | [finite_difference_methods.md](models/finite_difference_methods.md) | PDE discretization, schemes, boundaries |
| Monte Carlo methods | [monte_carlo_methods.md](models/monte_carlo_methods.md) | Path generation, variance reduction |
| Heston | [heston_volatility.md](models/heston_volatility.md) | Stochastic vol SDE, characteristic function |
| Hull-White | [hull_white.md](models/hull_white.md) | One-factor short rate, bond options |
| SABR | [sabr.md](models/sabr.md) | Hagan formula, calibration |
| Local volatility | [local_volatility.md](models/local_volatility.md) | Dupire equation, surface construction |
| Curve bootstrapping | [curve_bootstrapping.md](models/curve_bootstrapping.md) | Discount curves, interpolation |
| Volatility calibration | [volatility_calibration.md](models/volatility_calibration.md) | Surface fitting, arbitrage-free conditions |

See the sections below for the full reference index.

---

## Calibration

| Reference | Description |
|-----------|-------------|
| [Calibration Framework](calibration/calibration_framework.md) | Architecture, objectives, optimizers, model-specific calibrators |

---

## Analytic Models

| Reference | Description |
|-----------|-------------|
| [Black-Scholes-Merton](models/black_scholes_merton.md) | Derivation, Greeks, cost-of-carry formulation |
| [Black-76](models/black76.md) | Forward measure pricing, cap/floor formulas |
| [Bachelier](models/bachelier.md) | Normal model derivation, spread option pricing |

---

## Stochastic Volatility

| Reference | Description |
|-----------|-------------|
| [Heston](models/heston_volatility.md) | SDE, characteristic function, simulation schemes |
| [SABR](models/sabr.md) | Hagan approximation, calibration, normal vs lognormal |
| [Local Volatility](models/local_volatility.md) | Dupire formula, surface construction |

---

## Short Rate Models

| Reference | Description |
|-----------|-------------|
| [Hull-White](models/hull_white.md) | One-factor model, bond options, swaptions |
| [Black-Karasinski](models/black_karasinski.md) | Log-normal short rate, simulation |

---

## Forward Rate Models

| Reference | Description |
|-----------|-------------|
| [LMM](models/lmm.md) | LIBOR Market Model, drift correction, simulation |

---

## Jump and Lévy Models

| Reference | Description |
|-----------|-------------|
| [Merton Jump-Diffusion](models/merton_jump_diffusion.md) | Poisson jumps, characteristic function |
| [Variance Gamma](models/variance_gamma.md) | Lévy process, simulation methods |

---

## Numerical Methods

| Reference | Description |
|-----------|-------------|
| [Monte Carlo](models/monte_carlo_methods.md) | Path generation, variance reduction, convergence |
| [Finite Difference](models/finite_difference_methods.md) | PDE discretization, boundary conditions, stability |

---

## Market Data

| Reference | Description |
|-----------|-------------|
| [Curve Bootstrapping](models/curve_bootstrapping.md) | Instrument selection, interpolation, consistency |
| [Volatility Calibration](models/volatility_calibration.md) | Surface fitting, arbitrage-free conditions |

---

## Performance

| Reference | Description |
|-----------|-------------|
| [Performance Optimisation](performance_optimisation.md) | Vectorization, caching, backend selection |

---

*See also: [User Guides](../guides/) | [Tutorials](../tutorials/)*
