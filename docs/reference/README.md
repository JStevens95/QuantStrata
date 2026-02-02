# QuantStrata Technical Reference

This section contains mathematical foundations, derivations, and detailed model specifications.

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
