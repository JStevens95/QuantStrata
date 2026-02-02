# Calibration Framework Technical Reference

## Overview

The QuantStrata calibration framework provides a unified interface for calibrating model parameters to market data. The framework is built on three core components:

1. **CalibrationEngine**: Orchestrates optimization with pluggable objectives and optimizers
2. **ObjectiveFunction**: Defines what to minimize (least squares, likelihood, etc.)
3. **OptimizerConfig**: Configures the underlying scipy optimizer

## Architecture

```
src/calibration/
├── core/
│   ├── engine.py      # CalibrationEngine, CalibrationResult
│   ├── objectives.py  # WeightedLeastSquares, PenalizedObjective
│   └── optimizers.py  # LBFGSBConfig, DifferentialEvolutionConfig
├── stochastic_volatility/
│   └── heston.py      # Heston model calibration
├── short_rate/
│   └── hull_white.py  # Hull-White model calibration
└── volatility_surface/
    ├── sabr.py        # SABR calibration (FX and IR)
    └── dupire.py      # Local vol extraction
```

## CalibrationEngine

The `CalibrationEngine` class is the central orchestrator for all calibration tasks.

### Class Definition

```python
class CalibrationEngine:
    def __init__(
        self,
        optimizer: OptimizerConfig = LBFGSBConfig(),
        config: CalibrationConfig = CalibrationConfig(),
    ) -> None:
        ...
    
    def calibrate(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        bounds: Sequence[tuple[float, float]] | None = None,
        param_names: Sequence[str] | None = None,
    ) -> CalibrationResult:
        ...
```

### CalibrationResult

```python
@dataclass(frozen=True, slots=True)
class CalibrationResult:
    params: np.ndarray           # Calibrated parameters
    objective_value: float       # Final objective function value
    success: bool                # Whether optimization converged
    n_iterations: int            # Number of iterations
    n_function_evals: int        # Number of objective evaluations
    message: str                 # Termination message
    elapsed_time: float          # Wall-clock time (seconds)
    initial_params: np.ndarray   # Initial guess
    initial_objective: float     # Objective at initial guess
    
    @property
    def improvement_ratio(self) -> float:
        """Ratio of initial to final objective."""
        ...
```

## Objective Functions

### WeightedLeastSquares

The most common objective for calibration - minimizes weighted sum of squared errors:

$$\text{Objective} = \sum_i w_i \cdot (\text{model}_i - \text{market}_i)^2$$

```python
@dataclass
class WeightedLeastSquares:
    model_func: Callable[[np.ndarray], np.ndarray]
    market_values: np.ndarray
    weights: Optional[np.ndarray] = None
    use_relative_error: bool = False
```

### PenalizedObjective

Wraps any objective to add soft constraints (penalties):

```python
@dataclass
class PenalizedObjective:
    base_objective: Callable[[np.ndarray], float]
    penalty_func: Callable[[np.ndarray], float]
    penalty_weight: float = 1000.0
```

**Example: Feller Constraint for Heston**

```python
def feller_penalty(params):
    kappa, theta, xi = params[0], params[1], params[2]
    violation = xi**2 - 2 * kappa * theta  # Feller: 2κθ > ξ²
    return max(0, violation)**2

penalized = PenalizedObjective(
    base_objective=vol_fitting_objective,
    penalty_func=feller_penalty,
    penalty_weight=1000.0,
)
```

### MaxLikelihood

For probabilistic calibration:

$$\text{Objective} = -\sum_i \log L(x_i | \theta)$$

```python
@dataclass
class MaxLikelihood:
    log_likelihood_func: Callable[[np.ndarray], float]
    regularization: float = 0.0
```

## Optimizer Configurations

### LBFGSBConfig (Default)

L-BFGS-B is a quasi-Newton method with box constraints. Best for:
- Smooth objective functions
- When good initial guess is available
- Fast local convergence needed

```python
@dataclass(frozen=True, slots=True)
class LBFGSBConfig(OptimizerConfig):
    max_iter: int = 200
    tol: float = 1e-8
    gtol: float = 1e-5
    max_fun_evals: int = 15000
```

### DifferentialEvolutionConfig (Global)

Differential Evolution is a global optimizer. Best for:
- Multi-modal objectives (multiple local minima)
- Unknown starting region
- Robustness over speed

```python
@dataclass(frozen=True, slots=True)
class DifferentialEvolutionConfig(OptimizerConfig):
    strategy: str = "best1bin"
    max_iter: int = 1000
    popsize: int = 15
    tol: float = 0.01
    mutation: Tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    polish: bool = True  # Refine with L-BFGS-B
```

### LevenbergMarquardtConfig

Specialized for least-squares problems:

```python
@dataclass(frozen=True, slots=True)
class LevenbergMarquardtConfig(OptimizerConfig):
    max_fun_evals: int = 500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
```

## Model-Specific Calibration

### Heston Calibration

Calibrates Heston stochastic volatility model to implied vol surface.

**Parameters:**
- κ (kappa): Mean reversion speed
- θ (theta): Long-term variance
- ξ (xi): Vol-of-vol
- V₀ (v0): Initial variance
- ρ (rho): Spot-variance correlation

**Function:**
```python
def calibrate_heston_to_surface(
    surface: GridVolSurface,
    spot: float,
    r: float,
    q: float,
    config: HestonCalibrationConfig = HestonCalibrationConfig(),
    initial_guess: HestonParameters | None = None,
) -> HestonCalibrationResult:
```

**Pricing Method:** Carr-Madan FFT using characteristic function

**Key Options:**
- `fix_v0_to_atm=True`: Set V₀ = σ²_ATM (4-parameter calibration)
- `enforce_feller=True`: Penalize Feller violation
- `use_global_optimizer=True`: Use DE + local refinement

### Hull-White Calibration

Calibrates Hull-White 1-factor short rate model to swaption/cap volatilities.

**Parameters:**
- a: Mean reversion speed
- σ: Short rate volatility

**Functions:**
```python
def calibrate_hull_white_to_swaptions(
    swaption_vols: np.ndarray,
    expiries: np.ndarray,
    tenors: np.ndarray,
    yield_curve_df: Callable[[float], float],
    r0: float,
    config: HullWhiteCalibrationConfig = HullWhiteCalibrationConfig(),
) -> HullWhiteCalibrationResult:

def calibrate_hull_white_to_caps(
    cap_vols: np.ndarray,
    expiries: np.ndarray,
    yield_curve_df: Callable[[float], float],
    r0: float,
    config: HullWhiteCalibrationConfig = HullWhiteCalibrationConfig(),
) -> HullWhiteCalibrationResult:
```

**Pricing Method:** Jamshidian decomposition for swaptions, analytic caplet formula

### SABR Calibration

Calibrates SABR model to smile data.

**Parameters:**
- α (alpha): Initial volatility level
- β (beta): CEV exponent (typically fixed)
- ρ (rho): Correlation
- ν (nu): Vol-of-vol

**Functions:**
```python
# FX smile calibration
def calibrate_sabr_to_smile(
    forward: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    expiry: float,
    config: SabrConfig = SabrConfig(),
) -> SabrParameters:

# IR swaption smile calibration
def calibrate_sabr_to_swaption_smile(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    forward_swap_rate: float,
    expiry: float,
    tenor: float,
    vol_type: str = "normal",  # "normal" or "lognormal"
    config: SabrConfig = SabrConfig(beta=0.0),
) -> SabrParameters:
```

**Key Options:**
- `beta=1.0`: Log-normal SABR (FX)
- `beta=0.0`: Normal SABR (rates, handles negative rates)

## Numerical Considerations

### Initial Guess Selection

Good initial guesses are crucial for local optimizers:

1. **ATM volatility**: Use σ_ATM for initial variance (Heston V₀, SABR α)
2. **Historical estimates**: Use historical mean reversion if available
3. **Previous calibration**: Use yesterday's parameters as starting point

### Convergence Issues

Common causes and solutions:

| Issue | Solution |
|-------|----------|
| Local minimum | Use `DifferentialEvolutionConfig` with `polish=True` |
| Slow convergence | Increase `max_iter`, check objective scaling |
| Unstable results | Add regularization, tighten bounds |
| Feller violation | Use `PenalizedObjective` with Feller constraint |

### Performance Tips

1. **Subsample surface**: For Heston, use `use_subset=True` to reduce grid
2. **Vectorize model function**: Return all model values in single call
3. **Cache expensive computations**: Reuse characteristic function evaluations
4. **Parallel evaluation**: Use `workers=-1` for Differential Evolution

## References

1. Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility."
2. Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities."
3. Hagan, P. et al. (2002). "Managing Smile Risk." Wilmott Magazine.
4. Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform."

---

*QuantStrata Calibration Framework | Phase 5.1 | January 2026*
