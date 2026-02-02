"""
Calibration Engine.

This module provides the core CalibrationEngine class that orchestrates
optimization with pluggable objectives and optimizers.

The engine provides a unified interface for calibrating any model parameters
to market data, supporting multiple optimization backends and objective functions.

Example
-------
>>> from src.calibration.core import CalibrationEngine, LBFGSBConfig
>>> from src.calibration.core.objectives import WeightedLeastSquares
>>>
>>> # Define model function that maps params -> model values
>>> def model_func(params):
...     return some_model_price(params)
>>>
>>> objective = WeightedLeastSquares(
...     model_func=model_func,
...     market_values=market_prices,
...     weights=weights,
... )
>>>
>>> engine = CalibrationEngine(
...     optimizer=LBFGSBConfig(max_iter=200, tol=1e-8),
... )
>>>
>>> result = engine.calibrate(
...     objective=objective,
...     initial_params=x0,
...     bounds=bounds,
... )
>>> print(f"Calibrated params: {result.params}")
>>> print(f"Final error: {result.objective_value}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from scipy import optimize

from src.calibration.core.optimizers import (
    OptimizerConfig,
    LBFGSBConfig,
    LevenbergMarquardtConfig,
    DifferentialEvolutionConfig,
)


# =============================================================================
# Calibration Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """
    Result container for calibration.
    
    Attributes
    ----------
    params : np.ndarray
        Calibrated parameter values.
    objective_value : float
        Final objective function value.
    success : bool
        Whether optimization converged successfully.
    n_iterations : int
        Number of iterations performed.
    n_function_evals : int
        Number of objective function evaluations.
    message : str
        Optimizer termination message.
    elapsed_time : float
        Wall-clock time for calibration (seconds).
    initial_params : np.ndarray
        Initial parameter guess.
    initial_objective : float
        Objective value at initial parameters.
    optimizer_result : Any
        Raw result from underlying optimizer (for debugging).
    
    Properties
    ----------
    improvement_ratio : float
        Ratio of initial to final objective (higher = better fit).
    converged : bool
        Alias for success.
    """
    
    params: np.ndarray
    objective_value: float
    success: bool
    n_iterations: int
    n_function_evals: int
    message: str
    elapsed_time: float
    initial_params: np.ndarray
    initial_objective: float
    optimizer_result: Any = field(default=None, repr=False)
    
    @property
    def improvement_ratio(self) -> float:
        """Ratio of initial to final objective (higher = better improvement)."""
        if self.objective_value <= 0:
            return float("inf")
        return self.initial_objective / self.objective_value
    
    @property
    def converged(self) -> bool:
        """Alias for success."""
        return self.success
    
    def __str__(self) -> str:
        """Human-readable summary."""
        status = "CONVERGED" if self.success else "NOT CONVERGED"
        return (
            f"CalibrationResult({status})\n"
            f"  Final objective: {self.objective_value:.6e}\n"
            f"  Improvement: {self.improvement_ratio:.2f}x\n"
            f"  Iterations: {self.n_iterations}\n"
            f"  Time: {self.elapsed_time:.2f}s\n"
            f"  Message: {self.message}"
        )


# =============================================================================
# Calibration Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """
    Global calibration configuration.
    
    Parameters
    ----------
    verbose : bool
        Print progress during calibration.
    check_gradient : bool
        Verify gradient numerically (slow, for debugging).
    finite_diff_eps : float
        Step size for numerical gradient checking.
    retry_on_failure : bool
        If True, retry with perturbed initial guess on failure.
    max_retries : int
        Maximum number of retry attempts.
    perturbation_scale : float
        Scale of random perturbation for retries (fraction of bounds).
    """
    
    verbose: bool = False
    check_gradient: bool = False
    finite_diff_eps: float = 1e-8
    retry_on_failure: bool = True
    max_retries: int = 3
    perturbation_scale: float = 0.1


# =============================================================================
# Calibration Engine
# =============================================================================

class CalibrationEngine:
    """
    Generic calibration engine with pluggable objectives and optimizers.
    
    The engine orchestrates the optimization process:
    1. Validates inputs (bounds, initial params)
    2. Wraps objective for error handling
    3. Runs the optimizer
    4. Optionally retries on failure
    5. Returns standardized CalibrationResult
    
    Parameters
    ----------
    optimizer : OptimizerConfig
        Optimizer configuration (LBFGSBConfig, DifferentialEvolutionConfig, etc.)
    config : CalibrationConfig
        Global calibration settings.
    
    Examples
    --------
    >>> engine = CalibrationEngine(
    ...     optimizer=LBFGSBConfig(max_iter=500),
    ...     config=CalibrationConfig(verbose=True),
    ... )
    >>> result = engine.calibrate(objective, x0, bounds)
    """
    
    def __init__(
        self,
        optimizer: OptimizerConfig = LBFGSBConfig(),
        config: CalibrationConfig = CalibrationConfig(),
    ) -> None:
        self.optimizer = optimizer
        self.config = config
    
    def calibrate(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: np.ndarray | Sequence[float],
        bounds: Sequence[tuple[float, float]] | None = None,
        param_names: Sequence[str] | None = None,
    ) -> CalibrationResult:
        """
        Run calibration.
        
        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to minimize. Takes parameter array, returns scalar.
        initial_params : array-like
            Initial parameter guess.
        bounds : sequence of (min, max) tuples, optional
            Parameter bounds. Required for some optimizers.
        param_names : sequence of str, optional
            Parameter names for verbose output.
        
        Returns
        -------
        CalibrationResult
            Calibration results including optimal parameters and diagnostics.
        """
        # Convert inputs
        x0 = np.asarray(initial_params, dtype=float).reshape(-1)
        n_params = x0.size
        
        # Validate bounds
        if bounds is not None:
            bounds = list(bounds)
            if len(bounds) != n_params:
                raise ValueError(
                    f"bounds length ({len(bounds)}) must match params ({n_params})."
                )
        
        # Compute initial objective
        try:
            initial_obj = float(objective(x0))
        except Exception as e:
            raise ValueError(f"Objective evaluation failed at initial params: {e}")
        
        if not np.isfinite(initial_obj):
            raise ValueError(f"Initial objective is not finite: {initial_obj}")
        
        if self.config.verbose:
            print(f"Starting calibration with {n_params} parameters")
            print(f"Initial objective: {initial_obj:.6e}")
            if param_names:
                for name, val in zip(param_names, x0):
                    print(f"  {name}: {val:.6f}")
        
        # Wrap objective for error handling
        eval_count = [0]
        
        def safe_objective(x: np.ndarray) -> float:
            eval_count[0] += 1
            try:
                val = float(objective(x))
                if not np.isfinite(val):
                    return 1e20  # Large penalty for invalid params
                return val
            except Exception:
                return 1e20
        
        # Run optimization with retries
        best_result = None
        current_x0 = x0.copy()
        
        for attempt in range(1 + self.config.max_retries):
            if attempt > 0:
                if not self.config.retry_on_failure:
                    break
                # Perturb initial guess
                if bounds is not None:
                    lb = np.array([b[0] for b in bounds])
                    ub = np.array([b[1] for b in bounds])
                    scale = (ub - lb) * self.config.perturbation_scale
                    perturbation = np.random.uniform(-1, 1, n_params) * scale
                    current_x0 = np.clip(x0 + perturbation, lb, ub)
                else:
                    perturbation = np.random.randn(n_params) * self.config.perturbation_scale
                    current_x0 = x0 + perturbation * np.abs(x0)
                
                if self.config.verbose:
                    print(f"\nRetry {attempt}/{self.config.max_retries} with perturbed initial guess")
            
            start_time = time.perf_counter()
            result = self._run_optimizer(safe_objective, current_x0, bounds)
            elapsed = time.perf_counter() - start_time
            
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_elapsed = elapsed
                best_x0 = current_x0.copy()
            
            if result.success:
                break
        
        # Build result
        cal_result = CalibrationResult(
            params=np.asarray(best_result.x, dtype=float),
            objective_value=float(best_result.fun),
            success=bool(best_result.success),
            n_iterations=int(getattr(best_result, "nit", 0)),
            n_function_evals=eval_count[0],
            message=str(getattr(best_result, "message", "")),
            elapsed_time=best_elapsed,
            initial_params=x0,
            initial_objective=initial_obj,
            optimizer_result=best_result,
        )
        
        if self.config.verbose:
            print(f"\n{cal_result}")
        
        return cal_result
    
    def _run_optimizer(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Sequence[tuple[float, float]] | None,
    ) -> optimize.OptimizeResult:
        """Run the underlying optimizer."""
        
        if isinstance(self.optimizer, LBFGSBConfig):
            return self._run_lbfgsb(objective, x0, bounds)
        elif isinstance(self.optimizer, LevenbergMarquardtConfig):
            return self._run_levenberg_marquardt(objective, x0, bounds)
        elif isinstance(self.optimizer, DifferentialEvolutionConfig):
            return self._run_differential_evolution(objective, bounds)
        else:
            raise ValueError(f"Unknown optimizer type: {type(self.optimizer)}")
    
    def _run_lbfgsb(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Sequence[tuple[float, float]] | None,
    ) -> optimize.OptimizeResult:
        """Run L-BFGS-B optimizer."""
        cfg = self.optimizer
        assert isinstance(cfg, LBFGSBConfig)
        
        return optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": cfg.max_iter,
                "ftol": cfg.tol,
                "gtol": cfg.gtol,
                "maxfun": cfg.max_fun_evals,
                "disp": self.config.verbose,
            },
        )
    
    def _run_levenberg_marquardt(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Sequence[tuple[float, float]] | None,
    ) -> optimize.OptimizeResult:
        """
        Run Levenberg-Marquardt via scipy.optimize.least_squares.
        
        Note: This requires the objective to return residuals, not scalar.
        For scalar objectives, we use a wrapper.
        """
        cfg = self.optimizer
        assert isinstance(cfg, LevenbergMarquardtConfig)
        
        # For scalar objective, convert to single-element residual
        def residual_func(x: np.ndarray) -> np.ndarray:
            return np.array([np.sqrt(max(objective(x), 0))])
        
        # Convert bounds
        if bounds is not None:
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])
        else:
            lb = -np.inf * np.ones_like(x0)
            ub = np.inf * np.ones_like(x0)
        
        result = optimize.least_squares(
            residual_func,
            x0,
            method="lm" if bounds is None else "trf",
            bounds=(lb, ub),
            max_nfev=cfg.max_fun_evals,
            ftol=cfg.ftol,
            xtol=cfg.xtol,
            gtol=cfg.gtol,
            verbose=2 if self.config.verbose else 0,
        )
        
        # Convert to standard OptimizeResult format
        return optimize.OptimizeResult(
            x=result.x,
            fun=result.cost,
            success=result.success,
            message=result.message,
            nit=result.nfev,  # Approximate iterations by function evals
        )
    
    def _run_differential_evolution(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Sequence[tuple[float, float]] | None,
    ) -> optimize.OptimizeResult:
        """Run differential evolution (global optimizer)."""
        cfg = self.optimizer
        assert isinstance(cfg, DifferentialEvolutionConfig)
        
        if bounds is None:
            raise ValueError("Differential evolution requires bounds.")
        
        result = optimize.differential_evolution(
            objective,
            bounds,
            strategy=cfg.strategy,
            maxiter=cfg.max_iter,
            popsize=cfg.popsize,
            tol=cfg.tol,
            mutation=cfg.mutation,
            recombination=cfg.recombination,
            seed=cfg.seed,
            polish=cfg.polish,
            workers=cfg.workers,
            disp=self.config.verbose,
        )
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def calibrate(
    objective: Callable[[np.ndarray], float],
    initial_params: np.ndarray | Sequence[float],
    bounds: Sequence[tuple[float, float]] | None = None,
    optimizer: OptimizerConfig = LBFGSBConfig(),
    verbose: bool = False,
) -> CalibrationResult:
    """
    Convenience function for quick calibration.
    
    Parameters
    ----------
    objective : Callable
        Objective function to minimize.
    initial_params : array-like
        Initial parameter guess.
    bounds : sequence of tuples, optional
        Parameter bounds.
    optimizer : OptimizerConfig
        Optimizer configuration.
    verbose : bool
        Print progress.
    
    Returns
    -------
    CalibrationResult
        Calibration results.
    
    Examples
    --------
    >>> from src.calibration.core import calibrate
    >>> result = calibrate(
    ...     objective=lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
    ...     initial_params=[0, 0],
    ...     bounds=[(-10, 10), (-10, 10)],
    ... )
    >>> print(result.params)  # Should be close to [1, 2]
    """
    engine = CalibrationEngine(
        optimizer=optimizer,
        config=CalibrationConfig(verbose=verbose),
    )
    return engine.calibrate(objective, initial_params, bounds)
