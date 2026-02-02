"""
Optimizer Configurations.

This module provides dataclass configurations for various scipy optimizers:
- LBFGSBConfig: L-BFGS-B (gradient-based, bounded)
- LevenbergMarquardtConfig: Levenberg-Marquardt (least-squares)
- DifferentialEvolutionConfig: Differential Evolution (global)

Each config encapsulates optimizer-specific settings in a type-safe,
documented dataclass that can be passed to CalibrationEngine.

Example
-------
>>> from src.calibration.core import CalibrationEngine, LBFGSBConfig
>>>
>>> # Quick local optimization
>>> engine = CalibrationEngine(optimizer=LBFGSBConfig(max_iter=500))
>>>
>>> # Global optimization for difficult surfaces
>>> engine = CalibrationEngine(
...     optimizer=DifferentialEvolutionConfig(
...         strategy="best1bin",
...         polish=True,  # Refine with L-BFGS-B
...     )
... )
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union


# =============================================================================
# Base Config
# =============================================================================

@dataclass(frozen=True, slots=True)
class OptimizerConfig(ABC):
    """
    Base class for optimizer configurations.
    
    All optimizer configs should inherit from this class to enable
    type-checking in CalibrationEngine.
    """
    pass


# =============================================================================
# L-BFGS-B Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class LBFGSBConfig(OptimizerConfig):
    """
    Configuration for L-BFGS-B optimizer.
    
    L-BFGS-B is a quasi-Newton method that supports bound constraints.
    It approximates the Hessian using limited-memory BFGS updates.
    
    This is the default optimizer for most calibration tasks because:
    - Fast convergence near optima
    - Low memory usage
    - Handles box constraints
    - No gradient required (uses finite differences)
    
    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for function value convergence (ftol).
        Stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
    gtol : float
        Tolerance for gradient norm.
        Stops when max{|proj g_i|} <= gtol where proj g is projected gradient.
    max_fun_evals : int
        Maximum number of function evaluations.
    
    Notes
    -----
    - For ill-conditioned problems, consider increasing max_iter.
    - For noisy objectives, consider increasing tol.
    - gtol is less useful for numerical gradients.
    
    Examples
    --------
    >>> # Standard calibration
    >>> config = LBFGSBConfig(max_iter=200, tol=1e-8)
    
    >>> # High-precision calibration
    >>> config = LBFGSBConfig(max_iter=1000, tol=1e-12, gtol=1e-8)
    """
    
    max_iter: int = 200
    tol: float = 1e-8
    gtol: float = 1e-5
    max_fun_evals: int = 15000


# =============================================================================
# Levenberg-Marquardt Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class LevenbergMarquardtConfig(OptimizerConfig):
    """
    Configuration for Levenberg-Marquardt optimizer.
    
    Levenberg-Marquardt is specialized for least-squares problems.
    It interpolates between Gauss-Newton and gradient descent.
    
    Best suited for:
    - Problems where objective is sum of squared residuals
    - When you can provide individual residuals (not just total error)
    
    Parameters
    ----------
    max_fun_evals : int
        Maximum number of function evaluations.
    ftol : float
        Tolerance for function value change.
    xtol : float
        Tolerance for parameter change.
    gtol : float
        Tolerance for gradient.
    
    Notes
    -----
    - scipy.optimize.least_squares is used as backend
    - Bounds supported via 'trf' (Trust Region Reflective) method
    - Without bounds, uses pure 'lm' method
    
    Examples
    --------
    >>> config = LevenbergMarquardtConfig(max_fun_evals=500, ftol=1e-10)
    """
    
    max_fun_evals: int = 500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8


# =============================================================================
# Differential Evolution Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class DifferentialEvolutionConfig(OptimizerConfig):
    """
    Configuration for Differential Evolution optimizer.
    
    Differential Evolution is a global optimizer that uses a population
    of candidate solutions. It's robust to local minima but slower than
    gradient-based methods.
    
    Best suited for:
    - Multi-modal objective functions (multiple local minima)
    - When good initial guess is not available
    - When gradient is unreliable or expensive
    
    Parameters
    ----------
    strategy : str
        DE mutation strategy. Common choices:
        - "best1bin": Use best member for mutation (fast convergence)
        - "rand1bin": Random member (more exploration)
        - "best2bin": Two difference vectors from best
    max_iter : int
        Maximum number of generations.
    popsize : int
        Population size multiplier (actual pop = popsize * n_params).
    tol : float
        Convergence tolerance on standard deviation of population.
    mutation : tuple of float
        Mutation constant (F) or range (Fmin, Fmax) for dithering.
    recombination : float
        Crossover probability [0, 1]. Higher = more exploration.
    seed : int, optional
        Random seed for reproducibility.
    polish : bool
        If True, refine result with L-BFGS-B after DE converges.
        Strongly recommended for final precision.
    workers : int
        Number of parallel workers. -1 = use all cores.
    
    Notes
    -----
    - Always requires bounds (DE explores within bounds)
    - polish=True adds minimal overhead but improves precision
    - For faster results, reduce popsize (may miss global optimum)
    - workers > 1 requires picklable objective function
    
    Examples
    --------
    >>> # Standard global search
    >>> config = DifferentialEvolutionConfig(
    ...     strategy="best1bin",
    ...     max_iter=1000,
    ...     polish=True,
    ... )
    
    >>> # More exploration
    >>> config = DifferentialEvolutionConfig(
    ...     strategy="rand1bin",
    ...     popsize=25,
    ...     recombination=0.9,
    ... )
    
    >>> # Parallel evaluation
    >>> config = DifferentialEvolutionConfig(workers=-1)
    """
    
    strategy: str = "best1bin"
    max_iter: int = 1000
    popsize: int = 15
    tol: float = 0.01
    mutation: Tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: Optional[int] = None
    polish: bool = True
    workers: int = 1


# =============================================================================
# Factory Functions
# =============================================================================

def get_default_optimizer(
    problem_type: Literal["local", "global", "least_squares"] = "local"
) -> OptimizerConfig:
    """
    Get default optimizer configuration for problem type.
    
    Parameters
    ----------
    problem_type : str
        - "local": Standard local optimization (L-BFGS-B)
        - "global": Global optimization (Differential Evolution)
        - "least_squares": Least-squares problem (Levenberg-Marquardt)
    
    Returns
    -------
    OptimizerConfig
        Appropriate optimizer configuration.
    
    Examples
    --------
    >>> optimizer = get_default_optimizer("global")
    >>> engine = CalibrationEngine(optimizer=optimizer)
    """
    if problem_type == "local":
        return LBFGSBConfig()
    elif problem_type == "global":
        return DifferentialEvolutionConfig(polish=True)
    elif problem_type == "least_squares":
        return LevenbergMarquardtConfig()
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def create_global_then_local_optimizer(
    global_iters: int = 500,
    local_iters: int = 200,
) -> DifferentialEvolutionConfig:
    """
    Create optimizer that does global search followed by local refinement.
    
    This is recommended for difficult calibration problems like Heston
    where the objective surface has multiple local minima.
    
    Parameters
    ----------
    global_iters : int
        Maximum DE iterations.
    local_iters : int
        This is handled by polish=True (uses L-BFGS-B defaults).
    
    Returns
    -------
    DifferentialEvolutionConfig
        Configured for global + local optimization.
    
    Examples
    --------
    >>> optimizer = create_global_then_local_optimizer()
    >>> engine = CalibrationEngine(optimizer=optimizer)
    """
    return DifferentialEvolutionConfig(
        strategy="best1bin",
        max_iter=global_iters,
        polish=True,  # Refine with L-BFGS-B
    )
