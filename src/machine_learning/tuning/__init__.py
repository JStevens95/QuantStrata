"""
Hyperparameter tuning utilities for QuantStrata ML models.

Provides advanced tuning capabilities including:
- Search space definitions
- Optuna-based Bayesian optimization
- Trial pruning for early stopping
- Integration with experiment tracking

Usage:
    from src.machine_learning.tuning import (
        SearchSpace, OptunaSearchSpace, TrialPruner,
        run_optuna_tuning, TuningResult
    )
    
    # Define search space
    space = SearchSpace()
    space.add_float("learning_rate", 1e-4, 1e-2, log=True)
    space.add_int("hidden_units", 32, 256)
    space.add_categorical("activation", ["relu", "tanh"])
    
    # Run tuning
    result = run_optuna_tuning(
        objective_fn=train_and_evaluate,
        search_space=space,
        n_trials=100,
    )
"""

from src.machine_learning.tuning.search_space import (
    SearchSpace,
    OptunaSearchSpace,
    ParameterType,
    ParameterDefinition,
    TrialPruner,
    MedianPruner,
    PercentilePruner,
    TuningResult,
    TuningTrial,
    run_optuna_tuning,
    create_search_space,
)

__all__ = [
    # Search space
    "SearchSpace",
    "OptunaSearchSpace",
    "ParameterType",
    "ParameterDefinition",
    # Pruners
    "TrialPruner",
    "MedianPruner",
    "PercentilePruner",
    # Results
    "TuningResult",
    "TuningTrial",
    # Functions
    "run_optuna_tuning",
    "create_search_space",
]
