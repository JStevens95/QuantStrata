"""
Advanced hyperparameter tuning with Optuna integration.

Provides:
- SearchSpace: Define parameter distributions for tuning
- OptunaSearchSpace: Optuna-native search space wrapper
- TrialPruner: Early stopping for unpromising trials
- run_optuna_tuning: Run Bayesian optimization with Optuna

Example:
    from src.machine_learning.tuning import (
        SearchSpace, run_optuna_tuning, MedianPruner
    )
    
    # Define search space
    space = SearchSpace()
    space.add_float("learning_rate", 1e-4, 1e-2, log=True)
    space.add_int("hidden_units", 32, 256)
    space.add_categorical("activation", ["relu", "tanh", "gelu"])
    space.add_int("n_layers", 1, 5)
    
    # Define objective
    def objective(config, trial=None):
        model = create_model(config)
        history = model.fit(...)
        
        # Report intermediate values for pruning
        if trial is not None:
            for epoch, loss in enumerate(history["val_loss"]):
                trial.report(loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return history["val_loss"][-1]
    
    # Run tuning
    result = run_optuna_tuning(
        objective_fn=objective,
        search_space=space,
        n_trials=100,
        pruner=MedianPruner(n_startup_trials=5),
        direction="minimize",
    )
    
    print(f"Best params: {result.best_config}")
    print(f"Best score: {result.best_score}")
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Types and Definitions
# =============================================================================


class ParameterType(Enum):
    """Types of hyperparameters."""
    
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOL = "bool"


@dataclass
class ParameterDefinition:
    """Definition of a hyperparameter for search."""
    
    name: str
    param_type: ParameterType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate parameter definition."""
        if self.param_type in (ParameterType.FLOAT, ParameterType.INT):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Parameters {self.name}: low and high required "
                    f"for {self.param_type.value}"
                )
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"Parameter {self.name}: choices required for categorical"
                )
    
    def sample_random(self, rng: Any = None) -> Any:
        """Sample a random value from this parameter's distribution."""
        import random
        
        if rng is None:
            rng = random
        
        if self.param_type == ParameterType.FLOAT:
            if self.log:
                import math
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return math.exp(rng.uniform(log_low, log_high))
            return rng.uniform(self.low, self.high)
        
        elif self.param_type == ParameterType.INT:
            if self.log:
                import math
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return int(round(math.exp(rng.uniform(log_low, log_high))))
            return rng.randint(int(self.low), int(self.high))
        
        elif self.param_type == ParameterType.CATEGORICAL:
            return rng.choice(self.choices)
        
        elif self.param_type == ParameterType.BOOL:
            return rng.choice([True, False])
        
        raise ValueError(f"Unknown parameter type: {self.param_type}")


# =============================================================================
# Search Space
# =============================================================================


class SearchSpace:
    """
    Define hyperparameter search space for tuning.
    
    Provides a fluent API for defining parameter distributions.
    
    Example:
        space = SearchSpace()
        space.add_float("lr", 1e-4, 1e-2, log=True)
        space.add_int("units", 32, 256)
        space.add_categorical("opt", ["adam", "sgd"])
        
        # Sample a configuration
        config = space.sample()
        
        # Get parameter definitions
        for param in space.parameters:
            print(param.name, param.param_type)
    """
    
    def __init__(self) -> None:
        """Initialize empty search space."""
        self._parameters: Dict[str, ParameterDefinition] = {}
    
    @property
    def parameters(self) -> List[ParameterDefinition]:
        """Return list of parameter definitions."""
        return list(self._parameters.values())
    
    @property
    def names(self) -> List[str]:
        """Return parameter names."""
        return list(self._parameters.keys())
    
    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
    ) -> "SearchSpace":
        """
        Add a float parameter.
        
        Parameters
        ----------
        name : str
            Parameter name.
        low : float
            Lower bound (inclusive).
        high : float
            Upper bound (inclusive).
        log : bool
            If True, sample in log space.
        step : float, optional
            Discretization step.
            
        Returns
        -------
        SearchSpace
            Self for chaining.
        """
        self._parameters[name] = ParameterDefinition(
            name=name,
            param_type=ParameterType.FLOAT,
            low=low,
            high=high,
            log=log,
            step=step,
        )
        return self
    
    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        log: bool = False,
        step: int = 1,
    ) -> "SearchSpace":
        """
        Add an integer parameter.
        
        Parameters
        ----------
        name : str
            Parameter name.
        low : int
            Lower bound (inclusive).
        high : int
            Upper bound (inclusive).
        log : bool
            If True, sample in log space.
        step : int
            Step size.
            
        Returns
        -------
        SearchSpace
            Self for chaining.
        """
        self._parameters[name] = ParameterDefinition(
            name=name,
            param_type=ParameterType.INT,
            low=low,
            high=high,
            log=log,
            step=step,
        )
        return self
    
    def add_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> "SearchSpace":
        """
        Add a categorical parameter.
        
        Parameters
        ----------
        name : str
            Parameter name.
        choices : sequence
            List of possible values.
            
        Returns
        -------
        SearchSpace
            Self for chaining.
        """
        self._parameters[name] = ParameterDefinition(
            name=name,
            param_type=ParameterType.CATEGORICAL,
            choices=list(choices),
        )
        return self
    
    def add_bool(self, name: str) -> "SearchSpace":
        """
        Add a boolean parameter.
        
        Parameters
        ----------
        name : str
            Parameter name.
            
        Returns
        -------
        SearchSpace
            Self for chaining.
        """
        self._parameters[name] = ParameterDefinition(
            name=name,
            param_type=ParameterType.BOOL,
            choices=[True, False],
        )
        return self
    
    def sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Sample a random configuration from the search space.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        dict
            Configuration dictionary.
        """
        import random
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        return {
            name: param.sample_random(rng)
            for name, param in self._parameters.items()
        }
    
    def get(self, name: str) -> Optional[ParameterDefinition]:
        """Get a parameter definition by name."""
        return self._parameters.get(name)
    
    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self._parameters)
    
    def __contains__(self, name: str) -> bool:
        """Check if parameter exists."""
        return name in self._parameters
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export search space as dictionary."""
        return {
            name: {
                "type": param.param_type.value,
                "low": param.low,
                "high": param.high,
                "choices": param.choices,
                "log": param.log,
                "step": param.step,
            }
            for name, param in self._parameters.items()
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, Any]]) -> "SearchSpace":
        """Create search space from dictionary."""
        space = cls()
        for name, config in d.items():
            param_type = ParameterType(config["type"])
            if param_type == ParameterType.FLOAT:
                space.add_float(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                    step=config.get("step"),
                )
            elif param_type == ParameterType.INT:
                space.add_int(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                    step=config.get("step", 1),
                )
            elif param_type == ParameterType.CATEGORICAL:
                space.add_categorical(name, config["choices"])
            elif param_type == ParameterType.BOOL:
                space.add_bool(name)
        return space


class OptunaSearchSpace:
    """
    Optuna-native search space wrapper.
    
    Wraps a SearchSpace for use with Optuna's suggest_* methods.
    
    Example:
        space = SearchSpace()
        space.add_float("lr", 1e-4, 1e-2, log=True)
        
        optuna_space = OptunaSearchSpace(space)
        
        def objective(trial):
            config = optuna_space.sample(trial)
            return train_and_evaluate(config)
    """
    
    def __init__(self, search_space: SearchSpace) -> None:
        """
        Initialize Optuna search space wrapper.
        
        Parameters
        ----------
        search_space : SearchSpace
            The search space to wrap.
        """
        self._space = search_space
    
    def sample(self, trial: Any) -> Dict[str, Any]:
        """
        Sample configuration using Optuna trial.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
            
        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = {}
        
        for param in self._space.parameters:
            if param.param_type == ParameterType.FLOAT:
                config[param.name] = trial.suggest_float(
                    param.name,
                    param.low,
                    param.high,
                    log=param.log,
                    step=param.step,
                )
            elif param.param_type == ParameterType.INT:
                config[param.name] = trial.suggest_int(
                    param.name,
                    int(param.low),
                    int(param.high),
                    log=param.log,
                    step=int(param.step) if param.step else 1,
                )
            elif param.param_type == ParameterType.CATEGORICAL:
                config[param.name] = trial.suggest_categorical(
                    param.name,
                    param.choices,
                )
            elif param.param_type == ParameterType.BOOL:
                config[param.name] = trial.suggest_categorical(
                    param.name,
                    [True, False],
                )
        
        return config


# =============================================================================
# Trial Pruners
# =============================================================================


class TrialPruner(ABC):
    """Base class for trial pruners."""
    
    @abstractmethod
    def should_prune(
        self,
        trial_id: int,
        step: int,
        value: float,
        all_values: Dict[int, List[Tuple[int, float]]],
    ) -> bool:
        """
        Check if trial should be pruned.
        
        Parameters
        ----------
        trial_id : int
            Current trial ID.
        step : int
            Current step/epoch.
        value : float
            Current metric value.
        all_values : dict
            History of all trials: {trial_id: [(step, value), ...]}.
            
        Returns
        -------
        bool
            True if trial should be pruned.
        """
        ...
    
    def to_optuna(self) -> Any:
        """Convert to Optuna pruner if available."""
        raise NotImplementedError("Subclass must implement to_optuna()")


class MedianPruner(TrialPruner):
    """
    Prune trials below median performance.
    
    At each step, computes the median of all completed trials at that step.
    Prunes if current value is worse than median.
    
    Parameters
    ----------
    n_startup_trials : int
        Don't prune until this many trials complete.
    n_warmup_steps : int
        Don't prune until this many steps in each trial.
    interval_steps : int
        Only check pruning every N steps.
    """
    
    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
    ) -> None:
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
    
    def should_prune(
        self,
        trial_id: int,
        step: int,
        value: float,
        all_values: Dict[int, List[Tuple[int, float]]],
    ) -> bool:
        """Check if trial should be pruned based on median criterion."""
        # Don't prune during warmup
        if step < self.n_warmup_steps:
            return False
        
        # Only check at intervals
        if step % self.interval_steps != 0:
            return False
        
        # Collect values at this step from completed trials
        values_at_step = []
        for tid, history in all_values.items():
            if tid == trial_id:
                continue
            for s, v in history:
                if s == step:
                    values_at_step.append(v)
                    break
        
        # Don't prune if not enough comparison data
        if len(values_at_step) < self.n_startup_trials:
            return False
        
        # Compute median and compare
        import numpy as np
        median = np.median(values_at_step)
        
        return value > median
    
    def to_optuna(self) -> Any:
        """Convert to Optuna MedianPruner."""
        try:
            import optuna
            return optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )
        except ImportError:
            raise ImportError("Optuna required for to_optuna()")


class PercentilePruner(TrialPruner):
    """
    Prune trials below given percentile.
    
    Similar to MedianPruner but with configurable percentile threshold.
    
    Parameters
    ----------
    percentile : float
        Percentile threshold (0-100). Trials below this are pruned.
    n_startup_trials : int
        Don't prune until this many trials complete.
    n_warmup_steps : int
        Don't prune until this many steps in each trial.
    """
    
    def __init__(
        self,
        percentile: float = 25.0,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
    ) -> None:
        self.percentile = percentile
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
    
    def should_prune(
        self,
        trial_id: int,
        step: int,
        value: float,
        all_values: Dict[int, List[Tuple[int, float]]],
    ) -> bool:
        """Check if trial should be pruned based on percentile criterion."""
        if step < self.n_warmup_steps:
            return False
        
        values_at_step = []
        for tid, history in all_values.items():
            if tid == trial_id:
                continue
            for s, v in history:
                if s == step:
                    values_at_step.append(v)
                    break
        
        if len(values_at_step) < self.n_startup_trials:
            return False
        
        import numpy as np
        threshold = np.percentile(values_at_step, self.percentile)
        
        return value > threshold
    
    def to_optuna(self) -> Any:
        """Convert to Optuna PercentilePruner."""
        try:
            import optuna
            return optuna.pruners.PercentilePruner(
                percentile=self.percentile,
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
            )
        except ImportError:
            raise ImportError("Optuna required for to_optuna()")


# =============================================================================
# Results
# =============================================================================


@dataclass
class TuningTrial:
    """Result from a single tuning trial."""
    
    trial_id: int
    config: Dict[str, Any]
    score: Optional[float]
    status: str  # "COMPLETE", "PRUNED", "FAIL"
    duration_seconds: float
    intermediate_values: List[Tuple[int, float]] = field(default_factory=list)
    user_attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningResult:
    """
    Complete result from hyperparameter tuning.
    
    Attributes
    ----------
    best_config : dict
        Best hyperparameter configuration.
    best_score : float
        Best objective value achieved.
    best_trial_id : int
        ID of the best trial.
    trials : list
        All trial results.
    n_trials : int
        Total number of trials run.
    n_completed : int
        Number of completed trials.
    n_pruned : int
        Number of pruned trials.
    optimization_history : list
        Best score at each trial.
    metadata : dict
        Additional metadata.
    """
    
    best_config: Dict[str, Any]
    best_score: float
    best_trial_id: int
    trials: List[TuningTrial]
    n_trials: int
    n_completed: int
    n_pruned: int
    optimization_history: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result as dictionary."""
        return {
            "best_config": self.best_config,
            "best_score": self.best_score,
            "best_trial_id": self.best_trial_id,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_pruned": self.n_pruned,
            "optimization_history": self.optimization_history,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "config": t.config,
                    "score": t.score,
                    "status": t.status,
                    "duration_seconds": t.duration_seconds,
                }
                for t in self.trials
            ],
            "metadata": self.metadata,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TuningResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        trials = [
            TuningTrial(
                trial_id=t["trial_id"],
                config=t["config"],
                score=t["score"],
                status=t["status"],
                duration_seconds=t["duration_seconds"],
            )
            for t in data["trials"]
        ]
        
        return cls(
            best_config=data["best_config"],
            best_score=data["best_score"],
            best_trial_id=data["best_trial_id"],
            trials=trials,
            n_trials=data["n_trials"],
            n_completed=data["n_completed"],
            n_pruned=data["n_pruned"],
            optimization_history=data["optimization_history"],
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Optuna Tuning Function
# =============================================================================


def run_optuna_tuning(
    objective_fn: Callable[[Dict[str, Any], Any], float],
    search_space: SearchSpace,
    n_trials: int = 100,
    direction: str = "minimize",
    pruner: Optional[TrialPruner] = None,
    sampler: Optional[str] = "tpe",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = False,
    n_jobs: int = 1,
    timeout: Optional[float] = None,
    callbacks: Optional[List[Callable]] = None,
    show_progress_bar: bool = True,
    seed: Optional[int] = None,
) -> TuningResult:
    """
    Run hyperparameter tuning with Optuna.
    
    Parameters
    ----------
    objective_fn : callable
        Function (config, trial) -> score. The trial argument can be used
        for intermediate reporting and pruning.
    search_space : SearchSpace
        Parameter search space.
    n_trials : int
        Number of trials to run.
    direction : str
        "minimize" or "maximize".
    pruner : TrialPruner, optional
        Pruner for early stopping.
    sampler : str, optional
        Sampling algorithm: "tpe", "random", "cmaes", "grid".
    study_name : str, optional
        Name for the Optuna study.
    storage : str, optional
        Database URL for persistent storage.
    load_if_exists : bool
        If True, load existing study.
    n_jobs : int
        Number of parallel workers.
    timeout : float, optional
        Maximum time in seconds.
    callbacks : list, optional
        Callback functions called after each trial.
    show_progress_bar : bool
        Show progress bar during optimization.
    seed : int, optional
        Random seed for reproducibility (passed to Optuna sampler).
        
    Returns
    -------
    TuningResult
        Complete tuning results.
        
    Example:
        def objective(config, trial):
            model = create_model(**config)
            for epoch in range(100):
                loss = train_epoch(model)
                trial.report(loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return loss
        
        space = SearchSpace()
        space.add_float("lr", 1e-4, 1e-2, log=True)
        
        result = run_optuna_tuning(objective, space, n_trials=100)
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "Optuna is required for run_optuna_tuning(). "
            "Install with: pip install optuna"
        ) from e
    
    # Configure sampler
    sampler_kw: Dict[str, Any] = {} if seed is None else {"seed": seed}
    if sampler == "tpe":
        optuna_sampler = optuna.samplers.TPESampler(**sampler_kw)
    elif sampler == "random":
        optuna_sampler = optuna.samplers.RandomSampler(**sampler_kw)
    elif sampler == "cmaes":
        optuna_sampler = optuna.samplers.CmaEsSampler(**sampler_kw)
    elif sampler == "grid":
        # Grid sampler requires explicit search space
        raise NotImplementedError(
            "Grid sampler not yet supported in run_optuna_tuning. "
            "Use run_tuning() with method='grid' instead."
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    
    # Configure pruner
    optuna_pruner = None
    if pruner is not None:
        optuna_pruner = pruner.to_optuna()
    
    # Create study
    study = optuna.create_study(
        study_name=study_name or f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction=direction,
        sampler=optuna_sampler,
        pruner=optuna_pruner,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    
    # Create Optuna search space wrapper
    optuna_space = OptunaSearchSpace(search_space)
    
    # Wrap objective
    def wrapped_objective(trial: optuna.Trial) -> float:
        config = optuna_space.sample(trial)
        return objective_fn(config, trial)
    
    # Run optimization
    study.optimize(
        wrapped_objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        callbacks=callbacks,
        show_progress_bar=show_progress_bar,
    )
    
    # Collect results
    trials = []
    for trial in study.trials:
        trials.append(TuningTrial(
            trial_id=trial.number,
            config=trial.params,
            score=trial.value,
            status=trial.state.name,
            duration_seconds=(
                (trial.datetime_complete - trial.datetime_start).total_seconds()
                if trial.datetime_complete and trial.datetime_start
                else 0.0
            ),
            intermediate_values=list(trial.intermediate_values.items()),
            user_attrs=dict(trial.user_attrs),
        ))
    
    # Build optimization history
    best_so_far = float("inf") if direction == "minimize" else float("-inf")
    history = []
    for trial in trials:
        if trial.score is not None:
            if direction == "minimize":
                best_so_far = min(best_so_far, trial.score)
            else:
                best_so_far = max(best_so_far, trial.score)
        history.append(best_so_far if best_so_far != float("inf") and best_so_far != float("-inf") else None)
    
    return TuningResult(
        best_config=study.best_params,
        best_score=study.best_value,
        best_trial_id=study.best_trial.number,
        trials=trials,
        n_trials=len(study.trials),
        n_completed=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        n_pruned=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        optimization_history=[h for h in history if h is not None],
        metadata={
            "study_name": study.study_name,
            "direction": direction,
            "sampler": sampler,
        },
    )


# =============================================================================
# Factory Function
# =============================================================================


def create_search_space(config: Dict[str, Dict[str, Any]]) -> SearchSpace:
    """
    Create search space from configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration mapping parameter names to specifications.
        
        Each specification can have:
        - type: "float", "int", "categorical", "bool"
        - low, high: for float/int
        - choices: for categorical
        - log: bool for log-scale sampling
        
    Returns
    -------
    SearchSpace
        Configured search space.
        
    Example:
        config = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "hidden_units": {"type": "int", "low": 32, "high": 256},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
            "use_dropout": {"type": "bool"},
        }
        space = create_search_space(config)
    """
    return SearchSpace.from_dict(config)


__all__ = [
    # Classes
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
