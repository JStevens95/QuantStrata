# Production ML Infrastructure Reference

Technical specification and API reference for the production ML infrastructure.

## Overview

The production ML infrastructure provides enterprise-grade tooling for:
- **Experiment Tracking**: Unified interface for MLflow, Weights & Biases, and in-memory tracking
- **Hyperparameter Tuning**: Bayesian optimization with Optuna integration
- **Model Registry**: Version control and lifecycle management for ML models

---

## Experiment Tracking

### Module: `src.machine_learning.core.tracking`

### ExperimentTracker Protocol

```python
class ExperimentTracker(Protocol):
    """Protocol for experiment tracking backends."""
    
    @property
    def experiment_name(self) -> str: ...
    
    @property
    def current_run(self) -> Optional[RunInfo]: ...
    
    def start_run(self, run_name: str = None, tags: dict = None) -> ExperimentTracker: ...
    def end_run(self, status: str = "FINISHED") -> None: ...
    def log_params(self, params: Dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None: ...
    def log_artifact(self, local_path: Path, artifact_path: str = None) -> None: ...
    def set_tags(self, tags: Dict[str, str]) -> None: ...
```

### InMemoryTracker

In-memory tracker for testing and local development.

```python
from src.machine_learning.core.tracking import InMemoryTracker

tracker = InMemoryTracker(experiment_name="my_experiment")

with tracker.start_run("training_run") as run:
    tracker.log_params({"learning_rate": 0.001})
    tracker.log_metrics({"loss": 0.5}, step=1)
    
# Query runs
runs = tracker.get_all_runs()
best = tracker.get_best_run("loss", minimize=True)
```

### MLflowTracker

Production-grade MLflow integration.

```python
from src.machine_learning.core.tracking import MLflowTracker

tracker = MLflowTracker(
    experiment_name="option_pricer",
    tracking_uri="http://localhost:5000",
)

with tracker.start_run("training_v1"):
    tracker.log_params({"hidden_units": 128})
    tracker.log_metrics({"mse": 0.001})
    tracker.log_artifact("model.h5")
```

### WandBTracker

Weights & Biases integration with rich visualizations.

```python
from src.machine_learning.core.tracking import WandBTracker

tracker = WandBTracker(project="option_pricer", entity="my_team")

with tracker.start_run("training_v1", config={"lr": 0.001}):
    tracker.log_metrics({"loss": 0.5})
```

### Factory Function

```python
from src.machine_learning.core.tracking import create_tracker

# In-memory (default)
tracker = create_tracker()

# MLflow
tracker = create_tracker("mlflow", experiment_name="exp", tracking_uri="...")

# W&B
tracker = create_tracker("wandb", experiment_name="project", entity="team")
```

---

## Hyperparameter Tuning

### Module: `src.machine_learning.tuning.search_space`

### SearchSpace

Define hyperparameter search spaces with a fluent API.

```python
from src.machine_learning.tuning import SearchSpace

space = (
    SearchSpace()
    .add_float("learning_rate", 1e-4, 1e-2, log=True)
    .add_int("hidden_units", 32, 256)
    .add_categorical("activation", ["relu", "tanh", "gelu"])
    .add_bool("use_dropout")
)

# Sample configuration
config = space.sample(seed=42)

# Export/import
d = space.to_dict()
space2 = SearchSpace.from_dict(d)
```

### Parameter Types

| Type | Method | Parameters |
|------|--------|------------|
| Float | `add_float(name, low, high, log=False, step=None)` | Continuous value |
| Integer | `add_int(name, low, high, log=False, step=1)` | Discrete integer |
| Categorical | `add_categorical(name, choices)` | Discrete choice |
| Boolean | `add_bool(name)` | True/False |

### Pruners

Early stopping for unpromising trials.

```python
from src.machine_learning.tuning import MedianPruner, PercentilePruner

# Prune trials below median
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

# Prune trials below 25th percentile
pruner = PercentilePruner(percentile=25.0)
```

### Optuna Tuning

```python
from src.machine_learning.tuning import run_optuna_tuning, SearchSpace, MedianPruner

def objective(config, trial):
    model = create_model(**config)
    for epoch in range(100):
        loss = train_epoch(model)
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return loss

space = SearchSpace().add_float("lr", 1e-4, 1e-2, log=True)

result = run_optuna_tuning(
    objective_fn=objective,
    search_space=space,
    n_trials=100,
    direction="minimize",
    pruner=MedianPruner(),
)

print(f"Best config: {result.best_config}")
print(f"Best score: {result.best_score}")
```

### TuningResult

```python
@dataclass
class TuningResult:
    best_config: Dict[str, Any]
    best_score: float
    best_trial_id: int
    trials: List[TuningTrial]
    n_trials: int
    n_completed: int
    n_pruned: int
    optimization_history: List[float]
    
    def to_dict(self) -> Dict[str, Any]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "TuningResult": ...
```

---

## Model Registry

### Module: `src.machine_learning.registry.registry`

### ModelStage

```python
class ModelStage(Enum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
```

### ModelRegistry

Central registry for ML model versioning and lifecycle management.

```python
from src.machine_learning.registry import ModelRegistry, ModelStage

# Initialize registry
registry = ModelRegistry("./model_registry")

# Register a new model version
version = registry.register_model(
    name="gnn_pricer",
    model_path="./trained_models/gnn_v1",
    metrics={"mse": 0.001, "r2": 0.99},
    params={"hidden_units": 128},
    tags={"asset_class": "fx"},
)

# Promote to production
registry.transition_stage(
    name="gnn_pricer",
    version=version.version,
    stage=ModelStage.PRODUCTION,
)

# Load production model
artifact = registry.get_model("gnn_pricer", stage=ModelStage.PRODUCTION)
model = artifact.load()
```

### API Methods

| Method | Description |
|--------|-------------|
| `register_model(name, model_path, metrics, params, tags, description)` | Register new version |
| `get_model(name, version=None, stage=None)` | Get model artifact |
| `transition_stage(name, version, stage, archive_existing)` | Change lifecycle stage |
| `list_models()` | List all registered models |
| `list_versions(name, stage=None)` | List versions of a model |
| `delete_version(name, version)` | Delete a version |
| `update_tags(name, version, tags)` | Update version tags |
| `search_models(name_contains, tags, stage, metric_filter)` | Search models |

### ModelArtifact

```python
@dataclass
class ModelArtifact:
    name: str
    version: ModelVersion
    registry_path: Path
    
    @property
    def artifact_dir(self) -> Path: ...
    
    def load(self, custom_objects: dict = None) -> Any:
        """Load model from artifacts."""
        # Supports: SavedModel, .keras, .h5, .pt, .pth, .pkl
        ...
    
    def verify_integrity(self) -> bool:
        """Verify model against stored hash."""
        ...
```

---

## Integration Example

```python
from src.machine_learning.core.tracking import create_tracker
from src.machine_learning.tuning import SearchSpace, run_optuna_tuning
from src.machine_learning.registry import ModelRegistry, ModelStage

# Setup tracking
tracker = create_tracker("mlflow", experiment_name="option_pricer_tuning")

# Define search space
space = (
    SearchSpace()
    .add_float("lr", 1e-4, 1e-2, log=True)
    .add_int("hidden", 32, 256)
)

# Objective with tracking
def objective(config, trial):
    with tracker.start_run(f"trial_{trial.number}"):
        tracker.log_params(config)
        
        model = create_model(**config)
        for epoch in range(50):
            loss = train_epoch(model)
            tracker.log_metrics({"loss": loss}, step=epoch)
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return loss

# Run tuning
result = run_optuna_tuning(objective, space, n_trials=50)

# Register best model
registry = ModelRegistry("./models")
version = registry.register_model(
    name="option_pricer",
    model_path="./best_model",
    metrics={"loss": result.best_score},
    params=result.best_config,
)

# Promote to production
registry.transition_stage("option_pricer", version.version, ModelStage.PRODUCTION)
```
