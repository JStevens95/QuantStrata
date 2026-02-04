# Hyperparameter Tuning with Optuna

This guide shows how to define search spaces, run Optuna-based tuning with pruning, and plug results into the model registry. Full API: [Production ML Reference](../../reference/machine_learning/production_ml.md).

---

## 1. Define a search space

Use `SearchSpace` to declare parameters and sample or export for Optuna:

```python
from src.machine_learning.tuning import SearchSpace

space = (
    SearchSpace()
    .add_float("learning_rate", 1e-4, 1e-2, log=True)
    .add_int("hidden_units", 32, 256)
    .add_categorical("activation", ["relu", "tanh", "gelu"])
    .add_bool("use_dropout")
)

# One random sample (e.g. for testing)
config = space.sample(seed=42)
```

---

## 2. Objective and pruning

Your objective receives a sampled `config` and an Optuna `trial`; report intermediate values so pruners can stop bad trials early:

```python
import optuna
from src.machine_learning.tuning import run_optuna_tuning, SearchSpace, MedianPruner

def objective(config, trial):
    model = build_model(**config)
    for epoch in range(100):
        loss = train_one_epoch(model)
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return evaluate(model)

space = SearchSpace().add_float("learning_rate", 1e-4, 1e-2, log=True).add_int("hidden_units", 32, 256)

result = run_optuna_tuning(
    objective_fn=objective,
    search_space=space,
    n_trials=50,
    direction="minimize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    seed=42,
)

print("Best config:", result.best_config)
print("Best score:", result.best_score)
```

---

## 3. Using the orchestrator pipeline

For a standardised workflow, use the `ml.hyperparameter_tuning` pipeline and pass the search space and objective via config:

```python
from src.orchestrator.pipelines.ml.hyperparameter_tuning import (
    create_hyperparameter_tuning_pipeline,
    HyperparameterTuningConfig,
)

config = HyperparameterTuningConfig(
    search_space_config={...},  # dict form of SearchSpace
    n_trials=30,
    direction="minimize",
    study_name="gnn_pricer_tuning",
)

pipeline = create_hyperparameter_tuning_pipeline(config)
# Run with PipelineRunner and Context; best config and result in state/artifacts
```

See the [Production ML Reference](../../reference/machine_learning/production_ml.md) for `TuningResult` fields and [run_hyperparameter_tuning.py](../../../examples/pipelines/run_hyperparameter_tuning.py) for a full example.

---

## 4. Pruners

| Pruner | Use case |
|--------|----------|
| `MedianPruner` | Stop trial if intermediate value is worse than median of others |
| `PercentilePruner` | Same idea with a configurable percentile (e.g. 25th) |

Use `n_startup_trials` and `n_warmup_steps` so early trials and steps are not pruned. See [Production ML Reference](../../reference/machine_learning/production_ml.md#pruners).

---

## 5. After tuning

- **Best config:** `result.best_config` — use to train a final model or register it.
- **Full history:** `result.trials`, `result.optimization_history` — for analysis or plotting.
- **Model registry:** Log the best run and register the model with the chosen experiment tracker and `ModelRegistry`; see [Production ML Reference](../../reference/machine_learning/production_ml.md#model-registry).
