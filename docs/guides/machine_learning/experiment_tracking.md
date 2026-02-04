# Experiment Tracking with Production ML

This guide shows how to use the library’s experiment tracking (MLflow, Weights & Biases, or in-memory) to log runs, parameters, metrics, and artifacts during model training.

---

## When to use

- **Local / tests:** `InMemoryTracker` — no server, query runs in process.
- **Team / production:** `MLflowTracker` or `WandBTracker` — central UI, history, and comparison.

Full API and options: [Production ML Reference](../../reference/machine_learning/production_ml.md).

---

## 1. In-memory tracking (default)

Use for scripts, tests, or when you don’t need a server.

```python
from src.machine_learning.core.tracking import InMemoryTracker

tracker = InMemoryTracker(experiment_name="my_experiment")

with tracker.start_run("run_1"):
    tracker.log_params({"learning_rate": 0.001, "hidden_units": 128})
    tracker.log_metrics({"loss": 0.5, "val_loss": 0.6}, step=0)
    # ... train ...
    tracker.log_metrics({"loss": 0.1, "val_loss": 0.15}, step=100)
    tracker.log_artifact("model_weights.h5")

# After the run
runs = tracker.get_all_runs()
best = tracker.get_best_run("val_loss", minimize=True)
print("Best config:", best.params)
```

---

## 2. MLflow

Use for shared experiments and a central UI.

**Start MLflow server (optional):**
```bash
mlflow server --host 0.0.0.0 --port 5000
```

**In code:**
```python
from src.machine_learning.core.tracking import MLflowTracker

tracker = MLflowTracker(
    experiment_name="option_pricer",
    tracking_uri="http://localhost:5000",
)

with tracker.start_run("training_v1"):
    tracker.log_params({"hidden_units": 128, "activation": "relu"})
    for epoch in range(100):
        loss = train_epoch()
        tracker.log_metrics({"loss": loss}, step=epoch)
    tracker.log_artifact("saved_model/")
```

View runs at `http://localhost:5000`.

---

## 3. Weights & Biases

Use for rich dashboards and team collaboration.

```python
from src.machine_learning.core.tracking import WandBTracker

tracker = WandBTracker(project="option_pricer", entity="my_team")

with tracker.start_run("run_1", config={"lr": 0.001}):
    tracker.log_metrics({"loss": 0.5})
    tracker.log_artifact("model.h5")
```

---

## 4. Factory and integration with training

Use `create_tracker()` to choose backend by name and wire it into your training loop:

```python
from src.machine_learning.core.tracking import create_tracker

tracker = create_tracker("memory", experiment_name="exp")
# or: create_tracker("mlflow", experiment_name="exp", tracking_uri="...")
# or: create_tracker("wandb", experiment_name="project", entity="team")

with tracker.start_run("training"):
    tracker.log_params(config)
    for epoch in range(n_epochs):
        metrics = train_and_evaluate(epoch)
        tracker.log_metrics(metrics, step=epoch)
```

For pipeline-based runs (e.g. hyperparameter tuning), the orchestrator pipeline can accept a tracker in context and log metrics per trial; see [Hyperparameter Tuning](hyperparameter_tuning.md) and the [Production ML Reference](../../reference/machine_learning/production_ml.md).
