# Machine Learning Framework Guide

User guide for the QuantStrata ML integration (Phase 7.1): how to prepare data, train models, evaluate, and run inference.

---

## When to use what

| Goal | Entry point | Data | Docs / tutorial |
|------|-------------|------|------------------|
| Train a Keras model (e.g. NN pricer) with a generic loop | `Trainer` or `run_training(KerasTrainableAdapter(model), ...)` | `build_pricing_data()` or `create_pricing_dataset()` | ML lifecycle notebook |
| Train the Hybrid GNN-LSTM pricer | `TrainingManager` + `TrainingConfiguration` | `build_gnn_data(use_synthetic=True)` or FX | Hybrid GNN-LSTM tutorial |
| Build pricing data from MC or analytic pricer | `build_pricing_dataset_from_mc()` / `build_pricing_dataset_from_analytic()` | — | Reference: ml_framework.md |
| Build calibration dataset (IV → params) | `build_calibration_dataset()` | — | Reference: ml_framework.md |
| Save/load and predict (generic) | `save_model()`, `load_model()`, `predict()` in `pipelines.inference` | — | Reference: ml_framework.md |
| Save/load Keras / Hybrid GNN | `inference.model_io` or `model.save_weights()` / `load_weights()` | — | Hybrid GNN-LSTM tutorial |

---

## Quick start: Hybrid GNN-LSTM

```python
from src.machine_learning.models.gnn_rnn_hybrid import HybridGnnRnn, default_hybrid_model_config
from src.machine_learning.data.gnn_rnn_hybrid import build_gnn_data
from src.machine_learning.calibration.training_manager import TrainingManager, TrainingConfiguration

n_targets = 10
model_config = default_hybrid_model_config(n_targets=n_targets)
data = build_gnn_data(use_synthetic=True, n_targets=n_targets, n_samples=500, batch_size=32)
training_config = TrainingConfiguration(
    name="hybrid_pnl", model="HybridGnnRnn", epochs=30, batch_size=32,
    model_dir="./checkpoints", early_stopping=True,
)
manager = TrainingManager(data.train_ds, model_config, validation_ds=data.val_ds)
manager.run(stages=[training_config])
# Predict: manager.model(batch_inputs)
# Save: manager.model.save_weights("path.weights.h5")
```

---

## Quick start: Pricing dataset and training

```python
from src.machine_learning.data.pricing import build_pricing_data
from src.machine_learning.training.trainer import Trainer  # or pipelines.training.run_training

data = build_pricing_data(n_samples=10_000, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
# Build a Keras model, then:
# trainer = Trainer(model, train_ds=data.train_ds, val_ds=data.val_ds, ...)
# trainer.fit(epochs=100)
```

---

## Tutorials and reference

- **Tutorials:** [ML Model Lifecycle](../tutorials/machine_learning/ml_model_lifecycle.ipynb), [Hybrid GNN-LSTM](../tutorials/machine_learning/hybrid_gnn_lstm_tutorial.ipynb) (config → data → model → training → evaluation → deployment).
- **Technical reference:** [ML Framework Reference](../reference/machine_learning/ml_framework.md).
- **Progress and alignment:** [Phase 7.1 implementation notes](../development/progress/phase_7_1_implementation_notes.md).
