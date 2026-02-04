# How to Train and Use Neural SDEs

This guide shows how to build, train, and use Neural SDE models for path simulation and scenario generation.

---

## 1. Build a Neural SDE

```python
from src.models.neural_sde import (
    NeuralSDEDynamics,
    NeuralDriftNetwork,
    NeuralDiffusionNetwork,
    EulerMaruyamaSolver,
)

# Optional: custom config
drift_net = NeuralDriftNetwork(hidden_dims=[64, 64])
diffusion_net = NeuralDiffusionNetwork(hidden_dims=[64, 64])
solver = EulerMaruyamaSolver()

sde = NeuralSDEDynamics(
    drift_network=drift_net,
    diffusion_network=diffusion_net,
    solver=solver,
)
```

You can also use `NeuralSDEConfig` to build networks and solver consistently; see `src.models.neural_sde.dynamics.NeuralSDEConfig`.

---

## 2. Simulate Paths (Untrained)

You can simulate paths with default (random) weights to check the pipeline:

```python
import numpy as np

paths = sde.simulate(S0=100.0, T=1.0, n_steps=252, n_paths=10_000, seed=42)
# paths shape: (n_paths, n_steps+1) or (n_paths, n_steps+1, 1)
```

---

## 3. Train on Historical Data

Prepare historical paths: shape `(n_samples, n_steps+1)` (e.g. daily prices or returns).

```python
from src.models.neural_sde.training import NeuralSDETrainer, TrainingConfig, TrainingResult

config = TrainingConfig(
    n_epochs=100,
    learning_rate=1e-3,
    batch_size=32,
    n_sim_paths=1000,
    n_sim_steps=50,
    patience=10,
    verbose=True,
)
trainer = NeuralSDETrainer(config=config)

result = trainer.fit(sde, historical_paths)
print(result.summary())
print(f"Final loss: {result.final_loss:.6f}, converged: {result.converged}")
```

---

## 4. Generate Paths After Training

Use the trained `sde` with `PathGenerator`:

```python
from src.models.neural_sde.generation.generator import PathGenerator

generator = PathGenerator(sde, seed=123)
paths = generator.generate(S0=100.0, T=1.0, n_steps=252, n_paths=5_000)
```

Use these paths for scenario analysis, pricing (e.g. MC with existing payoffs), or data augmentation.

---

## 5. Integration with Pipelines

For orchestrated runs (config, logging, artifacts), use the Neural SDE pipeline:

- **Pipeline:** `src.orchestrator.pipelines.ml.train_neural_sde` (if implemented)
- **Example script:** `examples/pipelines/run_train_neural_sde.py`

See the pipeline and example for YAML/JSON config and CLI usage.

---

## 6. Tips

- **Data:** Use consistent time grid (e.g. daily) and normalise spot if needed (config `S_mean`, `S_std`).
- **Stability:** Enforce `min_vol` / `max_vol` in dynamics config to avoid numerical issues.
- **Training:** Start with small `n_epochs` and `n_sim_paths`; increase for better fit.
- **Validation:** Compare simulated distribution (e.g. terminal distribution, moments) with historical.

---

*Reference: [Neural SDE (reference)](../../reference/models/neural_sde.md).*
