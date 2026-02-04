# Neural SDE Reference

Technical specification and API reference for Neural Stochastic Differential Equations: learning drift and diffusion from data for market simulation and scenario generation.

---

## Overview

**Neural SDEs** model the underlying as:

\[
dS_t = \mu_\theta(S_t, t)\,dt + \sigma_\theta(S_t, t)\,dW_t
\]

where \(\mu_\theta\) and \(\sigma_\theta\) are neural networks. This allows:

- **Data-driven dynamics** instead of fixed parametric forms (e.g. GBM, Heston)
- **Path generation** for pricing, stress testing, and data augmentation
- **Calibration** to historical returns or option surfaces (via training losses)

**Research references:** Kidger et al. (2021) "Neural SDEs"; Gierjatowicz et al. (2020) "Robust pricing and hedging via neural SDEs".

---

## 1. Components

| Component | Module | Description |
|-----------|--------|-------------|
| **Networks** | `src.models.neural_sde.networks` | `NeuralDriftNetwork`, `NeuralDiffusionNetwork` (MLP, state + time input) |
| **Solvers** | `src.models.neural_sde.solvers` | `EulerMaruyamaSolver`, `MilsteinSolver`, `SDESolver` protocol |
| **Dynamics** | `src.models.neural_sde.dynamics` | `NeuralSDEDynamics` (combines networks + solver, `simulate`, `calibrate`) |
| **Training** | `src.models.neural_sde.training` | `NeuralSDETrainer`, `TrainingConfig`, `TrainingResult`; losses in `losses.py` |
| **Generation** | `src.models.neural_sde.generation` | `PathGenerator` (unconditional/conditioned path generation) |

---

## 2. Networks

**Module:** `src.models.neural_sde.networks`

### NeuralDriftNetwork

- **Inputs:** state \(S\) (or batch), time \(t\) (or batch)
- **Output:** scalar (or vector) drift \(\mu\)
- **Config:** `NetworkConfig`: `hidden_dims`, `activation`, `normalize_inputs`, `S_mean`, `S_std`

### NeuralDiffusionNetwork

- **Inputs:** state \(S\), time \(t\)
- **Output:** positive diffusion \(\sigma\) (e.g. softplus to enforce positivity)
- **Config:** same as drift; optional `min_vol`, `max_vol` in dynamics config

---

## 3. Solvers

**Module:** `src.models.neural_sde.solvers`

| Class | Description |
|-------|-------------|
| `SDESolver` | Protocol: `solve(drift, diffusion, S0, t_grid, n_paths, rng)` |
| `EulerMaruyamaSolver` | Euler-Maruyama scheme |
| `MilsteinSolver` | Milstein scheme (when applicable) |

---

## 4. Dynamics

**Module:** `src.models.neural_sde.dynamics`

### NeuralSDEDynamics

- **Construction:** `NeuralSDEDynamics(drift_network=..., diffusion_network=..., solver=..., config=...)` or from `NeuralSDEConfig`
- **Methods:**
  - `simulate(S0, T, n_steps, n_paths, seed)` → path array shape `(n_paths, n_steps+1)` or `(n_paths, n_steps+1, 1)`
  - `calibrate(historical_paths, config)` → fits drift/diffusion to historical data (via trainer)
- **Config:** `NeuralSDEConfig`: `drift_hidden_dims`, `diffusion_hidden_dims`, `solver_type`, `min_vol`, `max_vol`, `S_mean`, `S_std`

---

## 5. Training

**Module:** `src.models.neural_sde.training`

### NeuralSDETrainer

- **Config:** `TrainingConfig`: `n_epochs`, `learning_rate`, `batch_size`, `moment_weight`, `pathwise_weight`, `n_sim_paths`, `n_sim_steps`, `patience`, `verbose`
- **Losses:** `MomentMatchingLoss`, `PathwiseLoss` (in `losses.py`)
- **Method:** `fit(sde, historical_paths)` → `TrainingResult` (`final_loss`, `loss_history`, `epoch`, `converged`)

### TrainingResult

- `final_loss`, `loss_history`, `epoch`, `converged`
- Optional: `moment_losses`, `pathwise_losses`
- `summary()` → dict

---

## 6. Generation

**Module:** `src.models.neural_sde.generation` (e.g. `generator.py`)

### PathGenerator

- **Constructor:** `PathGenerator(model, seed)` where `model` is a `NeuralSDEDynamics` instance
- **Methods:**
  - `generate(S0, T, n_steps, n_paths)` → unconditional paths
  - `generate_conditioned(S0, S_T, T, n_paths, ...)` → paths with pinned terminal value (if implemented)

---

## 7. Top-Level Imports

**Module:** `src.models.neural_sde`

```python
from src.models.neural_sde import (
    NeuralSDEDynamics,
    NeuralDriftNetwork,
    NeuralDiffusionNetwork,
    EulerMaruyamaSolver,
    MilsteinSolver,
    SDESolver,
)
from src.models.neural_sde.training import NeuralSDETrainer, TrainingConfig, TrainingResult
from src.models.neural_sde.generation.generator import PathGenerator
```

---

## 8. Dependencies

- **External:** NumPy (core); optional TensorFlow/JAX for GPU training
- **Internal:** None (models layer)

---

## 9. Gaps / Extensions

- **Adjoint method:** Memory-efficient backpropagation through the SDE solver (not yet implemented)
- **Score matching / calibration to options:** Dedicated losses and calibrator (optional extensions)
- **MC pricer / Greeks:** Pricer using Neural SDE paths and Greeks via differentiation (optional)
- **Conditional / augmentation:** Additional generators in `generation/` (e.g. regime conditioning)

---

*See also: [Training Neural SDE (guide)](../../guides/models/training_neural_sde.md), roadmap Phase 7.7.*
