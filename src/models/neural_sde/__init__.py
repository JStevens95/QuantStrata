"""
Neural Stochastic Differential Equations (Neural SDE).

Implements neural networks that learn drift and diffusion functions
from data, enabling more realistic market simulation than parametric
models like GBM or Heston.

Research Foundation:
- Kidger et al. (2021) "Neural SDEs"
- Gierjatowicz et al. (2020) "Robust pricing and hedging via neural SDEs"

Components:
- networks: Neural drift and diffusion architectures
- solvers: SDE solvers (Euler-Maruyama, Milstein)
- dynamics: NeuralSDEDynamics for simulation
- training: Training pipelines (score matching, calibration)
- generation: Conditional generation and augmentation

Example:
    from src.models.neural_sde import (
        NeuralSDEDynamics,
        NeuralDriftNetwork,
        NeuralDiffusionNetwork,
        EulerMaruyamaSolver,
    )
    
    # Create neural SDE
    drift_net = NeuralDriftNetwork(hidden_dims=[64, 64])
    diffusion_net = NeuralDiffusionNetwork(hidden_dims=[64, 64])
    
    sde = NeuralSDEDynamics(
        drift_network=drift_net,
        diffusion_network=diffusion_net,
    )
    
    # Simulate paths
    paths = sde.simulate(S0=100.0, T=1.0, n_steps=252, n_paths=10000)
"""

from src.models.neural_sde.networks import (
    NeuralDriftNetwork,
    NeuralDiffusionNetwork,
)
from src.models.neural_sde.solvers import (
    EulerMaruyamaSolver,
    MilsteinSolver,
    SDESolver,
)
from src.models.neural_sde.dynamics import NeuralSDEDynamics

__all__ = [
    # Networks
    "NeuralDriftNetwork",
    "NeuralDiffusionNetwork",
    # Solvers
    "EulerMaruyamaSolver",
    "MilsteinSolver",
    "SDESolver",
    # Dynamics
    "NeuralSDEDynamics",
]
