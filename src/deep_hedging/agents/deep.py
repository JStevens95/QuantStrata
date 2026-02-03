"""
Deep Hedging Agent

Neural network-based hedging agent that learns optimal hedging policies
by minimising a risk measure over the P&L distribution.

Architecture
------------
The agent uses a feedforward neural network (MLP) as the policy:

    π_θ(s) = W_L · φ(W_{L-1} · φ(...φ(W_1 · s + b_1)...) + b_{L-1}) + b_L

where φ is an activation function (ReLU, tanh, etc.).

Input features (from HedgingState):
- Log-moneyness: log(S/K)
- Time to maturity: τ
- Current position: δ_{t-1}
- Running P&L (normalised)
- Optional: BSM delta, gamma, vega

Output:
- Hedge position δ_t (continuous, optionally clipped)

Training
--------
The agent is trained by:
1. Simulating many hedging episodes
2. Computing terminal P&L for each episode
3. Computing risk measure: ρ(-P&L)
4. Backpropagating through the policy

Loss = ρ(-P&L) = risk measure applied to negative P&L samples

This module provides a NumPy-based implementation for clarity. For production
training with GPU acceleration, use the TensorFlow/PyTorch adapters.

References
----------
- Bühler et al. (2019) "Deep Hedging"
- docs/reference/deep_hedging/theory.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.deep_hedging.core.types import HedgingState, DeepHedgingTrainingConfig
from src.deep_hedging.core.risk_measures import RiskMeasure, MeanVarianceRisk


# =============================================================================
# Activation Functions
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU activation."""
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))


def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ELU."""
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "elu": (elu, elu_derivative),
}


# =============================================================================
# MLP Policy Network
# =============================================================================

@dataclass
class MLPPolicy:
    """
    Multi-layer perceptron policy network.
    
    This is a simple feedforward neural network implemented in NumPy.
    For production use with GPU, wrap this with TensorFlow/PyTorch.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : list of int
        Number of units in each hidden layer.
    output_dim : int
        Number of outputs (typically 1 for hedge position).
    activation : str
        Activation function: "relu", "tanh", "sigmoid", "elu".
    output_activation : str or None
        Activation for output layer. None = linear.
    output_scale : float
        Scale factor for output (e.g., 2.0 to allow positions in [-2, 2]).
    init_scale : float
        Scale for weight initialisation.
    
    Attributes
    ----------
    weights : list of ndarray
        Weight matrices W_1, ..., W_L.
    biases : list of ndarray
        Bias vectors b_1, ..., b_L.
    """
    
    input_dim: int
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    output_dim: int = 1
    activation: str = "relu"
    output_activation: Optional[str] = "tanh"
    output_scale: float = 1.0
    init_scale: float = 0.1
    
    # Network parameters (initialised in __post_init__)
    weights: List[np.ndarray] = field(default_factory=list, repr=False)
    biases: List[np.ndarray] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        """Initialize network parameters."""
        if not self.weights:
            self._initialize_parameters()
    
    def _initialize_parameters(self, seed: Optional[int] = None) -> None:
        """
        Initialize weights using Xavier/He initialization.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        
        self.weights = []
        self.biases = []
        
        # Build layer dimensions: [input, hidden1, hidden2, ..., output]
        layer_dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            
            # Xavier/He initialization
            if self.activation == "relu":
                # He initialization for ReLU
                std = np.sqrt(2.0 / fan_in)
            else:
                # Xavier initialization
                std = np.sqrt(2.0 / (fan_in + fan_out))
            
            W = rng.normal(0, std * self.init_scale, (fan_in, fan_out))
            b = np.zeros(fan_out)
            
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, input_dim) or (input_dim,)
            Input features.
        
        Returns
        -------
        ndarray, shape (batch_size, output_dim) or (output_dim,)
            Network output (hedge position).
        """
        # Handle single sample
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        
        # Get activation function
        act_fn, _ = ACTIVATIONS.get(self.activation, ACTIVATIONS["relu"])
        
        # Forward through hidden layers
        h = x
        for i in range(len(self.weights) - 1):
            h = h @ self.weights[i] + self.biases[i]
            h = act_fn(h)
        
        # Output layer
        out = h @ self.weights[-1] + self.biases[-1]
        
        # Output activation
        if self.output_activation is not None:
            out_act_fn, _ = ACTIVATIONS.get(self.output_activation, ACTIVATIONS["tanh"])
            out = out_act_fn(out)
        
        # Scale output
        out = out * self.output_scale
        
        if single:
            out = out.ravel()
        
        return out
    
    def forward_with_cache(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Forward pass with cached activations for backpropagation.
        
        Returns
        -------
        output : ndarray
            Network output.
        cache : list of (pre_activation, post_activation) tuples
            Cached values for each layer.
        """
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        
        act_fn, _ = ACTIVATIONS.get(self.activation, ACTIVATIONS["relu"])
        cache = []
        
        h = x
        for i in range(len(self.weights) - 1):
            z = h @ self.weights[i] + self.biases[i]
            h = act_fn(z)
            cache.append((z, h))
        
        # Output layer
        z_out = h @ self.weights[-1] + self.biases[-1]
        
        if self.output_activation is not None:
            out_act_fn, _ = ACTIVATIONS.get(self.output_activation, ACTIVATIONS["tanh"])
            out = out_act_fn(z_out) * self.output_scale
        else:
            out = z_out * self.output_scale
        
        cache.append((z_out, out))
        
        if single:
            out = out.ravel()
        
        return out, cache
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get network parameters as a dictionary."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "output_scale": self.output_scale,
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set network parameters from a dictionary."""
        if "weights" in params:
            self.weights = [np.array(w) for w in params["weights"]]
        if "biases" in params:
            self.biases = [np.array(b) for b in params["biases"]]
    
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        n = 0
        for w, b in zip(self.weights, self.biases):
            n += w.size + b.size
        return n


# =============================================================================
# Deep Hedging Agent
# =============================================================================

@dataclass
class DeepHedgingAgent:
    """
    Deep hedging agent with neural network policy.
    
    This agent learns a hedging policy π_θ(s) → δ by minimising a risk
    measure over the terminal P&L distribution.
    
    The policy takes state features (spot, time, position, Greeks, etc.)
    and outputs the optimal hedge position.
    
    Parameters
    ----------
    policy : MLPPolicy
        Neural network policy.
    risk_measure : RiskMeasure
        Risk measure for training objective.
    learning_rate : float
        SGD/Adam learning rate.
    include_greeks : bool
        Whether to include BSM Greeks in state features.
    normalise_features : bool
        Whether to normalise input features.
    clip_actions : tuple of (float, float), optional
        Clip actions to this range.
    
    Training
    --------
    The agent is trained by simulating hedging episodes and minimising:
    
        Loss = ρ(-P&L) where ρ is the risk measure
    
    Gradients are computed via backpropagation through the simulation.
    
    Example
    -------
    >>> policy = MLPPolicy(input_dim=7, hidden_layers=[64, 64])
    >>> agent = DeepHedgingAgent(policy=policy)
    >>> 
    >>> # Training loop (simplified)
    >>> for epoch in range(100):
    ...     pnl_samples = simulate_episodes(agent, env, n_paths=256)
    ...     loss, grads = agent.compute_loss_and_gradients(pnl_samples)
    ...     agent.update_parameters(grads)
    """
    
    policy: MLPPolicy
    risk_measure: RiskMeasure = field(default_factory=lambda: MeanVarianceRisk(risk_aversion=0.5))
    learning_rate: float = 0.001
    include_greeks: bool = True
    normalise_features: bool = True
    clip_actions: Optional[Tuple[float, float]] = None
    
    # Training state
    _step_count: int = field(default=0, repr=False)
    _adam_m: Optional[List[np.ndarray]] = field(default=None, repr=False)
    _adam_v: Optional[List[np.ndarray]] = field(default=None, repr=False)
    
    @property
    def name(self) -> str:
        """Agent name."""
        return "DeepHedging"
    
    def select_action(
        self,
        state: Union[HedgingState, np.ndarray, Dict[str, Any]],
        *,
        training: bool = False,
        explore: bool = False,
    ) -> float:
        """
        Select action (hedge position) using the policy network.
        
        Parameters
        ----------
        state : HedgingState or ndarray or dict
            Current hedging state.
        training : bool
            If True, may add exploration noise.
        explore : bool
            If True, add Gaussian noise for exploration.
        
        Returns
        -------
        float
            Hedge position.
        """
        # Convert state to feature array
        features = self._state_to_features(state)
        
        # Forward pass through policy
        action = self.policy.forward(features)
        
        # Handle output shape
        if isinstance(action, np.ndarray):
            action = float(action.item() if action.size == 1 else action[0])
        
        # Optional exploration noise during training
        if training and explore:
            noise_std = 0.1 * max(1.0 - self._step_count / 10000, 0.01)
            action += np.random.normal(0, noise_std)
        
        # Clip actions
        if self.clip_actions is not None:
            action = np.clip(action, self.clip_actions[0], self.clip_actions[1])
        
        return float(action)
    
    def _state_to_features(
        self,
        state: Union[HedgingState, np.ndarray, Dict[str, Any]],
    ) -> np.ndarray:
        """Convert state to feature array for the policy network."""
        if isinstance(state, HedgingState):
            return state.to_array(
                include_greeks=self.include_greeks,
                normalise=self.normalise_features,
            )
        elif isinstance(state, np.ndarray):
            return state.astype(np.float32)
        elif isinstance(state, dict):
            # Build feature array from dict
            features = []
            if self.normalise_features and "strike" in state:
                features.append(np.log(state["spot"] / state["strike"]))
            else:
                features.append(state.get("spot", 0.0))
            features.append(state.get("time_to_maturity", 0.0))
            features.append(state.get("position", 0.0))
            features.append(state.get("pnl", 0.0))
            if self.include_greeks:
                features.append(state.get("delta_bs", 0.0))
                features.append(state.get("gamma_bs", 0.0))
                features.append(state.get("vega_bs", 0.0))
            return np.array(features, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
    
    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Update agent from transitions (for compatibility with RL pipeline).
        
        For deep hedging, we typically use batch training over full episodes
        rather than transition-by-transition updates. This method provides
        compatibility with the RLAgent protocol.
        
        Parameters
        ----------
        transitions : list, optional
            List of transitions (not typically used for deep hedging).
        batch : dict, optional
            Batch data with 'pnl_samples' and optionally gradients.
        
        Returns
        -------
        dict or None
            Training metrics.
        """
        if batch is None:
            return None
        
        # Expect batch to contain pre-computed gradients or loss
        if "gradients" in batch:
            self._apply_gradients(batch["gradients"])
            self._step_count += 1
            return {"loss": batch.get("loss", 0.0)}
        
        return None
    
    def compute_loss(self, pnl_samples: np.ndarray) -> float:
        """
        Compute risk measure loss from P&L samples.
        
        Parameters
        ----------
        pnl_samples : ndarray, shape (n_samples,)
            Terminal P&L for each episode.
        
        Returns
        -------
        float
            Risk measure value (lower is better).
        """
        # Risk measure is applied to losses (negative P&L)
        return self.risk_measure.compute(-pnl_samples)
    
    def _apply_gradients(
        self,
        gradients: Dict[str, List[np.ndarray]],
        use_adam: bool = True,
    ) -> None:
        """
        Apply gradients to update network parameters.
        
        Parameters
        ----------
        gradients : dict
            Dictionary with 'weights' and 'biases' gradient lists.
        use_adam : bool
            If True, use Adam optimizer. Otherwise, use SGD.
        """
        if use_adam:
            self._adam_update(gradients)
        else:
            # Simple SGD
            for i in range(len(self.policy.weights)):
                self.policy.weights[i] -= self.learning_rate * gradients["weights"][i]
                self.policy.biases[i] -= self.learning_rate * gradients["biases"][i]
    
    def _adam_update(
        self,
        gradients: Dict[str, List[np.ndarray]],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Adam optimizer update."""
        # Initialize momentum buffers
        if self._adam_m is None:
            self._adam_m = [np.zeros_like(w) for w in self.policy.weights]
            self._adam_m += [np.zeros_like(b) for b in self.policy.biases]
            self._adam_v = [np.zeros_like(w) for w in self.policy.weights]
            self._adam_v += [np.zeros_like(b) for b in self.policy.biases]
        
        t = self._step_count + 1
        
        # Update weights
        for i in range(len(self.policy.weights)):
            g = gradients["weights"][i]
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * g
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * g ** 2
            m_hat = self._adam_m[i] / (1 - beta1 ** t)
            v_hat = self._adam_v[i] / (1 - beta2 ** t)
            self.policy.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        # Update biases
        n_weights = len(self.policy.weights)
        for i in range(len(self.policy.biases)):
            g = gradients["biases"][i]
            j = n_weights + i
            self._adam_m[j] = beta1 * self._adam_m[j] + (1 - beta1) * g
            self._adam_v[j] = beta2 * self._adam_v[j] + (1 - beta2) * g ** 2
            m_hat = self._adam_m[j] / (1 - beta1 ** t)
            v_hat = self._adam_v[j] / (1 - beta2 ** t)
            self.policy.biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get agent parameters."""
        return {
            "policy": self.policy.get_parameters(),
            "learning_rate": self.learning_rate,
            "step_count": self._step_count,
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set agent parameters."""
        if "policy" in params:
            self.policy.set_parameters(params["policy"])
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "step_count" in params:
            self._step_count = params["step_count"]
    
    def reset(self) -> None:
        """Reset agent state."""
        pass
    
    def __repr__(self) -> str:
        return (
            f"DeepHedgingAgent("
            f"policy={self.policy.input_dim}→{self.policy.hidden_layers}→{self.policy.output_dim}, "
            f"risk={self.risk_measure.name})"
        )


def create_deep_hedging_agent(
    input_dim: int = 7,
    hidden_layers: Optional[List[int]] = None,
    activation: str = "relu",
    risk_measure: str = "mean_variance",
    risk_aversion: float = 0.5,
    learning_rate: float = 0.001,
    **kwargs,
) -> DeepHedgingAgent:
    """
    Factory function to create a deep hedging agent.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (default 7 for standard state).
    hidden_layers : list of int, optional
        Hidden layer sizes. Default [64, 64].
    activation : str
        Activation function.
    risk_measure : str
        Risk measure: "variance", "mean_variance", "cvar", "entropic".
    risk_aversion : float
        Risk aversion parameter (for mean_variance or entropic).
    learning_rate : float
        Learning rate.
    **kwargs : dict
        Additional arguments for DeepHedgingAgent.
    
    Returns
    -------
    DeepHedgingAgent
        Configured agent.
    """
    from src.deep_hedging.core.risk_measures import create_risk_measure
    
    if hidden_layers is None:
        hidden_layers = [64, 64]
    
    policy = MLPPolicy(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=1,
        activation=activation,
        output_activation="tanh",
        output_scale=1.0,
    )
    
    risk = create_risk_measure(risk_measure, risk_aversion=risk_aversion)
    
    return DeepHedgingAgent(
        policy=policy,
        risk_measure=risk,
        learning_rate=learning_rate,
        **kwargs,
    )


__all__ = [
    "MLPPolicy",
    "DeepHedgingAgent",
    "create_deep_hedging_agent",
]
