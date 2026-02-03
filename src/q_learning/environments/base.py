"""
Base RL environment implementing RLEnvironment protocol.

Provides a minimal, configurable env (e.g. random walk state, discrete actions)
for testing and as a template for trading/hedging sims that wrap pricers and market data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.q_learning.core.protocols import RLEnvironment


class BaseEnv:
    """
    Minimal RL environment: state is a 1D array, actions are integers in [0, n_actions-1].

    Transition: next_state = state + step_noise; reward = -abs(state - target) or custom.
    Episode ends when step >= max_steps or (optional) done condition.

    Use as template or for unit tests. Real trading/hedging envs should wrap
    backtesting engine, pricers, and market data and implement RLEnvironment.
    """

    def __init__(
        self,
        state_dim: int = 1,
        n_actions: int = 3,
        max_steps: int = 100,
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self._rng = np.random.default_rng(seed)
        self._step_count: int = 0
        self._state: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.standard_normal(self.state_dim).astype(np.float32)
        self._step_count = 0
        return self._state.copy(), {"step": 0}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        # Map action to delta: 0 = -1, 1 = 0, 2 = +1 (or clip for n_actions)
        a_scalar = int(np.asarray(action).ravel()[0]) if hasattr(action, "ravel") else int(action)
        a = int(np.clip(a_scalar, 0, self.n_actions - 1))
        delta = (a - (self.n_actions - 1) / 2) * 0.1
        noise = self._rng.standard_normal(self.state_dim).astype(np.float32) * 0.05
        next_state = self._state + delta + noise
        self._state = next_state
        self._step_count += 1
        # Simple reward: negative distance from 0
        reward = -float(np.abs(next_state).mean()) * self.reward_scale
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {"step": self._step_count, "action": a}
        return next_state.copy(), reward, terminated, truncated, info
