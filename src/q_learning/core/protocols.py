"""
Protocols for Q-Learning / Reinforcement Learning agents and environments.

Any agent that conforms to RLAgent can be trained via the generic RL pipeline.
Any environment that conforms to RLEnvironment can be used with any compliant agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class RLEnvironment(Protocol):
    """
    Minimal interface for an RL environment.

    Required:
        reset() -> state, info
        step(action) -> state, reward, terminated, truncated, info
        (optional) action_space, observation_space for typing

    Use case: Trading sim, hedging sim, or wrapper around backtesting/streaming.
    """

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment; return initial state and info.

        Returns
        -------
        state : Any
            Initial observation (e.g. ndarray, dict).
        info : dict
            Auxiliary information.
        """
        ...

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Parameters
        ----------
        action : Any
            Action chosen by the agent (e.g. int for discrete, ndarray for continuous).

        Returns
        -------
        state : Any
            Next observation.
        reward : float
            Reward for this step.
        terminated : bool
            True if episode ended (e.g. goal reached, failure).
        truncated : bool
            True if episode was cut (e.g. time limit).
        info : dict
            Auxiliary information.
        """
        ...


@runtime_checkable
class RLAgent(Protocol):
    """
    Minimal interface for an RL agent.

    Required:
        select_action(state, training / explore) -> action
        update(transitions or batch) -> optional metrics
        get_parameters() / set_parameters() for checkpointing and deployment

    Use case: Delta-hedging agent, algo-trading agent, DQN, policy gradient, etc.
    """

    def select_action(
        self,
        state: Any,
        *,
        training: bool = False,
        explore: bool = True,
    ) -> Any:
        """
        Select an action given the current state.

        Parameters
        ----------
        state : Any
            Current observation (e.g. ndarray, dict).
        training : bool
            Whether the agent is in training mode.
        explore : bool
            Whether to use exploration (e.g. epsilon-greedy); if False, use greedy.

        Returns
        -------
        action : Any
            Selected action (e.g. int, float, ndarray).
        """
        ...

    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Update the agent from a list of transitions or a batch.

        Parameters
        ----------
        transitions : list, optional
            List of (state, action, reward, next_state, terminated, truncated) or similar.
        batch : dict, optional
            Batched tensors/arrays (e.g. states, actions, rewards, next_states, dones).

        Returns
        -------
        metrics : dict or None
            Optional training metrics (e.g. loss, td_error).
        """
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return agent parameters for checkpointing.

        Returns
        -------
        dict
            Parameter dict (e.g. network weights).
        """
        ...

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Load agent parameters (e.g. from checkpoint or deployment).

        Parameters
        ----------
        params : dict
            Parameter dict from get_parameters().
        """
        ...
