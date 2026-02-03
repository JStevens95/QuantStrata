"""
Generic RL training loop for QuantStrata.

Provides run_training() and RLTrainingLoop for any agent conforming to RLAgent
and any environment conforming to RLEnvironment.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.q_learning.core.protocols import RLAgent, RLEnvironment
from src.q_learning.core.types import (
    RLTrainingConfig,
    RLTrainingResult,
    Transition,
)

logger = logging.getLogger(__name__)


class RLTrainingLoop:
    """
    Generic RL training loop: reset env → step with agent → collect transition → update agent.

    Supports:
    - Episode-based training with configurable max steps per episode
    - Checkpointing (save best, periodic)
    - Evaluation episodes (no exploration) for logging
    - Optional replay buffer: collect transitions and update in batches (caller can pass buffer)
    """

    def __init__(
        self,
        agent: RLAgent,
        env: RLEnvironment,
        config: RLTrainingConfig,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self._episode_returns: List[float] = []
        self._episode_lengths: List[int] = []
        self._history: Dict[str, List[float]] = {"loss": [], "epsilon": []}
        self._best_episode_return = float("-inf")
        self._best_episode = 0

    def _run_episode(self, training: bool, explore: bool) -> tuple[float, int, List[Transition]]:
        """Run one episode; return total return, length, and list of transitions."""
        state, info = self.env.reset()
        total_reward = 0.0
        steps = 0
        transitions: List[Transition] = []
        max_steps = self.config.max_steps_per_episode or 0

        while True:
            action = self.agent.select_action(state, training=training, explore=explore)
            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            steps += 1
            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                    info=dict(step_info),
                )
            )
            state = next_state

            if terminated or truncated:
                break
            if max_steps > 0 and steps >= max_steps:
                break

        return total_reward, steps, transitions

    def _save_checkpoint(self, episode: int, episode_return: float, is_best: bool) -> None:
        """Save agent checkpoint if configured."""
        if not self.config.checkpoint_dir:
            return
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        suffix = "best" if is_best else f"episode_{episode}"
        ckpt_path = ckpt_dir / f"agent_{suffix}.json"
        params = self.agent.get_parameters()
        # Serialise for JSON (e.g. numpy -> list)
        serialisable = _serialise_params(params)
        with open(ckpt_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        if self.config.verbose >= 1:
            logger.info(f"Checkpoint saved: {ckpt_path} (episode {episode}, return={episode_return:.4f})")

    def run(self) -> RLTrainingResult:
        """
        Run the RL training loop.

        Returns
        -------
        RLTrainingResult
            Episode returns, lengths, history, best episode, and metadata.
        """
        start_time = time.time()

        for episode in range(1, self.config.n_episodes + 1):
            # Training episode (with exploration)
            episode_return, episode_len, transitions = self._run_episode(
                training=True, explore=True
            )
            self._episode_returns.append(episode_return)
            self._episode_lengths.append(episode_len)

            # Update agent from collected transitions
            metrics = self.agent.update(transitions=transitions)
            if metrics:
                for k, v in metrics.items():
                    if k not in self._history:
                        self._history[k] = []
                    self._history[k].append(float(v))

            # Track best
            if episode_return > self._best_episode_return:
                self._best_episode_return = episode_return
                self._best_episode = episode
                if self.config.checkpoint_dir and self.config.save_best_only:
                    self._save_checkpoint(episode, episode_return, is_best=True)

            # Periodic checkpoint
            if (
                self.config.checkpoint_dir
                and self.config.checkpoint_frequency > 0
                and episode % self.config.checkpoint_frequency == 0
                and not self.config.save_best_only
            ):
                self._save_checkpoint(episode, episode_return, is_best=False)

            # Logging
            if self.config.verbose >= 1 and episode % self.config.log_every == 0:
                logger.info(
                    f"Episode {episode}/{self.config.n_episodes} — return: {episode_return:.4f}, "
                    f"length: {episode_len}"
                )

            # Optional evaluation episodes (no exploration)
            if self.config.eval_episodes > 0 and episode % self.config.log_every == 0:
                eval_returns = []
                for _ in range(self.config.eval_episodes):
                    er, _, _ = self._run_episode(training=False, explore=False)
                    eval_returns.append(er)
                if self.config.verbose >= 2:
                    logger.info(
                        f"Eval (episode {episode}): mean_return={sum(eval_returns) / len(eval_returns):.4f}"
                    )

        training_time = time.time() - start_time
        return RLTrainingResult(
            episode_returns=self._episode_returns,
            episode_lengths=self._episode_lengths,
            history=self._history,
            best_episode_return=self._best_episode_return,
            best_episode=self._best_episode,
            config=self.config,
            training_time_seconds=training_time,
            metadata={"n_episodes": self.config.n_episodes},
        )


def _serialise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert agent params to JSON-serialisable form (e.g. numpy -> list)."""
    try:
        import numpy as np
    except ImportError:
        np = None
    out = {}
    for k, v in params.items():
        if np is not None and isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "tolist"):
            out[k] = [x.tolist() for x in v]
        else:
            out[k] = v
    return out


def run_training(
    agent: RLAgent,
    env: RLEnvironment,
    config: RLTrainingConfig,
) -> RLTrainingResult:
    """
    Run RL training: generic loop over episodes (env reset → step → agent update).

    Parameters
    ----------
    agent : RLAgent
        Agent conforming to RLAgent protocol.
    env : RLEnvironment
        Environment conforming to RLEnvironment protocol.
    config : RLTrainingConfig
        Training configuration (n_episodes, gamma, checkpoint_dir, etc.).

    Returns
    -------
    RLTrainingResult
        Episode returns, lengths, history, best episode, training time.

    Example
    -------
    >>> from src.q_learning.core import RLAgent, RLEnvironment, RLTrainingConfig
    >>> from src.q_learning.pipelines import run_training
    >>> config = RLTrainingConfig(n_episodes=100, gamma=0.99)
    >>> result = run_training(agent, env, config)
    >>> print(result.best_episode_return)
    """
    loop = RLTrainingLoop(agent, env, config)
    return loop.run()


__all__ = ["run_training", "RLTrainingLoop"]
