"""
Generalised RL inference: save/load agent, select actions for deployment.

Artifact layout:
    artifact_dir/
    ├── config.json       # Optional RL/config
    ├── parameters.json  # Agent parameters (get_parameters())
    └── metadata.json    # Optional (agent class, version)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.q_learning.core.protocols import RLAgent

logger = logging.getLogger(__name__)


def _serialise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        np = None
    out = {}
    for k, v in params.items():
        if np is not None and hasattr(v, "tolist") and callable(getattr(v, "tolist")):
            out[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "tolist"):
            out[k] = [x.tolist() for x in v]
        else:
            out[k] = v
    return out


def _deserialise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is None:
        return params
    out = {}
    for k, v in params.items():
        if isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], (int, float)):
                out[k] = np.array(v)
            elif len(v) > 0 and isinstance(v[0], list):
                out[k] = np.array(v)
            else:
                out[k] = [np.array(x) for x in v]
        else:
            out[k] = v
    return out


def save_agent(
    agent: RLAgent,
    artifact_dir: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save agent parameters and optional config/metadata to artifact_dir.

    Parameters
    ----------
    agent : RLAgent
        Agent with get_parameters().
    artifact_dir : str
        Directory to write to.
    config : dict, optional
        Config to save (e.g. training config as dict).
    metadata : dict, optional
        Extra metadata (e.g. agent_class, version).

    Returns
    -------
    str
        Path to artifact directory.
    """
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    params = agent.get_parameters()
    with open(path / "parameters.json", "w") as f:
        json.dump(_serialise_params(params), f, indent=2)
    if config is not None:
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    meta = metadata or {}
    meta.setdefault("agent_class", type(agent).__name__)
    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Agent saved to {artifact_dir}")
    return str(path)


def load_agent(
    artifact_dir: str,
    agent_factory: Callable[..., RLAgent],
    factory_kwargs: Optional[Dict[str, Any]] = None,
) -> RLAgent:
    """
    Load agent from artifact_dir: load parameters and set on agent from factory.

    Parameters
    ----------
    artifact_dir : str
        Directory containing parameters.json (and optionally config.json).
    agent_factory : callable
        Called as agent_factory(**factory_kwargs) to create agent instance.
    factory_kwargs : dict, optional
        Arguments for agent_factory.

    Returns
    -------
    RLAgent
        Agent with parameters loaded.
    """
    path = Path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    params_path = path / "parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    with open(params_path) as f:
        params = json.load(f)
    params = _deserialise_params(params)
    kwargs = factory_kwargs or {}
    agent = agent_factory(**kwargs)
    agent.set_parameters(params)
    logger.info(f"Agent loaded from {artifact_dir}")
    return agent


def select_action(
    agent: RLAgent,
    state: Any,
    *,
    training: bool = False,
    explore: bool = False,
) -> Any:
    """
    Select a single action for deployment (e.g. in backtest or live).

    Parameters
    ----------
    agent : RLAgent
        Loaded agent.
    state : Any
        Current observation.
    training : bool
        Whether in training mode.
    explore : bool
        Whether to explore (e.g. epsilon-greedy); typically False for deployment.

    Returns
    -------
    action : Any
        Selected action.
    """
    return agent.select_action(state, training=training, explore=explore)


__all__ = ["save_agent", "load_agent", "select_action"]
