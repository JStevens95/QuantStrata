"""
Unit tests for src.q_learning.pipelines.inference.

Tests save_agent(), load_agent(), select_action() with a minimal agent.
"""

import pytest
import tempfile
from pathlib import Path

from src.q_learning.pipelines.inference import save_agent, load_agent, select_action


class SerializableAgent:
    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    def select_action(self, state, *, training=False, explore=False):
        return self.n_actions - 1

    def update(self, transitions=None, batch=None):
        return None

    def get_parameters(self):
        return {"n_actions": self.n_actions}

    def set_parameters(self, params):
        self.n_actions = params.get("n_actions", self.n_actions)


def agent_factory(n_actions: int = 3):
    return SerializableAgent(n_actions=n_actions)


class TestSaveLoadAgent:
    def test_save_agent_creates_dir_and_files(self):
        agent = SerializableAgent(n_actions=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_agent(agent, tmp, config={"n_episodes": 10}, metadata={"v": 1})
            assert path == tmp
            assert (Path(tmp) / "parameters.json").exists()
            assert (Path(tmp) / "config.json").exists()
            assert (Path(tmp) / "metadata.json").exists()

    def test_load_agent_restores_parameters(self):
        agent = SerializableAgent(n_actions=7)
        with tempfile.TemporaryDirectory() as tmp:
            save_agent(agent, tmp)
            loaded = load_agent(tmp, agent_factory, factory_kwargs={"n_actions": 3})
            # Factory gives 3, but parameters.json has 7 from saved agent
            params = loaded.get_parameters()
            assert params["n_actions"] == 7

    def test_load_agent_raises_if_dir_missing(self):
        with pytest.raises(FileNotFoundError):
            load_agent("/nonexistent/path", agent_factory)


class TestSelectAction:
    def test_select_action_returns_agent_action(self):
        agent = SerializableAgent(n_actions=5)
        action = select_action(agent, [0.0, 0.0], training=False, explore=False)
        assert action == 4  # n_actions - 1
