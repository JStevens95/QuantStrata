"""
Pipeline: rl.backtest_agent

Run an RL agent in backtest mode using q_learning BacktestRunner.

Purpose
-------
1. Load agent from state or from artifact path (optional agent_factory in config)
2. Build or load environment (from state or create BaseEnv from config)
3. Run BacktestRunner
4. Store BacktestResult in state (RL_BACKTEST_RESULT)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys


def _rl_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract rl config block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("rl", {})


@dataclass(slots=True)
class LoadAgentStep(Step):
    """Step 1: Load agent from state or from artifact path."""
    def run(self, ctx: Context) -> Context:
        agent = ctx.state.get(Keys.RL_AGENT)
        if agent is not None:
            if ctx.logger:
                ctx.logger.info("Using RL agent from state")
            return ctx
        rl = _rl_cfg(ctx.cfg)
        agent_path = rl.get("agent_path") or ctx.state.get(Keys.RL_AGENT_PATH)
        if agent_path and ctx.artifact_store:
            try:
                from src.q_learning.pipelines.inference import load_agent
                factory = rl.get("agent_factory")
                if factory is None:
                    if ctx.logger:
                        ctx.logger.warning("rl.agent_path set but agent_factory not provided; using stub agent")
                    ctx.put(Keys.RL_AGENT, _stub_agent())
                    return ctx
                agent = load_agent(
                    str(ctx.artifact_store.artifacts_root / agent_path),
                    agent_factory=factory,
                    factory_kwargs=rl.get("agent_factory_kwargs"),
                )
                ctx.put(Keys.RL_AGENT, agent)
                if ctx.logger:
                    ctx.logger.info("Loaded RL agent from %s", agent_path)
            except Exception as e:
                if ctx.logger:
                    ctx.logger.warning("Failed to load agent from path: %s", e)
                ctx.put(Keys.RL_AGENT, _stub_agent())
        else:
            ctx.put(Keys.RL_AGENT, _stub_agent())
            if ctx.logger:
                ctx.logger.info("No agent in state or path; using stub agent for demo backtest")
        return ctx


def _stub_agent() -> Any:
    """Minimal RLAgent-compliant stub (random action) for pipeline demo when no agent provided."""
    import numpy as np
    from src.q_learning.core.protocols import RLAgent

    class StubAgent:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(42)
        def select_action(self, state: Any, *, training: bool = False, explore: bool = True) -> Any:
            return self._rng.integers(0, 3)
        def update(self, transitions: Any = None, batch: Any = None) -> Optional[Dict[str, Any]]:
            return None
        def get_parameters(self) -> Dict[str, Any]:
            return {}
        def set_parameters(self, params: Dict[str, Any]) -> None:
            pass

    return StubAgent()


@dataclass(slots=True)
class BuildEnvStep(Step):
    """Step 2: Build or load RL environment."""
    def run(self, ctx: Context) -> Context:
        env = ctx.state.get(Keys.RL_ENV)
        if env is not None:
            if ctx.logger:
                ctx.logger.info("Using RL env from state")
            return ctx
        try:
            from src.q_learning.environments.base import BaseEnv
        except ImportError:
            if ctx.logger:
                ctx.logger.warning("q_learning not available; env set to None")
            ctx.put(Keys.RL_ENV, None)
            return ctx
        rl = _rl_cfg(ctx.cfg)
        backtest = rl.get("backtest", {})
        env = BaseEnv(
            state_dim=int(backtest.get("state_dim", 1)),
            n_actions=int(backtest.get("n_actions", 3)),
            max_steps=int(backtest.get("max_steps", 50)),
            reward_scale=float(backtest.get("reward_scale", 1.0)),
            seed=int(backtest.get("seed", 42)),
        )
        ctx.put(Keys.RL_ENV, env)
        if ctx.logger:
            ctx.logger.info("Built BaseEnv for backtest")
        return ctx


@dataclass(slots=True)
class RunBacktestStep(Step):
    """Step 3: Run BacktestRunner and store result."""
    def run(self, ctx: Context) -> Context:
        agent = ctx.state.get(Keys.RL_AGENT)
        env = ctx.state.get(Keys.RL_ENV)
        if agent is None or env is None:
            if ctx.logger:
                ctx.logger.warning("Missing agent or env; skipping RL backtest")
            ctx.put(Keys.RL_BACKTEST_RESULT, None)
            return ctx
        try:
            from src.q_learning.runners.backtest import BacktestRunner, BacktestConfig
        except ImportError:
            if ctx.logger:
                ctx.logger.warning("q_learning.runners.backtest not available")
            ctx.put(Keys.RL_BACKTEST_RESULT, None)
            return ctx
        rl = _rl_cfg(ctx.cfg)
        backtest = rl.get("backtest", {})
        config = BacktestConfig(
            n_episodes=int(backtest.get("n_episodes", 10)),
            compute_sharpe=bool(backtest.get("compute_sharpe", True)),
            compute_drawdown=bool(backtest.get("compute_drawdown", True)),
        )
        runner = BacktestRunner(agent=agent, env=env, config=config)
        result = runner.run()
        ctx.put(Keys.RL_BACKTEST_RESULT, result)
        if ctx.logger:
            ctx.logger.info(
                "RL backtest complete: n_episodes=%s, mean_return=%.4f, sharpe=%.2f",
                result.n_episodes, result.mean_pnl_return, result.sharpe_ratio,
            )
        return ctx


def build_pipeline(cfg: Optional[RunConfig] = None) -> Pipeline:
    """Build the rl.backtest_agent pipeline."""
    return Pipeline(
        name="rl.backtest_agent",
        steps=[
            LoadAgentStep(name="load_agent"),
            BuildEnvStep(name="build_env"),
            RunBacktestStep(name="run_backtest"),
        ],
    )
