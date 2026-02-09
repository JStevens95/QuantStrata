"""
Pipeline: rl.deploy_agent

Load a saved RL agent and prepare for deployment (backtest or live).

Purpose
-------
1. Load agent from artifact path using agent_factory from config
2. Optionally validate agent (e.g. run one step)
3. Store loaded agent in state (RL_AGENT) for downstream pipelines (e.g. rl.backtest_agent)
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
    """Step 1: Load agent from artifact path and put in state."""
    def run(self, ctx: Context) -> Context:
        rl = _rl_cfg(ctx.cfg)
        agent_path = rl.get("agent_path")
        if not agent_path:
            if ctx.logger:
                ctx.logger.warning("rl.agent_path not set; nothing to load")
            return ctx
        factory = rl.get("agent_factory")
        if factory is None:
            if ctx.logger:
                ctx.logger.warning("rl.agent_factory not set; cannot load agent")
            return ctx
        try:
            from src.q_learning.pipelines.inference import load_agent
            root = ctx.artifact_store.artifacts_root if ctx.artifact_store else None
            path = str(root / agent_path) if root else agent_path
            agent = load_agent(
                path,
                agent_factory=factory,
                factory_kwargs=rl.get("agent_factory_kwargs"),
            )
            ctx.put(Keys.RL_AGENT, agent)
            ctx.put(Keys.RL_AGENT_PATH, agent_path)
            if ctx.logger:
                ctx.logger.info("Loaded RL agent from %s", path)
        except Exception as e:
            if ctx.logger:
                ctx.logger.warning("Failed to load agent: %s", e)
        return ctx


def build_pipeline(cfg: Optional[RunConfig] = None) -> Pipeline:
    """Build the rl.deploy_agent pipeline."""
    return Pipeline(
        name="rl.deploy_agent",
        steps=[
            LoadAgentStep(name="load_agent"),
        ],
    )
