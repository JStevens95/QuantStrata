"""
Deep Hedging Module (Phase 7.6)

This module implements the deep hedging framework for learning optimal hedging
strategies using reinforcement learning. Key components:

Core (Generic):
- Transaction cost models: ProportionalCost, FixedCost, MarketImpactCost
- Risk measures: VarianceRisk, CVaRRisk, EntropicRisk
- HedgingEnv protocol and base classes

Environments (Model-Specific):
- GBMHedgingEnv: Hedging under GBM dynamics
- HestonHedgingEnv: Hedging under Heston dynamics (future)

Agents:
- DeltaHedgingAgent: Classical benchmark
- DeepHedgingAgent: Neural network policy

Training & Evaluation:
- HedgingTrainer: Training loop with risk measure optimisation
- HedgingEvaluator: Performance metrics and comparison

Quick Start
-----------
>>> from src.deep_hedging import (
...     HedgingConfig, ProportionalCost, GBMHedgingEnv,
...     DeltaHedgingAgent, DeepHedgingAgent, evaluate_agent, compare_agents
... )
>>> 
>>> # Setup
>>> config = HedgingConfig(option_type="call", strike=100, maturity=0.25)
>>> cost = ProportionalCost(spread_bps=10)
>>> env = GBMHedgingEnv(config, cost)
>>> 
>>> # Benchmark
>>> delta_agent = DeltaHedgingAgent()
>>> result = evaluate_agent(delta_agent, env, n_episodes=1000)
>>> print(f"Delta hedging: mean={result.mean_pnl:.4f}, std={result.std_pnl:.4f}")

References
----------
- Bühler et al. (2019) "Deep Hedging"
- Technical documentation: docs/reference/deep_hedging/theory.md
- Tutorial: docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb
"""

# Core components
from src.deep_hedging.core.costs import (
    TransactionCostModel,
    ProportionalCost,
    FixedCost,
    MarketImpactCost,
    CombinedCost,
    ZeroCost,
    create_realistic_cost,
)
from src.deep_hedging.core.risk_measures import (
    RiskMeasure,
    VarianceRisk,
    CVaRRisk,
    EntropicRisk,
    MeanVarianceRisk,
    create_risk_measure,
)
from src.deep_hedging.core.types import (
    HedgingConfig,
    HedgingState,
    HedgingResult,
    HedgingEpisode,
    DeepHedgingTrainingConfig,
)
from src.deep_hedging.core.protocols import (
    HedgingEnvironment,
    BaseHedgingEnv,
)

# Environments
from src.deep_hedging.environments import GBMHedgingEnv
from src.deep_hedging.environments.gbm import create_gbm_env

# Agents
from src.deep_hedging.agents import DeltaHedgingAgent, DeepHedgingAgent
from src.deep_hedging.agents.delta import NoHedgingAgent
from src.deep_hedging.agents.deep import MLPPolicy, create_deep_hedging_agent

# Training
from src.deep_hedging.training import (
    HedgingTrainer,
    simulate_hedging_batch,
    train_deep_hedging,
)

# Evaluation
from src.deep_hedging.evaluation import (
    HedgingEvaluator,
    evaluate_agent,
    compare_agents,
    compute_hedging_metrics,
)

__all__ = [
    # Transaction costs
    "TransactionCostModel",
    "ProportionalCost",
    "FixedCost",
    "MarketImpactCost",
    "CombinedCost",
    "ZeroCost",
    "create_realistic_cost",
    # Risk measures
    "RiskMeasure",
    "VarianceRisk",
    "CVaRRisk",
    "EntropicRisk",
    "MeanVarianceRisk",
    "create_risk_measure",
    # Types
    "HedgingConfig",
    "HedgingState",
    "HedgingResult",
    "HedgingEpisode",
    "DeepHedgingTrainingConfig",
    # Protocols
    "HedgingEnvironment",
    "BaseHedgingEnv",
    # Environments
    "GBMHedgingEnv",
    "create_gbm_env",
    # Agents
    "DeltaHedgingAgent",
    "DeepHedgingAgent",
    "NoHedgingAgent",
    "MLPPolicy",
    "create_deep_hedging_agent",
    # Training
    "HedgingTrainer",
    "simulate_hedging_batch",
    "train_deep_hedging",
    # Evaluation
    "HedgingEvaluator",
    "evaluate_agent",
    "compare_agents",
    "compute_hedging_metrics",
]
