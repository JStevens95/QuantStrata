"""
Hedging Agents

Agents that implement hedging policies for the deep hedging framework.

Available Agents
----------------
- DeltaHedgingAgent: Classical Black-Scholes delta hedging (benchmark)
- DeepHedgingAgent: Neural network policy trained via risk minimisation

All agents conform to the RLAgent protocol from q_learning.core.protocols,
ensuring compatibility with the generic RL training and evaluation pipelines.
"""

from src.deep_hedging.agents.delta import DeltaHedgingAgent
from src.deep_hedging.agents.deep import DeepHedgingAgent

__all__ = [
    "DeltaHedgingAgent",
    "DeepHedgingAgent",
]
