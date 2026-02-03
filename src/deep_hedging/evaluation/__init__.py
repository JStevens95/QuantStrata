"""
Deep Hedging Evaluation

Evaluation utilities for comparing hedging strategies:
- HedgingEvaluator: Run evaluation on multiple agents
- Performance metrics: Sharpe, CVaR, cost breakdown
- Comparison utilities: benchmark vs deep hedging
"""

from src.deep_hedging.evaluation.evaluator import (
    HedgingEvaluator,
    evaluate_agent,
    compare_agents,
    compute_hedging_metrics,
)

__all__ = [
    "HedgingEvaluator",
    "evaluate_agent",
    "compare_agents",
    "compute_hedging_metrics",
]
