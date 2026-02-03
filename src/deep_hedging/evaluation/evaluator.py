"""
Hedging Evaluation

Tools for evaluating and comparing hedging strategies.

Metrics
-------
- P&L distribution: mean, std, median, min, max
- Risk metrics: VaR, CVaR, Sharpe ratio
- Cost analysis: total costs, cost per rebalancing
- Tracking error: deviation from perfect hedge

Comparison
----------
The primary comparison is deep hedging vs. delta hedging:
- Does deep hedging achieve lower risk (variance, CVaR)?
- Does it have lower transaction costs?
- Is there a risk-cost trade-off?

Example
-------
>>> from src.deep_hedging.evaluation import compare_agents
>>> 
>>> results = compare_agents(
...     agents={"delta": delta_agent, "deep": deep_agent},
...     env=env,
...     n_episodes=1000,
... )
>>> print(results.to_dataframe())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.deep_hedging.core.types import HedgingResult, HedgingEpisode
from src.deep_hedging.core.protocols import BaseHedgingEnv
from src.deep_hedging.training.trainer import simulate_hedging_batch


def compute_hedging_metrics(
    pnl_samples: np.ndarray,
    cost_samples: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive hedging performance metrics.
    
    Parameters
    ----------
    pnl_samples : ndarray, shape (n_episodes,)
        Terminal P&L for each episode.
    cost_samples : ndarray, shape (n_episodes,), optional
        Total costs for each episode.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    
    Metrics
    -------
    - mean_pnl: Mean terminal P&L
    - std_pnl: Standard deviation of P&L
    - median_pnl: Median P&L
    - min_pnl: Worst P&L
    - max_pnl: Best P&L
    - sharpe: Sharpe ratio (mean / std)
    - var_95: 95% Value-at-Risk (5th percentile)
    - cvar_95: 95% CVaR (mean of worst 5%)
    - skewness: Skewness of P&L distribution
    - kurtosis: Excess kurtosis
    - mean_cost: Mean transaction cost (if provided)
    - pnl_minus_cost: Mean P&L net of average cost
    """
    pnl = np.asarray(pnl_samples)
    n = len(pnl)
    
    # Basic statistics
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl))
    
    metrics = {
        "n_episodes": n,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "median_pnl": float(np.median(pnl)),
        "min_pnl": float(np.min(pnl)),
        "max_pnl": float(np.max(pnl)),
    }
    
    # Sharpe ratio
    metrics["sharpe"] = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    
    # VaR and CVaR
    var_95 = float(np.percentile(pnl, 5))
    tail = pnl[pnl <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95
    
    metrics["var_95"] = var_95
    metrics["cvar_95"] = cvar_95
    
    # Higher moments
    if std_pnl > 0:
        centered = pnl - mean_pnl
        metrics["skewness"] = float(np.mean(centered ** 3) / std_pnl ** 3)
        metrics["kurtosis"] = float(np.mean(centered ** 4) / std_pnl ** 4 - 3)
    else:
        metrics["skewness"] = 0.0
        metrics["kurtosis"] = 0.0
    
    # Cost analysis
    if cost_samples is not None:
        costs = np.asarray(cost_samples)
        metrics["mean_cost"] = float(np.mean(costs))
        metrics["std_cost"] = float(np.std(costs))
        metrics["total_cost_ratio"] = float(np.mean(costs) / abs(mean_pnl)) if mean_pnl != 0 else 0.0
    
    return metrics


def evaluate_agent(
    agent: Any,
    env: BaseHedgingEnv,
    n_episodes: int = 1000,
    seed: Optional[int] = None,
    return_episodes: bool = False,
) -> HedgingResult:
    """
    Evaluate a hedging agent over multiple episodes.
    
    Parameters
    ----------
    agent : RLAgent-like
        Hedging agent.
    env : BaseHedgingEnv
        Hedging environment.
    n_episodes : int
        Number of evaluation episodes.
    seed : int, optional
        Random seed.
    return_episodes : bool
        If True, store full episode records.
    
    Returns
    -------
    HedgingResult
        Evaluation results with P&L samples and metrics.
    """
    result = simulate_hedging_batch(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        seed=seed,
        return_episodes=return_episodes,
    )
    
    if return_episodes:
        pnl_samples, cost_samples, episodes = result
    else:
        pnl_samples, cost_samples = result
        episodes = None
    
    agent_name = getattr(agent, "name", type(agent).__name__)
    
    return HedgingResult(
        pnl_samples=pnl_samples,
        cost_samples=cost_samples,
        episodes=episodes,
        agent_name=agent_name,
        n_episodes=n_episodes,
    )


@dataclass
class ComparisonResult:
    """
    Results of comparing multiple hedging agents.
    
    Attributes
    ----------
    results : dict
        Mapping from agent name to HedgingResult.
    metrics : dict
        Mapping from agent name to metrics dict.
    """
    
    results: Dict[str, HedgingResult]
    metrics: Dict[str, Dict[str, float]]
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for easy comparison."""
        try:
            import pandas as pd
            return pd.DataFrame(self.metrics).T
        except ImportError:
            # Return dict if pandas not available
            return self.metrics
    
    def improvement(
        self,
        agent_name: str,
        baseline_name: str = "DeltaHedging",
        metric: str = "std_pnl",
    ) -> float:
        """
        Compute improvement of agent over baseline.
        
        Parameters
        ----------
        agent_name : str
            Name of agent to evaluate.
        baseline_name : str
            Name of baseline agent.
        metric : str
            Metric to compare.
        
        Returns
        -------
        float
            Relative improvement (positive = agent is better).
        """
        agent_val = self.metrics[agent_name][metric]
        baseline_val = self.metrics[baseline_name][metric]
        
        if baseline_val == 0:
            return 0.0
        
        # For std and CVaR, lower is better
        if metric in ("std_pnl", "cvar_95", "var_95", "mean_cost"):
            return (baseline_val - agent_val) / abs(baseline_val)
        else:
            # For Sharpe and mean_pnl, higher is better
            return (agent_val - baseline_val) / abs(baseline_val)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Hedging Strategy Comparison", "=" * 40]
        
        for name, m in self.metrics.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Mean P&L: {m['mean_pnl']:>10.4f}")
            lines.append(f"  Std P&L:  {m['std_pnl']:>10.4f}")
            lines.append(f"  Sharpe:   {m['sharpe']:>10.3f}")
            lines.append(f"  CVaR 95%: {m['cvar_95']:>10.4f}")
            if "mean_cost" in m:
                lines.append(f"  Mean Cost:{m['mean_cost']:>10.4f}")
        
        return "\n".join(lines)


def compare_agents(
    agents: Dict[str, Any],
    env: BaseHedgingEnv,
    n_episodes: int = 1000,
    seed: Optional[int] = None,
) -> ComparisonResult:
    """
    Compare multiple hedging agents.
    
    Parameters
    ----------
    agents : dict
        Mapping from name to agent.
    env : BaseHedgingEnv
        Hedging environment.
    n_episodes : int
        Number of evaluation episodes.
    seed : int, optional
        Random seed (same seed used for all agents for fair comparison).
    
    Returns
    -------
    ComparisonResult
        Comparison results.
    
    Example
    -------
    >>> results = compare_agents(
    ...     agents={"Delta": delta_agent, "Deep": deep_agent},
    ...     env=env,
    ...     n_episodes=1000,
    ... )
    >>> print(results.summary())
    """
    results = {}
    metrics = {}
    
    for name, agent in agents.items():
        result = evaluate_agent(
            agent=agent,
            env=env,
            n_episodes=n_episodes,
            seed=seed,  # Same seed for fair comparison
        )
        results[name] = result
        metrics[name] = compute_hedging_metrics(
            result.pnl_samples,
            result.cost_samples,
        )
    
    return ComparisonResult(results=results, metrics=metrics)


@dataclass
class HedgingEvaluator:
    """
    Comprehensive hedging evaluation framework.
    
    This class provides utilities for:
    - Running evaluations
    - Comparing strategies
    - Generating reports
    - Visualising results
    
    Parameters
    ----------
    env : BaseHedgingEnv
        Hedging environment for evaluation.
    n_episodes : int
        Default number of evaluation episodes.
    seed : int, optional
        Random seed.
    """
    
    env: BaseHedgingEnv
    n_episodes: int = 1000
    seed: Optional[int] = None
    
    _results: Dict[str, HedgingResult] = field(default_factory=dict, repr=False)
    
    def evaluate(
        self,
        agent: Any,
        name: Optional[str] = None,
    ) -> HedgingResult:
        """
        Evaluate a single agent.
        
        Parameters
        ----------
        agent : RLAgent-like
            Agent to evaluate.
        name : str, optional
            Name for the result. Default: agent.name.
        
        Returns
        -------
        HedgingResult
            Evaluation results.
        """
        result = evaluate_agent(
            agent=agent,
            env=self.env,
            n_episodes=self.n_episodes,
            seed=self.seed,
        )
        
        name = name or result.agent_name
        self._results[name] = result
        
        return result
    
    def compare(
        self,
        agents: Optional[Dict[str, Any]] = None,
    ) -> ComparisonResult:
        """
        Compare all evaluated agents.
        
        Parameters
        ----------
        agents : dict, optional
            Additional agents to evaluate before comparison.
        
        Returns
        -------
        ComparisonResult
            Comparison results.
        """
        if agents:
            for name, agent in agents.items():
                self.evaluate(agent, name=name)
        
        metrics = {}
        for name, result in self._results.items():
            metrics[name] = compute_hedging_metrics(
                result.pnl_samples,
                result.cost_samples,
            )
        
        return ComparisonResult(results=self._results.copy(), metrics=metrics)
    
    def get_pnl_distribution(self, agent_name: str) -> np.ndarray:
        """Get P&L samples for an agent."""
        if agent_name not in self._results:
            raise KeyError(f"Agent '{agent_name}' not found. Run evaluate() first.")
        return self._results[agent_name].pnl_samples
    
    def plot_pnl_histogram(
        self,
        agent_names: Optional[List[str]] = None,
        bins: int = 50,
        ax=None,
    ):
        """
        Plot P&L histograms for comparison.
        
        Parameters
        ----------
        agent_names : list of str, optional
            Agents to plot. Default: all.
        bins : int
            Number of histogram bins.
        ax : matplotlib axis, optional
            Axis to plot on.
        
        Returns
        -------
        matplotlib axis
            The plot axis.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if agent_names is None:
            agent_names = list(self._results.keys())
        
        for name in agent_names:
            if name in self._results:
                pnl = self._results[name].pnl_samples
                ax.hist(pnl, bins=bins, alpha=0.5, label=name, density=True)
        
        ax.set_xlabel("Terminal P&L")
        ax.set_ylabel("Density")
        ax.set_title("P&L Distribution Comparison")
        ax.legend()
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        
        return ax


__all__ = [
    "HedgingEvaluator",
    "ComparisonResult",
    "evaluate_agent",
    "compare_agents",
    "compute_hedging_metrics",
]
