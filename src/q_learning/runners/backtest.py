"""
Backtest Runner for evaluating RL agents on historical data.

Provides utilities for:
- Running multiple episodes with different seeds/starting points
- Collecting performance metrics
- Comparing against benchmarks

Example:
    from src.q_learning.runners import BacktestRunner, BacktestConfig
    from src.q_learning.environments import TradingEnvironment
    
    # Create environment and agent
    env = TradingEnvironment(data_provider=historical_data)
    agent = load_trained_agent("path/to/agent")
    
    # Run backtest
    runner = BacktestRunner(
        agent=agent,
        env=env,
        config=BacktestConfig(n_episodes=100),
    )
    result = runner.run()
    
    print(f"Mean return: {result.mean_pnl_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.q_learning.core.protocols import RLAgent, RLEnvironment
from src.q_learning.runners.base import BaseRunner, RunnerConfig, RunResult, EpisodeResult


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BacktestConfig(RunnerConfig):
    """Configuration for backtesting runner."""
    
    # Episode settings
    n_episodes: int = 100
    episode_seeds: Optional[List[int]] = None  # If provided, use these seeds
    
    # Data settings
    use_random_starts: bool = True
    start_indices: Optional[List[int]] = None  # If provided, use these starts
    
    # Parallelization
    parallel: bool = False  # Run episodes in parallel
    n_jobs: int = 4
    
    # Metrics
    compute_sharpe: bool = True
    compute_drawdown: bool = True
    risk_free_rate: float = 0.0  # Annualized
    
    # Benchmark comparison
    benchmark_agent: Optional[RLAgent] = None


# =============================================================================
# Results
# =============================================================================


@dataclass
class BacktestResult(RunResult):
    """Extended results for backtesting."""
    
    # Portfolio metrics
    pnl_returns: List[float] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Benchmark comparison
    benchmark_results: Optional["BacktestResult"] = None
    outperformance: float = 0.0
    
    @property
    def mean_pnl_return(self) -> float:
        """Mean P&L return."""
        if not self.pnl_returns:
            return 0.0
        return np.mean(self.pnl_returns)
    
    @property
    def std_pnl_return(self) -> float:
        """Standard deviation of P&L returns."""
        if len(self.pnl_returns) < 2:
            return 0.0
        return np.std(self.pnl_returns)
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        base_summary = super().summary()
        base_summary.update({
            "mean_pnl_return": self.mean_pnl_return,
            "std_pnl_return": self.std_pnl_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        })
        if self.benchmark_results:
            base_summary["outperformance"] = self.outperformance
            base_summary["benchmark_sharpe"] = self.benchmark_results.sharpe_ratio
        return base_summary


# =============================================================================
# Backtest Runner
# =============================================================================


class BacktestRunner(BaseRunner):
    """
    Runner for backtesting RL agents on historical data.
    
    Features:
    - Multiple episode runs with statistics
    - Support for random or fixed starting points
    - Performance metrics (Sharpe, drawdown)
    - Optional benchmark comparison
    
    Example:
        runner = BacktestRunner(
            agent=trained_agent,
            env=trading_env,
            config=BacktestConfig(
                n_episodes=100,
                compute_sharpe=True,
            ),
        )
        
        result = runner.run()
        
        print(f"Episodes: {result.n_episodes}")
        print(f"Mean return: {result.mean_pnl_return:.2%}")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Max DD: {result.max_drawdown:.2%}")
    """
    
    def __init__(
        self,
        agent: RLAgent,
        env: RLEnvironment,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        """
        Initialize backtest runner.
        
        Parameters
        ----------
        agent : RLAgent
            Trained agent to evaluate.
        env : RLEnvironment
            Trading/hedging environment with historical data.
        config : BacktestConfig, optional
            Backtest configuration.
        """
        config = config or BacktestConfig()
        super().__init__(agent, env, config)
        self.bt_config: BacktestConfig = config
    
    def run(self, **kwargs: Any) -> BacktestResult:
        """
        Run backtest across multiple episodes.
        
        Parameters
        ----------
        **kwargs
            Override configuration parameters.
            
        Returns
        -------
        BacktestResult
            Complete backtest results.
        """
        # Override config with kwargs
        n_episodes = kwargs.get("n_episodes", self.bt_config.n_episodes)
        
        start_time = datetime.now()
        self._step_count = 0
        
        episodes: List[EpisodeResult] = []
        pnl_returns: List[float] = []
        
        # Determine seeds/starts
        if self.bt_config.episode_seeds:
            seeds = self.bt_config.episode_seeds[:n_episodes]
        else:
            seeds = list(range(n_episodes))
        
        # Run episodes
        for i, seed in enumerate(seeds):
            if self.bt_config.verbose and (i + 1) % 10 == 0:
                print(f"Running episode {i + 1}/{n_episodes}...")
            
            # Set options for start index if provided
            options = None
            if self.bt_config.start_indices and i < len(self.bt_config.start_indices):
                options = {"start_idx": self.bt_config.start_indices[i]}
            
            # Run episode
            episode = self._run_episode(
                episode_id=i,
                seed=seed if self.bt_config.use_random_starts else None,
            )
            episodes.append(episode)
            
            # Extract P&L return from final info
            pnl = episode.final_info.get("pnl", 0.0)
            initial = episode.final_info.get("initial_capital", 
                       episode.final_info.get("portfolio_value", 1.0) - pnl)
            if initial > 0:
                pnl_returns.append(pnl / initial)
            else:
                pnl_returns.append(episode.total_reward)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Compute metrics
        sharpe = self._compute_sharpe(pnl_returns) if self.bt_config.compute_sharpe else 0.0
        max_dd = self._compute_max_drawdown(episodes) if self.bt_config.compute_drawdown else 0.0
        
        result = BacktestResult(
            episodes=episodes,
            total_steps=self._step_count,
            total_time_seconds=total_time,
            config=self.bt_config.__dict__,
            pnl_returns=pnl_returns,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
        )
        
        # Benchmark comparison
        if self.bt_config.benchmark_agent:
            benchmark_result = self._run_benchmark()
            result.benchmark_results = benchmark_result
            result.outperformance = result.mean_pnl_return - benchmark_result.mean_pnl_return
        
        if self.bt_config.verbose:
            self._print_summary(result)
        
        return result
    
    def _compute_sharpe(
        self,
        returns: List[float],
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        returns_arr = np.array(returns)
        excess_returns = returns_arr - self.bt_config.risk_free_rate / periods_per_year
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return < 1e-8:
            return 0.0
        
        # Annualize
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def _compute_max_drawdown(self, episodes: List[EpisodeResult]) -> float:
        """Compute maximum drawdown across all episodes."""
        max_dd = 0.0
        
        for episode in episodes:
            # Get P&L trajectory from rewards if available
            if episode.rewards:
                cumulative = np.cumsum(episode.rewards)
                running_max = np.maximum.accumulate(cumulative + 1)
                drawdowns = (running_max - (cumulative + 1)) / running_max
                episode_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
                max_dd = max(max_dd, episode_dd)
        
        return float(max_dd)
    
    def _run_benchmark(self) -> BacktestResult:
        """Run benchmark agent for comparison."""
        benchmark_runner = BacktestRunner(
            agent=self.bt_config.benchmark_agent,
            env=self.env,
            config=BacktestConfig(
                n_episodes=self.bt_config.n_episodes,
                episode_seeds=self.bt_config.episode_seeds,
                verbose=False,
            ),
        )
        return benchmark_runner.run()
    
    def _print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Episodes:        {result.n_episodes}")
        print(f"Total steps:     {result.total_steps}")
        print(f"Time:            {result.total_time_seconds:.1f}s")
        print("-" * 50)
        print(f"Mean P&L return: {result.mean_pnl_return:.2%}")
        print(f"Std P&L return:  {result.std_pnl_return:.2%}")
        print(f"Sharpe ratio:    {result.sharpe_ratio:.2f}")
        print(f"Max drawdown:    {result.max_drawdown:.2%}")
        
        if result.benchmark_results:
            print("-" * 50)
            print("BENCHMARK COMPARISON")
            print(f"Benchmark return: {result.benchmark_results.mean_pnl_return:.2%}")
            print(f"Outperformance:   {result.outperformance:.2%}")
        
        print("=" * 50)


__all__ = [
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
]
