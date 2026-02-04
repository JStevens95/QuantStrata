"""
Live Runner for deploying RL agents in real-time/paper trading.

Provides utilities for:
- Running agents in live or paper trading mode
- Handling real-time data feeds
- Managing risk controls and position limits
- Logging and monitoring

Example:
    from src.q_learning.runners import LiveRunner, LiveConfig
    from src.q_learning.environments import StreamingEnvironment
    
    # Create streaming environment
    env = StreamingEnvironment(
        streaming_engine=engine,
        config=StreamingEnvConfig(mode="paper"),
    )
    
    # Load trained agent
    agent = load_trained_agent("path/to/agent")
    
    # Create and start live runner
    runner = LiveRunner(
        agent=agent,
        env=env,
        config=LiveConfig(
            max_runtime_hours=8,
            risk_check_interval=100,
        ),
    )
    
    # Start live trading (blocking)
    result = runner.run()
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.q_learning.core.protocols import RLAgent, RLEnvironment
from src.q_learning.runners.base import BaseRunner, RunnerConfig, RunResult, EpisodeResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LiveConfig(RunnerConfig):
    """Configuration for live trading runner."""
    
    # Runtime settings
    max_runtime_hours: Optional[float] = None  # None = run indefinitely
    max_steps: Optional[int] = None
    
    # Risk controls
    max_drawdown: float = 0.1  # Stop if drawdown exceeds this
    max_loss: float = float("inf")  # Stop if total loss exceeds this
    risk_check_interval: int = 10  # Check risk every N steps
    
    # Position limits
    max_position: float = 1.0  # Maximum position as fraction of capital
    
    # Monitoring
    heartbeat_interval: int = 60  # Seconds between heartbeat logs
    metrics_interval: int = 100  # Log metrics every N steps
    
    # Callbacks
    on_risk_breach: Optional[Callable[[str, float], None]] = None
    on_trade: Optional[Callable[[Dict[str, Any]], None]] = None
    
    # Graceful shutdown
    graceful_shutdown: bool = True
    shutdown_timeout: float = 30.0  # Seconds to wait for graceful shutdown


# =============================================================================
# Live Runner
# =============================================================================


class LiveRunner(BaseRunner):
    """
    Runner for live/paper trading with RL agents.
    
    Features:
    - Real-time execution with streaming environment
    - Risk controls (drawdown, max loss, position limits)
    - Graceful shutdown handling (SIGINT, SIGTERM)
    - Heartbeat and metrics logging
    - Trade callbacks for monitoring
    
    Example:
        config = LiveConfig(
            max_runtime_hours=8,
            max_drawdown=0.05,
            risk_check_interval=50,
        )
        
        runner = LiveRunner(
            agent=trained_agent,
            env=streaming_env,
            config=config,
        )
        
        # Run (blocks until stopped or limit reached)
        result = runner.run()
        
        # Or run in background
        runner.start_async()
        # ... do other things ...
        runner.stop()
    """
    
    def __init__(
        self,
        agent: RLAgent,
        env: RLEnvironment,
        config: Optional[LiveConfig] = None,
    ) -> None:
        """
        Initialize live runner.
        
        Parameters
        ----------
        agent : RLAgent
            Trained agent to deploy.
        env : RLEnvironment
            Streaming environment for live execution.
        config : LiveConfig, optional
            Live trading configuration.
        """
        config = config or LiveConfig()
        super().__init__(agent, env, config)
        self.live_config: LiveConfig = config
        
        # State
        self._is_running: bool = False
        self._should_stop: bool = False
        self._start_time: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        
        # Tracking
        self._peak_value: float = 0.0
        self._total_pnl: float = 0.0
        self._trades: List[Dict[str, Any]] = []
        self._last_heartbeat: datetime = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        if config.graceful_shutdown:
            self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        def handler(signum: int, frame: Any) -> None:
            logger.info("Received shutdown signal, stopping gracefully...")
            self.stop()
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def run(self, **kwargs: Any) -> RunResult:
        """
        Run live trading (blocking).
        
        Parameters
        ----------
        **kwargs
            Override configuration parameters.
            
        Returns
        -------
        RunResult
            Trading results.
        """
        self._is_running = True
        self._should_stop = False
        self._start_time = datetime.now()
        self._step_count = 0
        self._trades = []
        
        episodes: List[EpisodeResult] = []
        
        try:
            # Reset environment
            state, info = self.env.reset()
            
            # Initialize tracking
            self._peak_value = info.get("portfolio_value", 0.0)
            initial_value = self._peak_value
            episode_rewards: List[float] = []
            episode_actions: List[Any] = []
            
            logger.info("Live trading started at %s", self._start_time)
            
            while not self._should_stop:
                # Check runtime limit
                if self._check_runtime_limit():
                    logger.info("Runtime limit reached, stopping...")
                    break
                
                # Check step limit
                if self.live_config.max_steps and self._step_count >= self.live_config.max_steps:
                    logger.info("Step limit reached, stopping...")
                    break
                
                # Select action
                action = self.agent.select_action(
                    state,
                    training=False,
                    explore=self.live_config.explore,
                )
                
                # Execute step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Update tracking
                self._step_count += 1
                episode_rewards.append(reward)
                episode_actions.append(action)
                
                # Track trade
                if "trade_qty" in info and abs(info.get("trade_qty", 0)) > 1e-8:
                    trade_info = {
                        "step": self._step_count,
                        "timestamp": datetime.now(),
                        "action": action,
                        **{k: v for k, v in info.items() if k != "trade_qty"},
                        "trade_qty": info["trade_qty"],
                    }
                    self._trades.append(trade_info)
                    
                    if self.live_config.on_trade:
                        self.live_config.on_trade(trade_info)
                
                # Update P&L tracking
                portfolio_value = info.get("portfolio_value", self._peak_value)
                self._peak_value = max(self._peak_value, portfolio_value)
                self._total_pnl = info.get("pnl", 0.0)
                
                # Risk check
                if self._step_count % self.live_config.risk_check_interval == 0:
                    if self._check_risk_limits(portfolio_value):
                        logger.warning("Risk limit breached, stopping...")
                        break
                
                # Heartbeat
                self._send_heartbeat()
                
                # Metrics logging
                if self._step_count % self.live_config.metrics_interval == 0:
                    self._log_metrics(info)
                
                # Step callback
                if self.live_config.on_step:
                    self.live_config.on_step(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        info=info,
                    )
                
                # Check for episode end
                if terminated or truncated:
                    # Record episode
                    episode = EpisodeResult(
                        episode_id=len(episodes),
                        total_reward=sum(episode_rewards),
                        n_steps=len(episode_rewards),
                        final_info=info,
                        rewards=episode_rewards.copy(),
                        actions=episode_actions.copy(),
                        start_time=self._start_time,
                        end_time=datetime.now(),
                    )
                    episodes.append(episode)
                    
                    if self.live_config.on_episode_end:
                        self.live_config.on_episode_end(
                            episode_id=episode.episode_id,
                            total_reward=episode.total_reward,
                            n_steps=episode.n_steps,
                            info=info,
                        )
                    
                    # Reset for next episode
                    if not self._should_stop:
                        state, info = self.env.reset()
                        episode_rewards = []
                        episode_actions = []
                        continue
                
                state = next_state
        
        except Exception as e:
            logger.error("Error during live trading: %s", e)
            raise
        
        finally:
            self._is_running = False
            end_time = datetime.now()
            total_time = (end_time - self._start_time).total_seconds()
            
            logger.info("Live trading stopped after %.1f seconds", total_time)
        
        return RunResult(
            episodes=episodes,
            total_steps=self._step_count,
            total_time_seconds=total_time,
            config=self.live_config.__dict__,
        )
    
    def start_async(self) -> None:
        """Start live trading in background thread."""
        if self._is_running:
            raise RuntimeError("Runner is already running")
        
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        
        logger.info("Live runner started in background thread")
    
    def stop(self) -> None:
        """Stop live trading gracefully."""
        logger.info("Stopping live runner...")
        self._should_stop = True
        
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.live_config.shutdown_timeout)
            
            if self._thread.is_alive():
                logger.warning("Runner did not stop gracefully within timeout")
    
    def is_running(self) -> bool:
        """Check if runner is currently active."""
        return self._is_running
    
    def _check_runtime_limit(self) -> bool:
        """Check if runtime limit has been reached."""
        if self.live_config.max_runtime_hours is None:
            return False
        
        elapsed = datetime.now() - self._start_time
        max_duration = timedelta(hours=self.live_config.max_runtime_hours)
        
        return elapsed >= max_duration
    
    def _check_risk_limits(self, current_value: float) -> bool:
        """
        Check risk limits.
        
        Returns True if any limit is breached.
        """
        # Drawdown check
        if self._peak_value > 0:
            drawdown = (self._peak_value - current_value) / self._peak_value
            
            if drawdown > self.live_config.max_drawdown:
                if self.live_config.on_risk_breach:
                    self.live_config.on_risk_breach("drawdown", drawdown)
                logger.warning("Drawdown limit breached: %.2%% > %.2%%", 
                             drawdown * 100, self.live_config.max_drawdown * 100)
                return True
        
        # Loss check
        if self._total_pnl < -self.live_config.max_loss:
            if self.live_config.on_risk_breach:
                self.live_config.on_risk_breach("max_loss", self._total_pnl)
            logger.warning("Max loss limit breached: %.2f", self._total_pnl)
            return True
        
        return False
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat log if interval has passed."""
        now = datetime.now()
        elapsed = (now - self._last_heartbeat).total_seconds()
        
        if elapsed >= self.live_config.heartbeat_interval:
            logger.info(
                "Heartbeat: step=%d, pnl=%.2f, running_time=%s",
                self._step_count,
                self._total_pnl,
                now - self._start_time,
            )
            self._last_heartbeat = now
    
    def _log_metrics(self, info: Dict[str, Any]) -> None:
        """Log trading metrics."""
        metrics = {
            "step": self._step_count,
            "portfolio_value": info.get("portfolio_value", 0),
            "pnl": info.get("pnl", 0),
            "position": info.get("position", 0),
            "n_trades": len(self._trades),
        }
        
        logger.info("Metrics: %s", metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current runner status."""
        elapsed = (
            (datetime.now() - self._start_time).total_seconds()
            if self._start_time else 0
        )
        
        return {
            "is_running": self._is_running,
            "step_count": self._step_count,
            "elapsed_seconds": elapsed,
            "total_pnl": self._total_pnl,
            "n_trades": len(self._trades),
            "peak_value": self._peak_value,
        }
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get list of executed trades."""
        return self._trades.copy()


__all__ = [
    "LiveRunner",
    "LiveConfig",
]
