"""
Q-Learning / RL environments for trading and hedging.

Provides environments that implement the RLEnvironment protocol:
- BaseEnv: Simple base environment for testing and templates
- TradingEnvironment: For training trading agents on historical data
- HedgingEnvironment: For training hedging agents (option hedging)
- StreamingEnvironment: For live/paper trading with streaming data

Usage:
    from src.q_learning.environments import (
        TradingEnvironment, TradingEnvConfig,
        HedgingEnvironment, HedgingEnvConfig,
        StreamingEnvironment, StreamingEnvConfig,
    )
    
    # Trading environment
    env = TradingEnvironment(
        data_provider=SimpleDataProvider(prices),
        config=TradingEnvConfig(max_steps=252),
    )
    
    # Hedging environment
    env = HedgingEnvironment(
        config=HedgingEnvConfig(spot=100, strike=100, maturity=0.25),
    )
"""

from src.q_learning.environments.base import BaseEnv
from src.q_learning.environments.trading import (
    TradingEnvironment,
    TradingEnvConfig,
    SimpleDataProvider,
    create_trading_env_from_prices,
)
from src.q_learning.environments.hedging import (
    HedgingEnvironment,
    HedgingEnvConfig,
)
from src.q_learning.environments.streaming import (
    StreamingEnvironment,
    StreamingEnvConfig,
    MarketDataBuffer,
    create_simulated_streaming_env,
)

__all__ = [
    # Base
    "BaseEnv",
    # Trading
    "TradingEnvironment",
    "TradingEnvConfig",
    "SimpleDataProvider",
    "create_trading_env_from_prices",
    # Hedging
    "HedgingEnvironment",
    "HedgingEnvConfig",
    # Streaming
    "StreamingEnvironment",
    "StreamingEnvConfig",
    "MarketDataBuffer",
    "create_simulated_streaming_env",
]
