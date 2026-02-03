"""
Context passed to strategy at each streaming step.

Compatible with backtesting BacktestContext (current_date, step, total_steps, user_data)
so the same strategy signature works in both backtest and streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict


@dataclass
class LiveContext:
    """
    Context passed to strategy at each streaming step.

    Mirrors BacktestContext so strategy(market, portfolio, context) works
    in both backtest and live/streaming.

    Attributes
    ----------
    current_date : date
        Current date (parsed from timestamp when possible).
    timestamp : str
        Raw timestamp from stream (e.g. ISO date or datetime).
    step : int
        Current step number (0-indexed).
    total_steps : int
        Total steps (-1 if unknown in streaming).
    user_data : dict
        User-defined data persisted across steps.
    """

    current_date: date
    timestamp: str
    step: int
    total_steps: int = -1
    user_data: Dict[str, Any] = field(default_factory=dict)
