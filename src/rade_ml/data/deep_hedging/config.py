"""
Data pipeline configuration for the Deep Hedging model.

Inherits universal pipeline settings from DataPipelineConfig and adds
deep-hedging-specific parameters: market dynamics, option specification,
and simulation controls.
"""
from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal
from dataclasses import dataclass, field, asdict

from src.rade_ml.core.config import DataPipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketDynamicsConfig:
    """Configuration for the underlying price process."""

    model: Literal["gbm", "heston"] = "gbm"
    spot_0: float = 100.0
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0

    # GBM parameters
    volatility: float = 0.2

    # Heston parameters (used when model="heston")
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketDynamicsConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class OptionConfig:
    """Configuration for the derivative being hedged."""

    option_type: Literal["call", "put"] = "call"
    strike: float = 100.0
    maturity_years: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo path generation."""

    num_paths: int = 100_000
    num_steps: int = 63
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DeepHedgingDataConfig(DataPipelineConfig):
    """
    Data configuration for the Deep Hedging model.

    Inherits universal pipeline settings (batch size, shuffle, cache, etc.)
    from DataPipelineConfig and adds deep-hedging-specific settings for
    market simulation, option specification, and path generation.
    """

    market: MarketDynamicsConfig = field(default_factory=MarketDynamicsConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_build_kwargs()
        base.update({
            "market": self.market.to_dict(),
            "option": self.option.to_dict(),
            "simulation": self.simulation.to_dict(),
        })
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepHedgingDataConfig":
        return cls(
            market=MarketDynamicsConfig.from_dict(d["market"]) if d.get("market") else MarketDynamicsConfig(),
            option=OptionConfig.from_dict(d["option"]) if d.get("option") else OptionConfig(),
            simulation=SimulationConfig.from_dict(d["simulation"]) if d.get("simulation") else SimulationConfig(),
            batch_size=d.get("batch_size", 32),
            shuffle=d.get("shuffle", True),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DeepHedgingDataConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
