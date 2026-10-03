"""
Asset-class extension seam.

Everything that differs per asset class sits behind three protocols
(:class:`RiskFactorBuilder`, :class:`ElementaryGenerator`, :class:`Pricer`), bundled
into an :class:`AssetClassPlugin` and held in a :class:`Registry`. Adding an asset
class is: implement the three (one subpackage), bundle, register one line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import numpy as np

from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import RiskFactorData, RiskFactorSpec
from src.rade_static_replication.domain.errors import BuilderError
from src.rade_static_replication.domain.instruments import ElementaryTrade


class RiskFactorBuilder(Protocol):
    """Builds one factor's market data + shocks from the client."""

    def build(
        self, spec: RiskFactorSpec, dependencies: Dict[str, RiskFactorData],
        client: MarketDataClient, cob_date: str,
    ) -> RiskFactorData:
        ...


class ElementaryGenerator(Protocol):
    """Emits the elementary-trade grid for one factor."""

    def generate(self, rf: RiskFactorData, grid: Dict) -> List[ElementaryTrade]:
        ...


class Pricer(Protocol):
    """Prices one factor's elementary trades."""

    def base_prices(self, trades: List[ElementaryTrade], rf: RiskFactorData) -> Dict[str, float]:
        ...

    def scenario_pnl(
        self, trades: List[ElementaryTrade], rf: RiskFactorData, base: Dict[str, float],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class AssetClassPlugin:
    """A complete asset-class plugin: builder + generator + pricer under one name."""
    name: str
    builder: RiskFactorBuilder
    generator: ElementaryGenerator
    pricer: Pricer


class Registry:
    """Maps an asset-class name to its :class:`AssetClassPlugin`."""

    def __init__(self) -> None:
        self._plugins: Dict[str, AssetClassPlugin] = {}

    def register(self, plugin: AssetClassPlugin) -> "Registry":
        self._plugins[plugin.name.lower()] = plugin
        return self

    def get(self, name: str) -> AssetClassPlugin:
        plugin = self._plugins.get(str(name).lower())
        if plugin is None:
            raise BuilderError(
                f"no asset-class plugin registered for {name!r}; registered: {sorted(self._plugins)}"
            )
        return plugin

    def __contains__(self, name: str) -> bool:
        return str(name).lower() in self._plugins

    @property
    def names(self) -> List[str]:
        return sorted(self._plugins)
