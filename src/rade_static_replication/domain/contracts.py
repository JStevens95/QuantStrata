"""
Stage contracts — the typed handoffs between pipeline stages.

The pipeline is a sequence of pure stages, each consuming the previous contract and
emitting exactly one new one:

    RawPortfolio -> Portfolio -> RiskFactorUniverse -> FactorDataSet
        -> ElementaryUniverse -> BasePriceSet -> PnLResult -> ClusterSet -> ArtifactManifest

Contracts are dumb dataclasses with light helpers — no I/O, no business logic. They
are the stable interface that lets any one stage be swapped without disturbing its
neighbours; the orchestrator records each on the ``RunContext``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.scenarios import ScenarioSet
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot


# ----- portfolio -----

@dataclass(frozen=True)
class RawPortfolio:
    """Untouched portfolio payload from the portfolio client."""
    attributes: pd.DataFrame
    target_pnl: pd.DataFrame
    cob_date: str
    source: str = ""


@dataclass(frozen=True)
class Portfolio:
    """Normalised, validated portfolio — one row per trade."""
    attributes: pd.DataFrame
    target_pnl: pd.DataFrame
    scenario_ids: np.ndarray
    cob_date: str

    @property
    def trade_ids(self) -> List[str]:
        return [str(i) for i in self.attributes.index]

    @property
    def asset_classes(self) -> List[str]:
        if "AssetClass" not in self.attributes.columns:
            return []
        return sorted(self.attributes["AssetClass"].dropna().unique().tolist())


# ----- risk-factor universe -----

@dataclass(frozen=True)
class RiskFactorSpec:
    """Identity of a risk factor and its market-data dependencies."""
    factor_id: str
    asset_class: str
    dependencies: Tuple[str, ...] = ()
    is_primary: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskFactorUniverse:
    """De-duplicated set of risk factors the portfolio needs."""
    specs: Dict[str, RiskFactorSpec]
    factors_by_trade: Dict[str, List[str]]

    @property
    def primary_ids(self) -> List[str]:
        return [fid for fid, s in self.specs.items() if s.is_primary]

    def build_order(self) -> List[str]:
        """Dependency-respecting build order (topological sort)."""
        order: List[str] = []
        seen: set[str] = set()

        def _visit(fid: str, stack: Tuple[str, ...] = ()) -> None:
            if fid in seen:
                return
            if fid in stack:
                raise ValueError(f"cyclic risk-factor dependency at {fid}: {stack}")
            spec = self.specs.get(fid)
            if spec is not None:
                for dep in spec.dependencies:
                    _visit(dep, stack + (fid,))
            seen.add(fid)
            order.append(fid)

        for fid in self.specs:
            _visit(fid)
        return order


# ----- risk-factor data -----

@dataclass(frozen=True)
class RiskFactorData:
    """A factor with its built base snapshot + shocked scenario states."""
    spec: RiskFactorSpec
    snapshot: MarketSnapshot
    scenarios: ScenarioSet
    dependencies: Dict[str, "RiskFactorData"] = field(default_factory=dict)

    @property
    def factor_id(self) -> str:
        return self.spec.factor_id


@dataclass(frozen=True)
class FactorDataSet:
    """All built risk-factor data, keyed by ``factor_id``."""
    factors: Dict[str, RiskFactorData]

    @property
    def n_scenarios(self) -> int:
        for rf in self.factors.values():
            return rf.scenarios.n_scenarios
        return 0


# ----- elementary trades -----

@dataclass(frozen=True)
class ElementaryUniverse:
    """Elementary trades for the portfolio, grouped by ``factor_id``."""
    by_factor: Dict[str, List[ElementaryTrade]]

    def flat(self) -> List[ElementaryTrade]:
        out: List[ElementaryTrade] = []
        for fid in sorted(self.by_factor):
            out.extend(self.by_factor[fid])
        return out

    @property
    def n_trades(self) -> int:
        return sum(len(v) for v in self.by_factor.values())


# ----- pricing & PnL -----

@dataclass(frozen=True)
class BasePriceSet:
    """Base (COB) present values, ``factor_id -> {trade_id -> PV}``."""
    by_factor: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class FactorPnL:
    """Scenario PnL for one factor's elementary trades.

    ``pnl`` is ``(n_trades, n_scenarios)``, rows aligned to ``trade_ids`` and columns
    to ``scenario_ids``.
    """
    factor_id: str
    trade_ids: List[str]
    scenario_ids: np.ndarray
    pnl: np.ndarray


@dataclass(frozen=True)
class PnLResult:
    """All elementary PnL, ``factor_id -> FactorPnL``."""
    by_factor: Dict[str, FactorPnL]


# ----- clustering & artifacts -----

@dataclass(frozen=True)
class ClusterSpec:
    """A single-asset-class slice keyed by attribute values."""
    cluster_id: str
    key: Dict[str, str]
    asset_class: str
    risk_factor_ids: List[str]
    target_trade_ids: List[str]


@dataclass(frozen=True)
class ClusterSet:
    """The resolved list of clusters for the run."""
    clusters: List[ClusterSpec]


@dataclass(frozen=True)
class ArtifactPaths:
    """The four per-cluster files consumed by ``rade_ml``."""
    elementary_pnl: Path
    elementary_attributes: Path
    target_pnl: Path
    target_attributes: Path


@dataclass(frozen=True)
class ArtifactManifest:
    """Run-level manifest: every cluster's paths + metadata (the ``jobs.pkl`` payload)."""
    entries: List[Dict[str, Any]]
    root: Path
