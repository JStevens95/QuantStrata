"""
Core domain types for the static replication model.

The pipeline loads all assets, generates all trades and computes all pnl at portfolio level.
Only then does it resolve clusters and slice the pre-computed PortfolioData into lightweight cluster shells
(ReplicationJobs instance). This eliminates redundant per-cluster pricing when risk factors are shared across clusters.

Every inter-stage handoff uses a typed dataclass from this module. No raw dicts between stages - a typo on a dataclass
attribute is caught immediately by the IDE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from src.static_replication.core.exceptions import ValidationError

if TYPE_CHECKING:
    from src.static_replication.portfolio.matching import MatchResult


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
        return [fid for fid, spec in self.specs.items() if spec.is_primary]

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


@dataclass(frozen=True)
class PortfolioLoadResult:
    """Bundle returned by :meth:`Orchestrator.load_portfolio`."""

    # define parameters.
    attributes: pd.DataFrame
    pnl: pd.DataFrame
    match_result: "MatchResult"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a string summarizing the results."""
        return (f"PortfolioLoadResult(trades={len(self.attributes)}, "
                f"pnl_dates={self.pnl.shape[1] if not self.pnl.empty else 0}, "
                f"match_coverage={self.match_result.coverage:.1%})")


@dataclass(frozen=True)
class ClusterPaths:
    """
    Immutable set of four artifact paths for a single cluster.

    These paths point to the on-disk locations where the pipeline will read / save target & elementary pnl and
    attribute data. Frozen so they cannot be accidentally mutated after resolution.
    """

    # define parameters.
    target_pnl: Path
    target_attributes: Path

    elementary_pnl: Path
    elementary_attributes: Path


@dataclass(frozen=True)
class ElementaryTrade:
    """
    A single elementary (replicating) instrument in the trade universe.

    All pricing inputs are fully typed - no parameters: Dict[str, Any]. Trades are generated once at portfolio level
    and later sliced into cluster shells via PortfolioData.trades_for_factors()
    """

    # define parameters.
    trade_id: str
    asset_class: str    # FX | IR | EQ | CR
    payoff_type: str    # EC | EP | DC | DP | QC | QP
    factor_id: str
    notional: float = 1.0
    spot: float = 0.0
    strike: float = 0.0
    maturity: float = 0.0
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    vol: float = 0.0
    shocks: Dict[str, float] = field(default_factory=dict)


def make_trade_id(factor_id: str, code: str, spot: float, strike: float, maturity: float) -> str:
    """Construct a canonical elementary trade ID string."""
    return f"{factor_id}|{code}|S({spot:.3f})|K({strike:.3f})|T({maturity:.3f})"


@dataclass
class ElementaryTradePnL:
    """PnL vector for a single elementary trade across all scenarios."""

    # define parameters.
    trade_id: str
    factor_id: str
    pnl_vector: np.ndarray


@dataclass
class PortfolioSlice:
    """
    Subset of PortfolioData for one cluster.

    Produced by ``PortfolioData.slice()`` and sued to populate a ReplicationJob for a single cluster.
    """

    # define parameters.
    assets: Dict[str, Any]
    elementary_trades: Dict[str, List[ElementaryTrade]]
    elementary_trade_pnl: Dict[str, Dict[str, ElementaryTradePnL]]

    def all_trades_flat(self) -> List[ElementaryTrade]:
        """Flatten trades across risk factors into a single list."""
        flat: List[ElementaryTrade] = []
        for rf in sorted(self.elementary_trades):
            flat.extend(self.elementary_trades[rf])
        return flat

    def all_pnls_flat(self) -> Dict[str, ElementaryTradePnL]:
        """Flatten nested PnL dict in ``trade_id --> Pnl``."""
        flat: Dict[str, ElementaryTradePnL] = {}
        for factor_pnls in self.elementary_trade_pnl.values():
            flat.update(factor_pnls)
        return flat


@dataclass
class PortfolioData:
    """
    Complete portfolio-level data computed once.

    After phase 1 the pipeline holds exactly one PortfolioData containing:
        - Every loaded asset.
        - Every generated trade and every compute PnL vector.
        - Portfolio attributes and PnL sourced.

    Phase 2 merely *slices* this object into per-cluster ReplicationJob shells - no recomputation needed.
    """

    # define parameters.
    assets: Dict[str, Any]
    elementary_trades: Dict[str, List[ElementaryTrade]]
    elementary_trade_pnl: Dict[str, Dict[str, ElementaryTradePnL]]

    # SAGE sourced portfolio data.
    portfolio_attributes: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    as_of_date: Optional[date] = None

    def validate(self) -> None:
        """Validate that portfolio data is populated."""
        if self.portfolio_attributes.empty:
            raise ValidationError("PortfolioData.portfolio_attributes is empty - check SAGE client")
        if self.portfolio_pnl.empty:
            raise ValidationError("PortfolioData.portfolio_pnl is empty - check SAGE client")

    def slice(self, risk_factors: List[str]) -> "PortfolioSlice":
        """Return the subset of all three dicts for the given risk factors."""
        return PortfolioSlice(
            assets={
                rf: self.assets[rf] for rf in risk_factors if rf in self.assets
            },
            elementary_trades={
                rf: self.elementary_trades[rf] for rf in risk_factors if rf in self.elementary_trades
            },
            elementary_trade_pnl={
                rf: self.elementary_trade_pnl[rf] for rf in risk_factors if rf in self.elementary_trade_pnl
            },
        )

    def all_trades_flat(self) -> List[ElementaryTrade]:
        """Flatten trades across risk factors into a single list, ordered by factor_id."""
        flat: List[ElementaryTrade] = []
        for rf in sorted(self.elementary_trades.keys()):
            flat.extend(self.elementary_trades[rf])
        return flat

    def all_pnls_flat(self) -> Dict[str, ElementaryTradePnL]:
        """Flatten nested PnL dict in ``trade_id --> Pnl``."""
        flat: Dict[str, ElementaryTradePnL] = {}
        for factor_pnls in self.elementary_trade_pnl.values():
            flat.update(factor_pnls)
        return flat

    @property
    def n_factors(self) -> int:
        """Number of loaded risk factors."""
        return len(self.assets)

    @property
    def n_trades(self):
        """Number of loaded trades."""
        return sum(len(trades) for trades in self.elementary_trades.values())

    @property
    def n_scenarios(self) -> int:
        """Number of shock scenarios (from first PnL vector), Assumes all PnL vectors have same length."""
        for factor_pnls in self.elementary_trade_pnl.values():
            if factor_pnls:
                first_pnl = next(iter(factor_pnls.values()))
                return first_pnl.pnl_vector.shape[0]
        return 0

    @property
    def risk_factors(self) -> Set[str]:
        """Set of risk factors."""
        return set(self.assets.keys())


@dataclass
class ReplicationJob:
    """
    Final pipeline output for one cluster - what downstream ML consumes.

    A cluster can contain one or many risk factors. All three data dicts are keyed by risk factor name, matching the
    PortfolioData structure. The PnL dict is nested one level deeper to map risk_factor -> trade_id -> PnL vector.
    """

    # define parameters.
    cluster_id: str
    paths: Optional[ClusterPaths]
    assets: Dict[str, Any]
    elementary_trades: Dict[str, List[ElementaryTrade]]
    elementary_trade_pnl: Dict[str, Dict[str, ElementaryTradePnL]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_pnl: Optional[pd.Series] = None

    def pnl_matrix(self) -> np.ndarray:
        """
        Build ``(n_trades, n_scenarios)`` pnl matrix.

        Rows ordered by risk factor (sorted), then by trade order within each factor ``pnl_matrix()[i]``
        corresponds to ``all_trades_flat()[i]``.

        :return:
        """
        pnl_flat = self.all_pnls_flat()
        return np.stack([
            pnl_flat[t.trade_id].pnl_vector for t in self.all_trades_flat()
        ])

    def all_trades_flat(self) -> List[ElementaryTrade]:
        """Flatten trades across risk factors into a single list."""
        flat: List[ElementaryTrade] = []
        for rf in sorted(self.elementary_trades):
            flat.extend(self.elementary_trades[rf])
        return flat

    @property
    def trade_ids(self) -> List[str]:
        """Trade IDs in flattened order."""
        return [t.trade_id for t in self.all_trades_flat()]

    @property
    def risk_factor_ids(self) -> List[str]:
        """Risk factor names in this cluster."""
        return list(self.assets.keys())

    def to_job_dict(self) -> Dict[str, Any]:
        """Convert to the raw dict format expected by downstream consumers."""
        trades = self.all_trades_flat()
        if not trades:
            return {
                "cluster_id": self.cluster_id,
                "trade_ids": [],
                "pnl_matrix": [],
                "risk_factor_ids": self.risk_factor_ids,
                "metadata": self.metadata,
            }
        return {
            "cluster_id": self.cluster_id,
            "trade_ids": self.trade_ids,
            "pnl_matrix": self.pnl_matrix().tolist(),
            "risk_factor_ids": self.risk_factor_ids,
            "metadata": self.metadata,
        }
