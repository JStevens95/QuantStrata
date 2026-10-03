"""
Core domain types for the static replication library.

Architecture: Portfolio-first preprocessing.

The pipeline loads ALL assets, generates ALL trades, and computes ALL PnL
ONCE at portfolio level.  Only then does it resolve clusters and populate
them with the relevant pre-computed data.

Asset objects are self-contained — an FXAsset loads its own domestic
and foreign IR curves internally via ``_load_dependencies()``.

A cluster can contain one or more risk factors depending on the
clustering method.  Simple key grouping (e.g. by ccy_pair) yields
1:1 clusters, while k-means or other methods can group multiple
risk factors into a single cluster.

Every inter-stage handoff uses a typed dataclass from this module.
This module must have ZERO imports from other rade_sr layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Cluster filesystem paths
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterPaths:
    """Immutable set of four artifact paths for a single cluster.

    Frozen so they cannot be accidentally mutated after resolution.
    """
    target_pnl: Path
    target_attributes: Path
    elem_pnl: Path
    elem_attributes: Path


# ─────────────────────────────────────────────────────────────────────
# Elementary trade (portfolio-level — no cluster affiliation)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ElementaryTrade:
    """One elementary (replicating) instrument in the trade universe.

    Trades are generated once at portfolio level, keyed by the risk
    factor they belong to, then populated per cluster in Phase 2.

    Parameters
    ----------
    trade_id : str
        Globally unique trade identifier.
    asset_class : str
        Asset class of the underlying risk factor.
    payoff_type : str
        Instrument payoff type (e.g. ``"call"``, ``"put"``, ``"forward"``).
    factor_id : str
        The risk factor this trade is priced against.
    parameters : dict
        Instrument-specific parameters.
    notional : float
        Trade notional (default 1.0 for unit replication).
    """
    trade_id: str
    asset_class: str
    payoff_type: str
    factor_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    notional: float = 1.0


# ─────────────────────────────────────────────────────────────────────
# PnL result for one elementary trade
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ElementaryTradePnL:
    """PnL vector for a single elementary trade across all scenarios.

    Parameters
    ----------
    trade_id : str
        Must match the corresponding ``ElementaryTrade.trade_id``.
    factor_id : str
        The risk factor this PnL was computed against.
    pnl_vector : np.ndarray
        Shape ``(n_scenarios,)`` — one PnL value per shock scenario.
    """
    trade_id: str
    factor_id: str
    pnl_vector: np.ndarray


# ─────────────────────────────────────────────────────────────────────
# PortfolioData — single source of truth for the whole portfolio
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioData:
    """Complete portfolio-level data computed once in Phase 1.

    All three dicts share the same key space — the risk factor name
    (e.g. ``"EURUSD"``, ``"GBPUSD"``).

    A cluster may contain one or many risk factors.  The ``slice()``
    method returns the subset of all three dicts for a given set of
    risk factor keys.

    Parameters
    ----------
    assets : dict[str, Any]
        ``risk_factor → loaded Asset`` (self-contained with dependencies).
    elementary_trades : dict[str, list[ElementaryTrade]]
        ``risk_factor → [trades generated for that factor]``.
    trade_pnls : dict[str, dict[str, ElementaryTradePnL]]
        ``risk_factor → {trade_id → PnL vector}``.
    """
    assets: Dict[str, Any]
    elementary_trades: Dict[str, List[ElementaryTrade]]
    trade_pnls: Dict[str, Dict[str, ElementaryTradePnL]]

    # ── Cluster population helper ──────────────────────────────────

    def slice(
        self, risk_factors: List[str],
    ) -> "PortfolioSlice":
        """Return the subset of all three dicts for the given risk factors.

        Used in Phase 2 to populate a cluster with its relevant data.
        """
        return PortfolioSlice(
            assets={
                rf: self.assets[rf]
                for rf in risk_factors if rf in self.assets
            },
            elementary_trades={
                rf: self.elementary_trades[rf]
                for rf in risk_factors if rf in self.elementary_trades
            },
            trade_pnls={
                rf: self.trade_pnls[rf]
                for rf in risk_factors if rf in self.trade_pnls
            },
        )

    # ── Flat views (for logging, export, etc.) ─────────────────────

    def all_trades_flat(self) -> List[ElementaryTrade]:
        """Flatten all trades across risk factors into a single list."""
        flat: List[ElementaryTrade] = []
        for trades in self.elementary_trades.values():
            flat.extend(trades)
        return flat

    def all_pnls_flat(self) -> Dict[str, ElementaryTradePnL]:
        """Flatten nested PnL dict into ``trade_id → PnL``."""
        flat: Dict[str, ElementaryTradePnL] = {}
        for factor_pnls in self.trade_pnls.values():
            flat.update(factor_pnls)
        return flat

    # ── Convenience properties ─────────────────────────────────────

    @property
    def n_factors(self) -> int:
        """Number of loaded risk factors."""
        return len(self.assets)

    @property
    def n_trades(self) -> int:
        """Total elementary trades across the portfolio."""
        return sum(len(trades) for trades in self.elementary_trades.values())

    @property
    def n_scenarios(self) -> int:
        """Number of shock scenarios (from first PnL vector)."""
        for factor_pnls in self.trade_pnls.values():
            if factor_pnls:
                first_pnl = next(iter(factor_pnls.values()))
                return first_pnl.pnl_vector.shape[0]
        return 0

    @property
    def risk_factors(self) -> Set[str]:
        """Set of all risk factor names."""
        return set(self.assets.keys())


@dataclass
class PortfolioSlice:
    """Subset of PortfolioData for one cluster.

    Produced by ``PortfolioData.slice()`` — same three-dict structure
    but only the risk factors relevant to one cluster.
    """
    assets: Dict[str, Any]
    elementary_trades: Dict[str, List[ElementaryTrade]]
    trade_pnls: Dict[str, Dict[str, ElementaryTradePnL]]

    def all_trades_flat(self) -> List[ElementaryTrade]:
        """Flatten trades across risk factors into a single list."""
        flat: List[ElementaryTrade] = []
        for rf in sorted(self.elementary_trades):
            flat.extend(self.elementary_trades[rf])
        return flat

    def all_pnls_flat(self) -> Dict[str, ElementaryTradePnL]:
        """Flatten nested PnL dict into ``trade_id → PnL``."""
        flat: Dict[str, ElementaryTradePnL] = {}
        for factor_pnls in self.trade_pnls.values():
            flat.update(factor_pnls)
        return flat


# ─────────────────────────────────────────────────────────────────────
# ReplicationJob — final output consumed by downstream ML
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ReplicationJob:
    """Final pipeline output for one cluster — what downstream ML consumes.

    A cluster can contain one or many risk factors.  All three data
    dicts are keyed by risk factor name, matching the PortfolioData
    structure.

    Parameters
    ----------
    cluster_id : str
        Cluster identifier.
    cluster_paths : ClusterPaths or None
        On-disk artifact paths for this cluster.
    assets : dict[str, Any]
        ``risk_factor → self-contained Asset`` for this cluster.
    elementary_trades : dict[str, list[ElementaryTrade]]
        ``risk_factor → [trades]`` for this cluster.
    trade_pnls : dict[str, dict[str, ElementaryTradePnL]]
        ``risk_factor → {trade_id → PnL}`` for this cluster.
    target_pnl : pd.DataFrame or None
        Target trade PnL for this cluster (from external risk system).
        Rows = target trades, columns = scenarios.
    target_attributes : pd.DataFrame or None
        Target trade attributes for this cluster (desk, product type,
        ccy, notional, etc.).
    elementary_attributes : pd.DataFrame or None
        Derived attributes for the elementary trades in this cluster
        (strike, expiry, payoff_type, notional, etc.).
    metadata : dict
        Arbitrary cluster metadata (origin, method, etc.).
    """
    cluster_id: str
    cluster_paths: Optional[ClusterPaths]
    assets: Dict[str, Any]
    elementary_trades: Dict[str, List[ElementaryTrade]]
    trade_pnls: Dict[str, Dict[str, ElementaryTradePnL]]
    target_pnl: Optional[pd.DataFrame] = None
    target_attributes: Optional[pd.DataFrame] = None
    elementary_attributes: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def pnl_matrix(self) -> np.ndarray:
        """Build ``(n_trades, n_scenarios)`` matrix.

        Rows ordered by risk factor (sorted), then by trade order
        within each factor.  ``pnl_matrix()[i]`` corresponds to
        ``all_trades_flat()[i]``.
        """
        pnl_flat = self.all_pnls_flat()
        return np.stack([
            pnl_flat[t.trade_id].pnl_vector
            for t in self.all_trades_flat()
        ])

    def all_trades_flat(self) -> List[ElementaryTrade]:
        """Flatten trades into a single list (sorted by risk factor)."""
        flat: List[ElementaryTrade] = []
        for rf in sorted(self.elementary_trades):
            flat.extend(self.elementary_trades[rf])
        return flat

    def all_pnls_flat(self) -> Dict[str, ElementaryTradePnL]:
        """Flatten nested PnL dict into ``trade_id → PnL``."""
        flat: Dict[str, ElementaryTradePnL] = {}
        for factor_pnls in self.trade_pnls.values():
            flat.update(factor_pnls)
        return flat

    @property
    def n_trades(self) -> int:
        """Total elementary trades across all risk factors in this cluster."""
        return sum(len(ts) for ts in self.elementary_trades.values())

    @property
    def n_scenarios(self) -> int:
        """Number of shock scenarios."""
        for factor_pnls in self.trade_pnls.values():
            if factor_pnls:
                first_pnl = next(iter(factor_pnls.values()))
                return first_pnl.pnl_vector.shape[0]
        return 0

    @property
    def trade_ids(self) -> List[str]:
        """Trade IDs in flattened order."""
        return [t.trade_id for t in self.all_trades_flat()]

    @property
    def risk_factor_ids(self) -> List[str]:
        """Risk factor names in this cluster."""
        return list(self.assets.keys())

    def to_job_dict(self) -> Dict[str, Any]:
        """Convert to the legacy dict format expected by rade_ml_pt.

        TODO(wire): Adapt keys to match your build_dataset() expectations.
        """
        cluster_info: Dict[str, str] = {}
        if self.cluster_paths is not None:
            cluster_info = {
                "target_pnl_path": str(self.cluster_paths.target_pnl),
                "target_attribs_path": str(self.cluster_paths.target_attributes),
                "elementary_pnl_path": str(self.cluster_paths.elem_pnl),
                "elementary_attribs_path": str(self.cluster_paths.elem_attributes),
            }
        return {
            "cluster_info": cluster_info,
            "cluster_id": self.cluster_id,
            "assets": self.assets,
            "n_trades": self.n_trades,
            "n_scenarios": self.n_scenarios,
        }
