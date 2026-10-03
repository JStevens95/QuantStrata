"""
Portfolio manager — aggregates ReplicationJob outputs into portfolio-level views.

After the preprocessing pipeline produces a list of ReplicationJobs (one
per cluster), this manager combines them into portfolio-wide structures:
trade catalogues, aggregated PnL matrices, and job manifests.

This is the bridge between rade_sr (preprocessing) and rade_ml_pt (ML training).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rade_sr.core.types import ReplicationJob

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Aggregates ReplicationJob outputs into portfolio-level structures.

    Parameters
    ----------
    jobs : list[ReplicationJob]
        Output from PreprocessingPipeline.run().
    """

    def __init__(self, jobs: List[ReplicationJob]) -> None:
        self._jobs = jobs
        self._job_map: Dict[str, ReplicationJob] = {j.cluster_id: j for j in jobs}

    @property
    def cluster_ids(self) -> List[str]:
        """Cluster IDs in pipeline order."""
        return [j.cluster_id for j in self._jobs]

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self._jobs)

    @property
    def total_trades(self) -> int:
        """Total elementary trades across all clusters."""
        return sum(j.n_trades for j in self._jobs)

    def get_job(self, cluster_id: str) -> ReplicationJob:
        """Retrieve a specific cluster's job."""
        return self._job_map[cluster_id]

    def build_trade_catalogue(self) -> pd.DataFrame:
        """Build a portfolio-wide trade catalogue DataFrame.

        Each row is one elementary trade with its cluster membership,
        asset class, payoff type, factor, and parameters.

        Returns
        -------
        pd.DataFrame
            Columns: trade_id, cluster_id, asset_class, payoff_type,
            factor_id, notional, + flattened parameters.
        """
        rows: List[Dict[str, Any]] = []
        for job in self._jobs:
            for trade in job.all_trades_flat():
                row = {
                    "trade_id": trade.trade_id,
                    "cluster_id": job.cluster_id,
                    "asset_class": trade.asset_class,
                    "payoff_type": trade.payoff_type,
                    "factor_id": trade.factor_id,
                    "notional": trade.notional,
                }
                row.update(trade.parameters)
                rows.append(row)
        return pd.DataFrame(rows)

    def build_portfolio_pnl_matrix(self) -> np.ndarray:
        """Build portfolio-wide PnL matrix by stacking cluster PnL matrices.

        Returns
        -------
        np.ndarray
            Shape ``(total_trades, n_scenarios)``.
            Row order matches ``build_trade_catalogue()`` row order.
        """
        matrices = [job.pnl_matrix() for job in self._jobs]
        return np.vstack(matrices)

    def build_cluster_mapping(self) -> Dict[str, List[str]]:
        """Build ``{cluster_id: [trade_ids]}`` mapping.

        This is the cluster_mapping format used by rade_ml_pt's
        EnsembleConfig.
        """
        return {
            job.cluster_id: job.trade_ids
            for job in self._jobs
        }

    def build_job_manifest(self) -> Dict[str, Any]:
        """Build a JSON-serialisable manifest for reproducibility.

        Contains cluster IDs, trade counts, factor counts, and
        total scenarios. Useful for audit trails and pipeline logging.
        """
        return {
            "n_clusters": self.n_clusters,
            "total_trades": self.total_trades,
            "cluster_summary": {
                job.cluster_id: {
                    "n_trades": job.n_trades,
                    "n_scenarios": job.n_scenarios,
                    "n_factors": len(job.risk_factor_ids),
                    "factor_ids": job.risk_factor_ids,
                }
                for job in self._jobs
            },
        }

    def to_ml_jobs(self) -> List[Dict[str, Any]]:
        """Convert all jobs to the dict format expected by rade_ml_pt.

        Returns
        -------
        list[dict]
            One dict per cluster in pipeline order, matching the
            ``job`` dict structure that ``build_dataset()`` expects.

        TODO(wire): Adjust the dict keys in ReplicationJob.to_job_dict()
        to match your exact rade_ml_pt build_dataset() interface.
        """
        return [job.to_job_dict() for job in self._jobs]
