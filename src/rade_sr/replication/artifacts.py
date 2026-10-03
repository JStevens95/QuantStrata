"""
Artifact serialisation — the rade_sr → rade_ml_pt handoff.

Writes the four per-cluster files that ``rade_ml_pt.data.hybrid_gnn_rnn.build_dataset``
consumes, plus a ``jobs.pkl`` manifest. The formats below are dictated by that
consumer (verified against ``data/hybrid_gnn_rnn/build.py`` + ``data/io.py``):

  - **PnL files** (``*_pnl``): Parquet, oriented **[scenarios × trade-ids]**
    (rows = scenarios, columns = trade ids). This is the transpose of
    ``ReplicationJob.pnl_matrix()`` (which is trades × scenarios).
    Elementary column ids keep the ``UND|TYPE|...`` shape so the consumer's
    ``col.split("|")`` grouping in dimensionality-reduction works.

  - **Attribute files** (``*_attributes``): pickled ``Dict[str, list]``
    (column-oriented; one list per attribute, all the same length). Must contain
    ``trade_id`` and ``yrs_to_maturity`` keys; other keys must cover your
    ``AttributeEncoderConfig`` numeric/categorical/multi-label keys.

  - **Manifest** (``jobs.pkl``): ``list[dict]`` with each cluster's
    ``cluster_info`` (the four paths) — the ``job`` dicts ``build_dataset`` takes.

Scenario alignment: elementary (shock) and target PnL must share the scenario
(time) axis. When the counts match, the elementary frame is re-indexed onto the
target's scenario labels; a mismatch is logged loudly (a real-env data issue).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rade_sr.core.types import ClusterPaths, ReplicationJob

logger = logging.getLogger(__name__)

# Attribute keys the downstream encoder always needs.
ID_KEY = "trade_id"
TTM_KEY = "yrs_to_maturity"

# Target attribute columns carried through from ingestion (best-effort; missing
# columns are simply skipped).
_TARGET_ATTR_COLS = (
    "AssetClass", "Product", "ProductGroup", "DeskName", "SubDeskName",
    "CallPut", "StrikePrice", "NotionalSign", "SignedNotional",
)


# ─────────────────────────────────────────────────────────────────────────
# Target-data attachment (per cluster, from Step 0 ingestion)
# ─────────────────────────────────────────────────────────────────────────

def attach_target_data(job: ReplicationJob, inputs: Any) -> None:
    """Slice the portfolio's target PnL + attributes onto one cluster's job.

    A cluster owns the target trades whose risk factor falls in the cluster's
    risk factors. Sets ``job.target_pnl`` (trades × scenarios, matching the
    dataclass convention) and ``job.target_attributes`` (one row per trade).
    """
    rfs = set(job.risk_factor_ids)
    rf_by_trade = inputs.risk_factor_by_trade
    target_ids = [str(t) for t, rf in rf_by_trade.items() if rf in rfs]

    if inputs.target_pnl is not None:
        present = [t for t in target_ids if t in inputs.target_pnl.index]
        job.target_pnl = inputs.target_pnl.loc[present]
    if inputs.attributes is not None:
        present_a = [t for t in target_ids if t in inputs.attributes.index]
        job.target_attributes = inputs.attributes.loc[present_a]

    n = 0 if job.target_pnl is None else len(job.target_pnl)
    logger.info("Cluster %s: attached %d target trades", job.cluster_id, n)


# ─────────────────────────────────────────────────────────────────────────
# Builders (job → consumer-shaped objects)
# ─────────────────────────────────────────────────────────────────────────

def build_elementary_pnl_df(
    job: ReplicationJob, scenario_index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Elementary PnL as [scenarios × trade-ids] (transpose of pnl_matrix)."""
    trades = job.all_trades_flat()
    if not trades:
        return pd.DataFrame()
    matrix = job.pnl_matrix()  # (n_trades, n_scenarios)
    ids = [t.trade_id for t in trades]
    df = pd.DataFrame(matrix.T, columns=ids)
    if scenario_index is not None and len(scenario_index) == len(df.index):
        df.index = scenario_index
    df.index.name = "scenario"
    return df


def build_elementary_attributes(job: ReplicationJob) -> Dict[str, List[Any]]:
    """Elementary trade attributes as a column-oriented ``Dict[str, list]``.

    Flattens each trade's pricer parameters and adds ``trade_id``,
    ``underlying``, ``product_type``, ``yrs_to_maturity``, ``notional``.
    """
    trades = job.all_trades_flat()
    attrs: Dict[str, List[Any]] = {
        ID_KEY: [], "underlying": [], "product_type": [], TTM_KEY: [], "notional": [],
    }
    for t in trades:
        params = dict(t.parameters or {})
        attrs[ID_KEY].append(t.trade_id)
        attrs["underlying"].append(t.factor_id)
        attrs["product_type"].append(t.payoff_type)
        attrs[TTM_KEY].append(float(params.get("expiry", params.get("T", np.nan))))
        attrs["notional"].append(float(t.notional))
        for k, v in params.items():
            attrs.setdefault(k, [np.nan] * len(attrs[ID_KEY][:-1]))
            attrs[k].append(v)
    # right-pad any param keys that appeared late so all lists are equal length
    n = len(attrs[ID_KEY])
    for k, v in attrs.items():
        if len(v) < n:
            attrs[k] = v + [np.nan] * (n - len(v))
    return attrs


def build_target_pnl_df(
    job: ReplicationJob, scenario_index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Target PnL as [scenarios × trade-ids] (transpose of stored trades × scenarios)."""
    if job.target_pnl is None or job.target_pnl.empty:
        return pd.DataFrame()
    df = job.target_pnl.T.copy()  # scenarios × trades
    if scenario_index is not None and len(scenario_index) == len(df.index):
        df.index = scenario_index
    df.index.name = "scenario"
    df.columns = [str(c) for c in df.columns]
    return df


def build_target_attributes(job: ReplicationJob) -> Dict[str, List[Any]]:
    """Target trade attributes as a column-oriented ``Dict[str, list]``."""
    ta = job.target_attributes
    if ta is None or len(ta) == 0:
        return {ID_KEY: [], TTM_KEY: []}

    ids = [str(i) for i in ta.index]
    attrs: Dict[str, List[Any]] = {ID_KEY: ids}
    if "expiry_years" in ta.columns:
        attrs[TTM_KEY] = [float(x) for x in ta["expiry_years"].to_numpy()]
    else:
        attrs[TTM_KEY] = [np.nan] * len(ids)
    for col in _TARGET_ATTR_COLS:
        if col in ta.columns:
            attrs[col] = ta[col].tolist()
    return attrs


# ─────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────

def _write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)  # preserves the scenario index


def save_cluster_artifacts(job: ReplicationJob) -> ClusterPaths:
    """Write the four artifact files for one cluster to ``job.cluster_paths``.

    Returns the ``ClusterPaths`` written. Raises if paths are unset.
    """
    if job.cluster_paths is None:
        raise ValueError(f"Cluster {job.cluster_id!r} has no cluster_paths to write to")
    paths = job.cluster_paths

    # Build target first so we can share its scenario axis with the elementary frame.
    target_pnl = build_target_pnl_df(job)
    scen_index = target_pnl.index if not target_pnl.empty else None
    elem_pnl = build_elementary_pnl_df(job, scenario_index=scen_index)

    if (not elem_pnl.empty and not target_pnl.empty
            and len(elem_pnl.index) != len(target_pnl.index)):
        logger.warning(
            "Cluster %s: elementary scenarios (%d) != target scenarios (%d); "
            "the elementary shocks and target PnL must share the scenario axis.",
            job.cluster_id, len(elem_pnl.index), len(target_pnl.index),
        )

    _write_parquet(elem_pnl, Path(paths.elem_pnl))
    _write_parquet(target_pnl, Path(paths.target_pnl))
    _write_pickle(build_elementary_attributes(job), Path(paths.elem_attributes))
    _write_pickle(build_target_attributes(job), Path(paths.target_attributes))

    # cache derived frames on the job for in-process consumers / inspection
    job.elementary_attributes = elem_pnl  # PnL frame for convenience; attrs are on disk
    logger.info(
        "Cluster %s: wrote artifacts (%d elem trades, %d target trades, %d scenarios)",
        job.cluster_id, elem_pnl.shape[1] if not elem_pnl.empty else 0,
        target_pnl.shape[1] if not target_pnl.empty else 0,
        elem_pnl.shape[0] if not elem_pnl.empty else 0,
    )
    return paths


def build_job_manifest(jobs: List[ReplicationJob]) -> List[Dict[str, Any]]:
    """Build the ``list[dict]`` manifest of ``job`` dicts for ``build_dataset``.

    Each entry carries ``cluster_info`` (the four artifact paths) plus light
    metadata. The loaded Asset objects are intentionally excluded.
    """
    manifest: List[Dict[str, Any]] = []
    for job in jobs:
        cluster_info: Dict[str, str] = {}
        if job.cluster_paths is not None:
            cluster_info = {
                "target_pnl_path": str(job.cluster_paths.target_pnl),
                "elementary_pnl_path": str(job.cluster_paths.elem_pnl),
                "target_attribs_path": str(job.cluster_paths.target_attributes),
                "elementary_attribs_path": str(job.cluster_paths.elem_attributes),
            }
        manifest.append({
            "cluster_id": job.cluster_id,
            "cluster_info": cluster_info,
            "risk_factors": job.risk_factor_ids,
            "n_trades": job.n_trades,
            "n_scenarios": job.n_scenarios,
        })
    return manifest


def save_jobs_manifest(jobs: List[ReplicationJob], path: Path) -> Path:
    """Write the ``jobs.pkl`` manifest consumed by the training entry point."""
    manifest = build_job_manifest(jobs)
    _write_pickle(manifest, Path(path))
    logger.info("Wrote jobs manifest with %d clusters to %s", len(manifest), path)
    return Path(path)
