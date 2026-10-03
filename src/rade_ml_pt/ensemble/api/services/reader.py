"""Read-side counterpart to the evaluation pipeline's writers.

One :class:`ArtifactReader` instance is constructed at server startup and
shared across all routers via FastAPI ``Depends(get_reader)``.  Its
methods return pandas DataFrames (parquet artifacts) or plain dicts
(JSON metadata).

Hot paths (``cluster_attributes``, ``ensemble_metrics``,
``per_member_metrics``, ``graph_stats``) and per-split parquets go
through the mtime-keyed cache in
:mod:`src.rade_ml_pt.ensemble.api.services.cache`, so repeated reads of
the same file inside one evaluation run are served from RAM.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.rade_ml_pt.ensemble.api.models.trade_graph import (
    TradeGraphEdge,
    TradeGraphNode,
    TradeGraphResponse,
    TradeGraphStats,
)
from src.rade_ml_pt.ensemble.api.services.cache import read_with_mtime_cache
from src.rade_ml_pt.ensemble.api.services.paths import ArtifactPaths

logger = logging.getLogger(__name__)


# ── Module-level loaders ──────────────────────────────────────────
# Must be module-level (not lambdas / closures) so the lru_cache inside
# read_with_mtime_cache keys on a stable callable identity.

def _load_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_joblib(path: Path) -> Any:
    """Load a joblib file — deferred import to keep ``joblib`` optional.

    Only the trade-graph reader needs joblib, and a minority of API
    deployments will ever call it.  Importing at module load would pull
    ``joblib`` into every FastAPI boot.
    """
    import joblib
    return joblib.load(path)


# Canonical scaled / original PnL-measure column pairs for the
# portfolio + cluster timeseries parquets (B1 / B2 v3 schemas).
# Used by :func:`_apply_space` to present a single canonical view to
# routers regardless of which PnL space the caller selected.
_PNL_MEASURE_COLUMNS = (
    "predictions",
    "targets",
    "error",
    "abs_error",
    "squared_error",
)


def _apply_space(df: pd.DataFrame, space: str) -> pd.DataFrame:
    """Project a timeseries parquet onto one PnL space's columns.

    The B1 / B2 parquets carry both scaled (``predictions``, …) and
    original (``predictions_original``, …) measure columns.  Callers
    select one space via the ``space`` parameter; this helper returns
    a DataFrame whose canonical measure columns reflect the chosen
    space, so every downstream router can keep reading
    ``df["predictions"]`` etc. regardless.

    Parameters
    ----------
    df    : pd.DataFrame
        The raw timeseries DataFrame as read off disk.
    space : ``"scaled"`` | ``"original"``
        Which PnL space to surface as the canonical measure columns.

    Returns
    -------
    pd.DataFrame
        - When ``space == "scaled"``: the original frame with the
          ``*_original`` columns dropped.
        - When ``space == "original"``: ``*_original`` columns
          renamed to their canonical names and the scaled originals
          dropped.  Values may be NaN when no cluster contributed
          original-space data (writers NaN-fill in that case).

    Raises
    ------
    ValueError
        Unknown ``space`` value.
    """
    if space == "scaled":
        drop = [f"{c}_original" for c in _PNL_MEASURE_COLUMNS]
        return df.drop(columns=[c for c in drop if c in df.columns])

    if space == "original":
        keep_identity = [c for c in df.columns if not c.endswith("_original") and c not in _PNL_MEASURE_COLUMNS]
        rename = {f"{c}_original": c for c in _PNL_MEASURE_COLUMNS}
        # Build the projected frame: identity columns + renamed measures.
        out = df[keep_identity + list(rename.keys())].rename(columns=rename)
        return out

    raise ValueError(
        f"Unknown space '{space}'. Expected one of: 'scaled', 'original'."
    )


class ArtifactReader:
    """Typed accessor over one ensemble evaluation directory."""

    def __init__(self, paths: ArtifactPaths):
        self.paths = paths

    # ── Run metadata ──────────────────────────────────────────────
    def manifest(self) -> Dict[str, Any]:
        """Load ``manifest.json`` and return its top-level dict.

        Tolerates a legacy writer that wrapped the manifest dict in a
        single-element list (``[{...}]``) — unwraps it so every caller
        sees the canonical dict shape regardless of how it was written.
        Raises ``ValueError`` for any other unexpected shape so problems
        surface loudly at the API boundary, not deep inside a router.
        """
        raw = read_with_mtime_cache(self.paths.manifest, _load_json)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
            return raw[0]
        raise ValueError(
            f"manifest.json at {self.paths.manifest} has unexpected shape "
            f"{type(raw).__name__!r}; expected a top-level dict (or a "
            f"single-element list wrapping a dict)."
        )

    def trade_cluster_map(self) -> Dict[str, str]:
        return read_with_mtime_cache(self.paths.trade_cluster_map, _load_json)

    # ── Ensemble-scoped parquets ──────────────────────────────────
    def cluster_attributes(self) -> pd.DataFrame:
        return read_with_mtime_cache(self.paths.cluster_attributes, _load_parquet)

    def ensemble_metrics(self) -> pd.DataFrame:
        return read_with_mtime_cache(self.paths.ensemble_metrics, _load_parquet)

    def per_member_metrics(
        self, *, split: Optional[str] = None, cluster_id: Optional[str] = None,
    ) -> pd.DataFrame:
        df = read_with_mtime_cache(self.paths.per_member_metrics, _load_parquet)
        if split is not None:
            df = df[df["split"] == split]
        if cluster_id is not None:
            df = df[df["cluster_id"] == cluster_id]
        return df

    def graph_stats(
        self, *, cluster_id: Optional[str] = None,
    ) -> pd.DataFrame:
        df = read_with_mtime_cache(self.paths.graph_stats, _load_parquet)
        if cluster_id is not None:
            df = df[df["cluster_id"] == cluster_id]
        return df

    # ── Per-split parquets ────────────────────────────────────────
    def portfolio_timeseries(
        self, split: str, *, space: str = "scaled",
    ) -> pd.DataFrame:
        """Return the portfolio timeseries for one split, projected onto
        the requested PnL space.

        ``space`` must be ``"scaled"`` (legacy default) or ``"original"``.
        The returned DataFrame's measure columns always carry their
        canonical names (``predictions``, ``targets``, ``error``,
        ``abs_error``, ``squared_error``) — see :func:`_apply_space`.
        """
        df = read_with_mtime_cache(
            self.paths.portfolio_timeseries(split), _load_parquet,
        )
        return _apply_space(df, space)

    def cluster_timeseries(
        self,
        split:      str,
        *,
        cluster_id: Optional[str] = None,
        space:      str            = "scaled",
    ) -> pd.DataFrame:
        """Return the per-cluster timeseries for one split, optionally
        filtered to a single cluster and projected onto the requested
        PnL space.

        Same column-canonicalisation contract as
        :meth:`portfolio_timeseries` — callers see the canonical names
        regardless of ``space``.
        """
        df = read_with_mtime_cache(
            self.paths.cluster_timeseries(split), _load_parquet,
        )
        if cluster_id is not None:
            df = df[df["cluster_id"] == cluster_id]
        return _apply_space(df, space)

    def trade_metrics(
        self, split: str, *, cluster_id: Optional[str] = None,
    ) -> pd.DataFrame:
        df = read_with_mtime_cache(
            self.paths.trade_metrics(split), _load_parquet,
        )
        if cluster_id is not None:
            df = df[df["cluster_id"] == cluster_id]
        return df

    def group_correlations(
        self, split: str, *, attribute: Optional[str] = None,
    ) -> pd.DataFrame:
        df = read_with_mtime_cache(
            self.paths.group_correlations(split), _load_parquet,
        )
        if attribute is not None:
            df = df[df["attribute"] == attribute]
        return df

    def group_correlation_attributes(self, split: str) -> List[str]:
        """Distinct ``attribute`` values present in one split's parquet."""
        df = read_with_mtime_cache(
            self.paths.group_correlations(split), _load_parquet,
        )
        return sorted(df["attribute"].unique().tolist())

    def completeness(
        self, split: str, *, cluster_id: Optional[str] = None,
    ) -> pd.DataFrame:
        df = read_with_mtime_cache(
            self.paths.completeness(split), _load_parquet,
        )
        if cluster_id is not None:
            df = df[df["cluster_id"] == cluster_id]
        return df

    def feature_summary(
        self, split: str, *, cluster_id: Optional[str] = None,
    ) -> pd.DataFrame:
        df = read_with_mtime_cache(
            self.paths.feature_summary(split), _load_parquet,
        )
        if cluster_id is not None:
            df = df[df["cluster_id"] == cluster_id]
        return df

    # ── Trade graph (joblib + trade_universe) ─────────────────────
    def trade_graph(self, cluster_id: str) -> TradeGraphResponse:
        """Build the full trade-graph payload for one cluster.

        Reads ``members/{cluster_id}/graph_results.joblib`` and the
        adjacent ``trade_universe.json``; both are staged by the eval
        pipeline (``_copy_member_graph_artifacts``).

        * Node IDs come from ``target_ids + elementary_ids`` in the
          trade universe, indexed by position — the same ordering the
          graph builder used when it assembled the sparse adjacency.
        * Node class (``target`` vs ``elementary``) is looked up in the
          trade universe; nodes whose index falls outside both lists
          are omitted and the payload carries a warning.
        * Self-loops (``i == j``) are dropped — the graph builder
          sometimes emits a unit-weight diagonal.

        Raises
        ------
        FileNotFoundError
            If ``graph_results.joblib`` is missing for the cluster.
        """
        graph_path = self.paths.member_graph_results(cluster_id)
        tu_path = self.paths.member_trade_universe(cluster_id)

        if not graph_path.exists():
            raise FileNotFoundError(
                f"graph_results.joblib missing for cluster '{cluster_id}' "
                f"(looked for {graph_path}). Re-run the eval pipeline to "
                f"stage member graph artefacts."
            )

        gr = read_with_mtime_cache(graph_path, _load_joblib)

        indices = np.asarray(gr.get("sparse_indices", []), dtype=int)
        values = np.asarray(gr.get("sparse_values", []), dtype=float)
        shape = gr.get("sparse_shape", [0, 0])
        n = int(shape[0]) if shape and shape[0] > 0 else 0

        warnings: List[str] = []

        # Trade universe — optional; node-type classification degrades
        # gracefully to "elementary" when absent so the graph still
        # renders.
        target_ids: List[str] = []
        elementary_ids: List[str] = []
        if tu_path.exists():
            try:
                tu = read_with_mtime_cache(tu_path, _load_json)
                target_ids = [str(x) for x in (tu.get("target_ids") or [])]
                elementary_ids = [
                    str(x) for x in (tu.get("elementary_ids") or [])
                ]
            except Exception as exc:
                warnings.append(
                    f"Could not read trade_universe.json: {exc!r}"
                )
        else:
            warnings.append(
                "trade_universe.json missing — all nodes classified as "
                "'elementary'. Re-run the eval pipeline to stage this file."
            )

        # Node id lookup by position in the graph-builder's ordering
        # (targets first, then elementaries).
        node_ordering = target_ids + elementary_ids
        target_set = set(target_ids)

        nodes: List[TradeGraphNode] = []
        for i in range(n):
            if i < len(node_ordering):
                tid = node_ordering[i]
                ttype = "target" if tid in target_set else "elementary"
            else:
                tid = f"{cluster_id}_trade_{i}"
                ttype = "elementary"
                warnings.append(
                    f"Node index {i} out of trade-universe range "
                    f"({len(node_ordering)}); synthetic id assigned."
                )
            nodes.append(
                TradeGraphNode(
                    trade_id=tid, cluster_id=cluster_id, trade_type=ttype,
                )
            )

        # Edges — drop self-loops, map node indices to trade ids.
        #
        # The graph builder has historically emitted ``sparse_indices``
        # in two shapes depending on the producer version:
        #
        # * **(2, n_edges)** — the canonical "COO row/col" stack used by
        #   ``scipy.sparse.coo_matrix`` and the current eval pipeline.
        # * **(n_edges, 2)** — the older "IJV pairs" layout still
        #   present in legacy ``graph_results.joblib`` files (the
        #   ``ensemble_analytics`` UI's adjacency loader handles both).
        #
        # We accept either; anything else gets a warning and yields
        # zero edges (rather than silently dropping every edge — that's
        # what masked the problem before).
        edges: List[TradeGraphEdge] = []
        if indices.size > 0 and indices.ndim == 2:
            if indices.shape[0] == 2:
                src_arr = indices[0]
                dst_arr = indices[1]
            elif indices.shape[1] == 2:
                src_arr = indices[:, 0]
                dst_arr = indices[:, 1]
            else:
                src_arr = dst_arr = None  # type: ignore[assignment]
                warnings.append(
                    "graph_results.joblib: sparse_indices has shape "
                    f"{tuple(indices.shape)}; expected (2, n_edges) "
                    "or (n_edges, 2).  No edges rendered."
                )

            if src_arr is not None and dst_arr is not None:
                for k in range(src_arr.shape[0]):
                    i, j = int(src_arr[k]), int(dst_arr[k])
                    if i == j:
                        continue
                    if i >= n or j >= n:
                        continue
                    edges.append(
                        TradeGraphEdge(
                            source=nodes[i].trade_id,
                            target=nodes[j].trade_id,
                            weight=float(values[k]),
                        )
                    )
        elif indices.size > 0:
            warnings.append(
                "graph_results.joblib: sparse_indices has unexpected "
                f"ndim={indices.ndim} (size={indices.size}). "
                "No edges rendered."
            )

        # Summary stats — density uses the full node count, matching B8.
        n_edges_total = int(values.size)
        density = float(n_edges_total / (n * n)) if n > 0 else 0.0
        mean_w = float(np.mean(values)) if values.size > 0 else 0.0

        n_target = sum(1 for node in nodes if node.trade_type == "target")
        n_elem = len(nodes) - n_target

        # Collapse near-duplicate warnings — any single missing node
        # id typically triggers the "out of range" path for every
        # orphan index, and the caller only needs to know it happened
        # once.
        dedup_warnings = sorted(set(warnings)) if warnings else None

        return TradeGraphResponse(
            cluster_id=cluster_id,
            n_target_trades=n_target,
            n_elementary_trades=n_elem,
            stats=TradeGraphStats(
                n_nodes=n,
                n_edges=n_edges_total,
                density=density,
                mean_weight=mean_w,
            ),
            nodes=nodes,
            edges=edges,
            warnings=dedup_warnings,
        )

    # ── Per-member training curves (parquet) ──────────────────────
    def training_curves(self, cluster_id: str) -> pd.DataFrame:
        """Per-epoch training curves for one cluster member.

        Staged by the eval pipeline from
        ``{registry_dir}/{member_version}/training_curves.parquet``.

        Always contains ``epoch`` + ``train_loss``; any additional
        per-epoch series the trainer emitted (``val_loss``, ``mae``,
        ``val_mae``, …) are preserved untouched.  The UI introspects
        the available metric columns at render time — the column set
        is deliberately open so new trainer metrics flow through
        without coordinated schema bumps.

        Raises
        ------
        FileNotFoundError
            If ``training_curves.parquet`` is missing for the cluster.
            Callers should return a 404 — the absence of curves
            typically means a legacy run or a pipeline that skipped
            the trainer-side writer.
        """
        path = self.paths.member_training_curves(cluster_id)
        if not path.exists():
            raise FileNotFoundError(
                f"training_curves.parquet missing for cluster "
                f"'{cluster_id}' (looked for {path}). Re-run the eval "
                f"pipeline to stage member training artefacts."
            )
        return read_with_mtime_cache(path, _load_parquet)

    def elementary_pnl(self, cluster_id: str) -> pd.DataFrame:
        """Per-scenario elementary-trade PnL for one cluster.

        Staged by the eval pipeline from
        ``{registry_dir}/{member_version}/elementary_pnl.parquet``
        alongside ``training_curves.parquet`` and ``trade_universe.json``.

        Returned as a wide DataFrame: one row per scenario, one
        column per elementary trade id, values are the raw PnL of
        that hedge instrument under each scenario.  Elementary
        trades are *model inputs* (atomic legs that compose the
        target structures), so values are not predictions / targets.

        Cached by mtime — same hot-path policy as
        :meth:`training_curves` because the file is small enough to
        sit in memory and rarely changes once the eval bundle is
        sealed.

        Raises
        ------
        FileNotFoundError
            If ``elementary_pnl.parquet`` is missing for the cluster
            (legacy eval bundle that didn't stage this file).
            Callers should return a 404; the UI degrades to an
            empty Elementary PnL Explorer placeholder rather than
            blocking the rest of the page.
        """
        path = self.paths.member_elementary_pnl(cluster_id)
        if not path.exists():
            raise FileNotFoundError(
                f"elementary_pnl.parquet missing for cluster "
                f"'{cluster_id}' (looked for {path}). Re-run the eval "
                f"pipeline to stage member graph + training artefacts."
            )
        return read_with_mtime_cache(path, _load_parquet)

    # ── Raw predictions (NPZ, streamed, not cached) ───────────────
    def member_predictions_path(
        self,
        cluster_id: str,
        split:      str,
        *,
        space:      str = "scaled",
    ) -> Path:
        """Return the NPZ path for a (cluster, split, PnL space) triple.

        Phase 3.4 — ``space`` selects between the legacy ``{split}.npz``
        (scaled) and the Phase 3.1 ``{split}_original.npz`` shard.  The
        file is not read or cached here — routers that need the raw
        predictions stream the file directly (potentially hundreds of MB).
        """
        return self.paths.member_predictions(cluster_id, split, space)

    # ── Convenience ───────────────────────────────────────────────
    def available_splits(self) -> List[str]:
        """Infer evaluated splits by scanning ``portfolio_timeseries_*.parquet``."""
        prefix = "portfolio_timeseries_"
        portfolio_dir = self.paths.eval_dir / "portfolio_summary"
        if not portfolio_dir.exists():
            return []
        return sorted(
            f.stem.removeprefix(prefix)
            for f in portfolio_dir.glob(f"{prefix}*.parquet")
        )
