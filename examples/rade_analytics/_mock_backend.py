"""Shared synthetic :class:`RadeBackend` for every preview script.

Each preview script (``05_overview_preview_live.py``,
``06_portfolio_preview_live.py``, …) builds the full Rade app and
injects this mock so callbacks can be exercised end-to-end without
running the FastAPI backend.

Design
------
* Every public method of :class:`RadeBackend` that any page consumes
  is overridden here.
* Data is deterministic — seeded once at construction — so screenshots
  are reproducible.
* :meth:`_cluster_timeseries` is the **single source** for synthetic
  per-cluster residuals; :meth:`portfolio_df` derives its aggregate
  rollup from it so the cluster-level and portfolio-level views stay
  internally consistent (important once the user starts filtering and
  comparing numbers between the two).
* Pydantic responses are constructed from the real model classes so
  any schema drift breaks at import-time instead of silently shipping
  a bad mock.

Import from a preview script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _mock_backend import MockRadeBackend
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rade_ml_pt.ensemble.api.models.clusters import (
    ClusterInfo,
    ClustersResponse,
)
from src.rade_ml_pt.ensemble.api.models.meta import (
    HealthResponse,
    VersionsResponse,
)
from src.rade_ml_pt.ensemble.api.models.trade_graph import (
    TradeGraphEdge,
    TradeGraphNode,
    TradeGraphResponse,
    TradeGraphStats,
)
from src.ui.apps.rade_analytics.data.backend import BackendResult, RadeBackend


# ─────────────────────────────────────────────────────────────────────
# Constants — vocabulary for synthetic cluster attributes
# ─────────────────────────────────────────────────────────────────────

_ASSET_CLASSES = ("rates", "fx", "credit", "equity")
_CURRENCIES    = ("USD", "EUR", "GBP", "JPY")
_DESKS         = ("Alpha", "Beta", "Gamma")
_PRODUCTS      = ("swap", "option", "forward", "bond")
_N_SCENARIOS   = 48


class MockRadeBackend(RadeBackend):
    """Deterministic in-memory backend for Dash UI preview scripts.

    Parameters
    ----------
    n_clusters
        How many synthetic clusters to generate.  Default 12 — enough
        to populate 3 desks × ~4 products without overwhelming the
        heatmap.
    seed
        Seeds both :mod:`random` and :mod:`numpy.random` so sessions
        are reproducible across the two RNGs we use.
    """

    def __init__(self, *, n_clusters: int = 12, seed: int = 42) -> None:
        # NOT calling ``super().__init__`` on purpose — the parent
        # constructor requires a client and cache, neither of which
        # makes sense here.  Every public method is overridden below.
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._seed = seed
        self._n_clusters = n_clusters
        self._cluster_ids: List[str] = [
            f"cluster_{i + 1:02d}" for i in range(n_clusters)
        ]
        self._versions: List[str] = [
            "v2026.04.17-a1b2c",
            "v2026.04.10-f3e4d7",
            "v2026.04.03-9876a2",
        ]
        # Stable attribute map — computed once, reused forever.
        self._cluster_attrs: Dict[str, Dict[str, Any]] = {
            cid: {
                "asset_class":   self._rng.choice(_ASSET_CLASSES),
                "currency_code": self._rng.choice(_CURRENCIES),
                "desk":          self._rng.choice(_DESKS),
                "product_code":  self._rng.choice(_PRODUCTS),
            }
            for cid in self._cluster_ids
        }
        # Stable per-cluster weights used when reconstructing the
        # portfolio rollup — weights sum to ~1.0 so portfolio-level
        # numbers stay on a reasonable scale.
        raw_weights = self._np_rng.uniform(0.2, 1.8, size=n_clusters)
        self._cluster_weights: Dict[str, float] = dict(
            zip(self._cluster_ids, raw_weights / raw_weights.sum() * n_clusters)
        )

    # ─────────────────────────────────────────────────────────────
    # Meta endpoints
    # ─────────────────────────────────────────────────────────────

    def health(self) -> BackendResult[HealthResponse]:
        return BackendResult.success(
            HealthResponse(
                status="ok",
                version=self._versions[0],
                artifacts_dir="/tmp/mock-artifacts",
            )
        )

    def versions(self) -> BackendResult[VersionsResponse]:
        return BackendResult.success(
            VersionsResponse(
                active=self._versions[0],
                available=list(self._versions),
            )
        )

    # ─────────────────────────────────────────────────────────────
    # Metrics endpoints
    # ─────────────────────────────────────────────────────────────

    def ensemble_metrics_df(self) -> BackendResult[pd.DataFrame]:
        rows: List[Dict[str, Any]] = []
        for split, scale in (("train", 0.9), ("val", 1.05), ("test", 1.15)):
            mae = 0.0012 * scale
            mse = (mae ** 2) * 1.4
            rmse = float(np.sqrt(mse))
            rows.append(
                {
                    "split":  split,
                    "mae":    mae,
                    "mse":    mse,
                    "rmse":   rmse,
                    "max_ae": mae * 18.0,
                    "p95_ae": mae * 4.5,
                    "p99_ae": mae * 9.0,
                }
            )
        return BackendResult.success(pd.DataFrame(rows))

    def per_member_metrics_df(
        self,
        *,
        split:      Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        # Derive per-cluster metrics from the synthetic timeseries so
        # rows 1-3 (aggregate) and rows 4-5 (grouped) agree on numbers.
        splits = [split] if split else ["train", "val", "test"]
        frames: List[pd.DataFrame] = []
        for s in splits:
            ts = self._cluster_timeseries(s)
            if cluster_id:
                ts = ts[ts["cluster_id"] == cluster_id]
            grouped = (
                ts.assign(
                    _abs_err=ts["abs_error"],
                    _sq_err=ts["squared_error"],
                )
                .groupby("cluster_id", as_index=False)
                .agg(
                    mae=("_abs_err", "mean"),
                    rmse_sq=("_sq_err", "mean"),
                    max_ae=("_abs_err", "max"),
                    p95_ae=("_abs_err", lambda x: float(np.quantile(x, 0.95))),
                    p99_ae=("_abs_err", lambda x: float(np.quantile(x, 0.99))),
                    n_scenarios=("scenario_idx", "nunique"),
                )
            )
            grouped["rmse"] = np.sqrt(grouped["rmse_sq"].astype(float))
            grouped["mse"] = grouped["rmse"] ** 2
            grouped["split"] = s
            grouped["n_targets"] = [
                self._rng.randint(120, 480) for _ in range(len(grouped))
            ]
            frames.append(
                grouped[
                    [
                        "cluster_id", "split", "mae", "mse", "rmse",
                        "max_ae", "p95_ae", "p99_ae",
                        "n_targets", "n_scenarios",
                    ]
                ]
            )
        if not frames:
            return BackendResult.success(pd.DataFrame())
        return BackendResult.success(pd.concat(frames, ignore_index=True))

    # ─────────────────────────────────────────────────────────────
    # Cluster endpoints
    # ─────────────────────────────────────────────────────────────

    def clusters(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[ClustersResponse]:
        entries: List[ClusterInfo] = []
        for cid in self._cluster_ids:
            if cluster_id and cid != cluster_id:
                continue
            # n_trades follows a deterministic pattern so re-runs match.
            n_trades = 80 + int(self._cluster_weights[cid] * 180)
            entries.append(
                ClusterInfo(
                    cluster_id=cid,
                    n_trades=n_trades,
                    attributes=dict(self._cluster_attrs[cid]),
                )
            )
        return BackendResult.success(
            ClustersResponse(
                clusters=entries,
                attribute_names=["asset_class", "currency_code", "desk", "product_code"],
            )
        )

    def clusters_df(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        res = self.clusters(cluster_id=cluster_id)
        if not res.ok or res.data is None:
            return BackendResult.failure(
                error=res.error or "",
                status_code=res.status_code,
            )
        rows = [
            {"cluster_id": c.cluster_id, "n_trades": c.n_trades, **c.attributes}
            for c in res.data.clusters
        ]
        return BackendResult.success(pd.DataFrame(rows))

    # ─────────────────────────────────────────────────────────────
    # Portfolio + cluster timeseries
    # ─────────────────────────────────────────────────────────────

    def portfolio_df(self, split: str) -> BackendResult[pd.DataFrame]:
        ts = self._cluster_timeseries(split)
        if ts.empty:
            return BackendResult.success(ts)
        agg = (
            ts.groupby(["scenario_idx", "scenario_label"], as_index=False)
            .agg(
                predicted=("predicted", "sum"),
                actual=("actual", "sum"),
            )
        )
        agg["error"] = agg["predicted"] - agg["actual"]
        agg["abs_error"] = agg["error"].abs()
        agg["squared_error"] = agg["error"] ** 2
        return BackendResult.success(agg)

    def cluster_timeseries_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        df = self._cluster_timeseries(split)
        if cluster_id:
            df = df[df["cluster_id"] == cluster_id]
        return BackendResult.success(df.reset_index(drop=True))

    # ─────────────────────────────────────────────────────────────
    # Graph stats + trade-graph (Phase E.3)
    # ─────────────────────────────────────────────────────────────

    def graph_stats_df(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        """Per-cluster graph topology summary.

        Topology numbers are derived from the same synthetic graphs
        :meth:`_trade_graph_payload` emits, so the Trade-Graph tab's
        stats cards and the secondary charts always agree with the
        rendered network.
        """
        rows: List[Dict[str, Any]] = []
        for cid in self._cluster_ids:
            payload = self._trade_graph_payload(cid)
            n = payload.stats.n_nodes
            rows.append(
                {
                    "cluster_id":   cid,
                    "n_nodes":      int(n),
                    "n_edges":      int(payload.stats.n_edges),
                    "density":      float(payload.stats.density),
                    "mean_weight":  float(payload.stats.mean_weight),
                }
            )
        df = pd.DataFrame(rows)
        if cluster_id:
            df = df[df["cluster_id"] == cluster_id].reset_index(drop=True)
        return BackendResult.success(df)

    def trade_graph(
        self, *, cluster_id: str,
    ) -> BackendResult[TradeGraphResponse]:
        if cluster_id not in self._cluster_ids:
            return BackendResult.failure(
                error=f"unknown cluster '{cluster_id}'",
                status_code=404,
            )
        return BackendResult.success(self._trade_graph_payload(cluster_id))

    # ─────────────────────────────────────────────────────────────
    # Trades — per-trade metrics (Phase E.4)
    # ─────────────────────────────────────────────────────────────

    def trades_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        """Synthetic per-trade metrics.

        Trade ids match those emitted by :meth:`_trade_graph_payload`
        (``{cluster_id}_T###`` for targets, ``{cluster_id}_E###`` for
        elementary trades) so the Cluster Deep-Dive page can merge
        trade-type colour on top of these rows without surprises.
        """
        cluster_ids = (
            [cluster_id]
            if cluster_id
            else list(self._cluster_ids)
        )
        frames: List[pd.DataFrame] = []
        for cid in cluster_ids:
            if cid not in self._cluster_ids:
                continue
            frames.append(_trades_df_cached(
                mock_id=id(self),
                cluster_id=cid,
                split=split,
                seed=self._seed,
            ))
        if not frames:
            return BackendResult.success(pd.DataFrame())
        return BackendResult.success(pd.concat(frames, ignore_index=True))

    # ─────────────────────────────────────────────────────────────
    # Training curves — per-cluster per-epoch (Phase E.4 Row 2)
    # ─────────────────────────────────────────────────────────────

    def training_curves_df(
        self, *, cluster_id: str,
    ) -> BackendResult[pd.DataFrame]:
        """Synthetic per-epoch training curves for one cluster.

        Produces a small deterministic curve set — ``train_loss``,
        ``val_loss``, ``mae``, ``val_mae`` — seeded off the cluster
        id so every refetch for the same cluster returns the same
        numbers.  Available metric names are stashed on
        ``df.attrs["metrics"]`` exactly like the real backend so the
        chip-row callback needs no mock-specific branch.
        """
        if cluster_id not in self._cluster_ids:
            return BackendResult.failure(
                error=f"training curves missing for '{cluster_id}'",
                status_code=404,
            )
        df = _training_curves_cached(
            mock_id=id(self),
            cluster_id=cluster_id,
            seed=self._seed,
        )
        return BackendResult.success(df)

    # ─────────────────────────────────────────────────────────────
    # Internal — seeded synthetic data generators
    # ─────────────────────────────────────────────────────────────

    def _trade_graph_payload(self, cluster_id: str) -> TradeGraphResponse:
        """Build a deterministic trade-graph payload for one cluster.

        Uses the cluster id as the RNG seed so every call for the same
        cluster returns identical nodes / edges.  Topology numbers
        (n_nodes, n_edges, density, mean_weight) are derived from the
        result — the graph-stats endpoint builds its mock off the same
        helper, so the two views cannot drift.
        """
        return _trade_graph_cached(
            mock_id=id(self),
            cluster_id=cluster_id,
            seed=self._seed,
        )

    def _cluster_timeseries(self, split: str) -> pd.DataFrame:
        """Deterministic per-cluster per-scenario synthetic frame.

        Memoised per ``(mock_id, split)`` so repeated calls inside one
        callback render don't regenerate (and re-randomise) the
        underlying numbers.  The ``mock_id`` step lets multiple
        MockRadeBackend instances coexist in the same Python process
        (e.g. tests) without bleeding state between them.
        """
        return _cluster_timeseries_cached(
            mock_id=id(self),
            split=split,
            seed=self._seed,
            cluster_ids=tuple(self._cluster_ids),
            cluster_weights=tuple(
                (cid, self._cluster_weights[cid]) for cid in self._cluster_ids
            ),
        )


# ─────────────────────────────────────────────────────────────────────
# Module-level LRU cache for the synthetic timeseries.  Keyed on the
# MockRadeBackend instance id + split so data stays stable within a
# Dash session but regenerates fresh when the server restarts.
# ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _cluster_timeseries_cached(
    *,
    mock_id:          int,
    split:            str,
    seed:             int,
    cluster_ids:      tuple[str, ...],
    cluster_weights:  tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    del mock_id  # cache key only
    rng = np.random.default_rng(seed + hash(split) % 10_000)
    labels = pd.date_range("2025-11-01", periods=_N_SCENARIOS, freq="D").strftime(
        "%Y-%m-%d"
    )
    noise_scale = {"train": 0.008, "val": 0.012, "test": 0.018}.get(split, 0.012)

    weights = dict(cluster_weights)
    frames: List[pd.DataFrame] = []
    for cid in cluster_ids:
        w = float(weights[cid])
        # Each cluster follows its own random walk scaled by weight;
        # the sum across clusters reconstructs a sensible portfolio
        # trend.
        actual_inc = rng.normal(0.003 * w, 0.004 * max(w, 0.3), _N_SCENARIOS)
        actual = np.cumsum(actual_inc) + 0.08 * w
        pred = actual + rng.normal(0, noise_scale * max(w, 0.3), _N_SCENARIOS)
        error = pred - actual
        frames.append(
            pd.DataFrame(
                {
                    "cluster_id":     cid,
                    "scenario_idx":   list(range(_N_SCENARIOS)),
                    "scenario_label": labels,
                    "predicted":      pred,
                    "actual":         actual,
                    "error":          error,
                    "abs_error":      np.abs(error),
                    "squared_error":  error ** 2,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# Trade-graph synthesis — seeded per cluster so nodes / edges never
# change for the same (mock_id, cluster_id) pair within one preview run.
# ─────────────────────────────────────────────────────────────────────

# Node budget per cluster.  Random within [lo, hi]; small enough to
# keep Cytoscape renders snappy, large enough to look like a real
# trade graph.
_TRADE_NODE_MIN = 40
_TRADE_NODE_MAX = 90
# Roughly 1/5 of nodes are "target" trades; rest are elementary
# building blocks.
_TARGET_FRACTION = 0.2


@lru_cache(maxsize=256)
def _trade_graph_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    seed:       int,
) -> TradeGraphResponse:
    del mock_id   # cache key only
    # Seed per-cluster so different clusters look different but the
    # same cluster is stable across re-fetches.
    rng = np.random.default_rng(seed + (abs(hash(cluster_id)) % 10_000))

    n = int(rng.integers(_TRADE_NODE_MIN, _TRADE_NODE_MAX + 1))
    n_target = max(1, int(round(n * _TARGET_FRACTION)))
    n_elem = n - n_target

    target_ids = [f"{cluster_id}_T{i:03d}" for i in range(n_target)]
    elem_ids = [f"{cluster_id}_E{i:03d}" for i in range(n_elem)]

    nodes: List[TradeGraphNode] = []
    for tid in target_ids:
        nodes.append(
            TradeGraphNode(trade_id=tid, cluster_id=cluster_id, trade_type="target")
        )
    for tid in elem_ids:
        nodes.append(
            TradeGraphNode(
                trade_id=tid, cluster_id=cluster_id, trade_type="elementary",
            )
        )

    # Edge budget ≈ 2 × n (moderately sparse, enough to look
    # connected without overwhelming the canvas).
    n_edges_target = max(n, 2 * n)
    edges: List[TradeGraphEdge] = []
    seen: set[tuple[int, int]] = set()
    tries = 0
    while len(edges) < n_edges_target and tries < n_edges_target * 4:
        tries += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        weight = float(rng.uniform(0.05, 1.0))
        edges.append(
            TradeGraphEdge(
                source=nodes[i].trade_id,
                target=nodes[j].trade_id,
                weight=weight,
            )
        )

    # Density uses the sparse-values count — consistent with the real
    # pipeline's ``_save_graph_stats_parquet`` formula.
    density = float(len(edges) / (n * n)) if n > 0 else 0.0
    mean_w = float(np.mean([e.weight for e in edges])) if edges else 0.0

    return TradeGraphResponse(
        cluster_id=cluster_id,
        n_target_trades=n_target,
        n_elementary_trades=n_elem,
        stats=TradeGraphStats(
            n_nodes=n,
            n_edges=len(edges),
            density=density,
            mean_weight=mean_w,
        ),
        nodes=nodes,
        edges=edges,
        warnings=None,
    )


# ─────────────────────────────────────────────────────────────────────
# Per-trade metrics synthesis — one row per trade, seeded off the trade
# id so the numbers are stable across re-fetches within one preview run.
# Uses the trade-graph payload as the authoritative trade list so the
# two views stay aligned.
# ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def _trades_df_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    split:      str,
    seed:       int,
) -> pd.DataFrame:
    del mock_id   # cache key only
    graph = _trade_graph_cached(mock_id=0, cluster_id=cluster_id, seed=seed)

    # Noise scale mirrors ``_cluster_timeseries_cached`` so the two
    # pages agree on "test is harder than val, val is harder than
    # train".
    noise_scale = {"train": 0.008, "val": 0.012, "test": 0.018}.get(split, 0.012)
    rng = np.random.default_rng(seed + (abs(hash((cluster_id, split))) % 10_000))

    rows: List[Dict[str, Any]] = []
    for node in graph.nodes:
        # Target trades carry more residual spread than elementary ones
        # — matches the real pipeline's behaviour where targets absorb
        # the model's final prediction error.
        type_scale = 1.8 if node.trade_type == "target" else 1.0
        residuals = rng.normal(
            loc=0.0,
            scale=noise_scale * type_scale,
            size=_N_SCENARIOS,
        )
        # Add a slight systematic bias so mean_residual is non-zero
        # and the violin / scatter look lifelike.
        bias = float(rng.normal(0.0, noise_scale * 0.4 * type_scale))
        residuals += bias
        abs_err = np.abs(residuals)
        sq_err = residuals ** 2
        rows.append(
            {
                "cluster_id":     cluster_id,
                "trade_id":       node.trade_id,
                "split":          split,
                "mae":            float(abs_err.mean()),
                "mse":            float(sq_err.mean()),
                "rmse":           float(np.sqrt(sq_err.mean())),
                "max_ae":         float(abs_err.max()),
                "p95_ae":         float(np.quantile(abs_err, 0.95)),
                "p99_ae":         float(np.quantile(abs_err, 0.99)),
                "mean_residual":  float(residuals.mean()),
                "std_residual":   float(residuals.std()),
                "n_scenarios":    int(_N_SCENARIOS),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Training-curves synthesis — deterministic per-cluster multi-metric
# series.  Mirrors the real pipeline's ``training_curves.parquet`` shape
# closely enough that the UI can't tell the two apart.
# ─────────────────────────────────────────────────────────────────────

_N_EPOCHS = 40


@lru_cache(maxsize=128)
def _training_curves_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    seed:       int,
) -> pd.DataFrame:
    del mock_id   # cache key only
    rng = np.random.default_rng(seed + (abs(hash(("curves", cluster_id))) % 10_000))
    epochs = np.arange(_N_EPOCHS, dtype=np.int32)

    # Per-cluster starting MAE & decay speed so clusters look different.
    start = float(rng.uniform(0.08, 0.20))
    decay = float(rng.uniform(0.08, 0.18))
    noise = rng.normal(0.0, 0.003, size=_N_EPOCHS)
    train_loss = start * np.exp(-decay * epochs) + np.abs(noise)
    # Val lags train slightly and carries a bit more noise — classic
    # learning-curve silhouette.
    val_loss = (
        start * 1.08 * np.exp(-decay * 0.95 * epochs)
        + np.abs(rng.normal(0.0, 0.006, size=_N_EPOCHS))
    )
    # Secondary metrics (mae) track loss but on a different scale so
    # the chart actually needs the metric-picker to show them meaning-
    # fully.
    mae = train_loss * 0.85 + np.abs(rng.normal(0.0, 0.002, size=_N_EPOCHS))
    val_mae = val_loss * 0.88 + np.abs(rng.normal(0.0, 0.004, size=_N_EPOCHS))

    df = pd.DataFrame(
        {
            "epoch":      epochs,
            "train_loss": train_loss.astype(np.float32),
            "val_loss":   val_loss.astype(np.float32),
            "mae":        mae.astype(np.float32),
            "val_mae":    val_mae.astype(np.float32),
        }
    )
    df.attrs["metrics"] = ["val_loss", "mae", "val_mae"]
    df.attrs["cluster_id"] = cluster_id
    return df


__all__ = ["MockRadeBackend"]
