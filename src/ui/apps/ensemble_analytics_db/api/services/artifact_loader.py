"""
Artifact loading service for the PRISM API.

All evaluation artifacts *and* registry metadata are loaded eagerly at
startup via :meth:`ArtifactCache.load_all`.  After that, every ``get_*``
method is a pure dict lookup with zero disk I/O.

Storage strategy:

* **JSON files** → Python dicts / lists  (already the natural format).
* **NPZ files** → numpy float32 arrays   (compact; 4 bytes per value
  instead of 28 bytes for a Python float).
* **joblib files** → Python objects (graph adjacency, market data).
* **Conversion to lists** happens per-request inside ``get_*`` methods
  only for the data being returned (never at startup).
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PRIMARY_SPLIT = "test"


@contextmanager
def _timed(label: str) -> Generator[None, None, None]:
    """Log wall-clock time for a block under ``[cache] <label>``."""
    t0 = time.perf_counter()
    yield
    logger.info("[cache] %s: %.2fs", label, time.perf_counter() - t0)


class ArtifactCache:
    """Central in-memory store for *all* dashboard data.

    Holds evaluation artifacts, registry metadata, per-cluster graph /
    market data, member configs, and convergence images.  Call
    :meth:`load_all` once at startup.  After that, every public accessor
    is a dict lookup — no disk reads, no lazy branches.
    """

    def __init__(
        self,
        artifacts_dir: Path,
        registry_dir: Path,
        version: str = "latest",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._registry_dir = Path(registry_dir)
        self._version_hint = version
        # Resolved after _load_registry_metadata (may differ from version if "latest")
        self._eval_dir: Optional[Path] = None

        # ── Registry / config metadata ───────────────────────────
        self.ensemble_version: str = ""
        self.member_versions: Dict[str, str] = {}
        self.cluster_mapping: Dict[str, List[str]] = {}
        self.member_configs: Dict[str, Dict[str, Any]] = {}

        # ── Eval metadata (Python dicts from JSON) ───────────────
        self.manifest: Dict[str, Any] = {}
        self.cluster_attributes: Dict[str, Dict[str, Any]] = {}
        self.trade_cluster_map: Dict[str, str] = {}
        self.graph_stats: Dict[str, Dict[str, Any]] = {}
        self.ensemble_metrics: Dict[str, Dict[str, float]] = {}
        self.member_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.trade_catalogue: List[Dict[str, Any]] = []

        # ── Per-cluster registry data (loaded eagerly) ───────────
        # {cluster_id: {"graph_results": dict, "encoder_results": dict, "trade_universe": dict}}
        self._graph_data: Dict[str, Dict[str, Any]] = {}
        # {cluster_id: {asset_name: {rf_name: np.ndarray}}}
        self._market_data: Dict[str, Dict[str, Any]] = {}
        # {cluster_id: {"trade_universe": dict, "eval_metrics": dict, "version": str, ...}}
        self._cluster_displays: Dict[str, Dict[str, Any]] = {}
        # {cluster_id: bytes (PNG)}
        self._convergence_pngs: Dict[str, bytes] = {}
        # {cluster_id: pd.DataFrame}  — loaded lazily as bytes, converted on access
        self._elementary_pnl: Dict[str, Any] = {}

        # ── Timeseries (numpy arrays from NPZ) ──────────────────
        # {split: {"predictions": 1-D ndarray, "targets": 1-D ndarray, "n_scenarios": int}}
        self._portfolio: Dict[str, Dict[str, Any]] = {}
        # {split: {cluster_id: {"predictions": 1-D ndarray, "targets": 1-D ndarray}}}
        self._cluster_summary: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        # {split: {cluster_id: {"predictions": 2-D ndarray, "targets": 2-D ndarray}}}
        self._cluster_predictions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        # ── Pre-computed (Python dicts, built from the above) ────
        # {split: {attr_key: {group_val: {"predictions": list, "targets": list, ...}}}}
        self._group_aggregations: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        # {split: {attr_key: {"columns": list, "values": list[list]}}}
        self._group_correlations: Dict[str, Dict[str, Any]] = {}
        # {split: [{"cluster_id": str, "trade_id": str, "mae": float, ...}, ...]}
        self._trade_metrics: Dict[str, List[Dict[str, Any]]] = {}
        # {split: [{"percentile": str, "predicted": float, ...}, ...]}
        self._portfolio_percentiles: Dict[str, List[Dict[str, Any]]] = {}
        # {split: [{"rank": int, "scenario": int, ...}, ...]}
        self._worst_scenarios: Dict[str, List[Dict[str, Any]]] = {}
        # {split: {attr_key: {group_val: {"mae": float, "rmse": float, ...}}}}
        self._group_summaries: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

        # ── Governance ───────────────────────────────────────────
        self._registry_versions: List[Dict[str, Any]] = []

    # ══════════════════════════════════════════════════════════════
    # Eager startup — load everything
    # ══════════════════════════════════════════════════════════════

    @property
    def cluster_ids(self) -> List[str]:
        """Sorted cluster IDs (derived from cluster_mapping)."""
        return sorted(self.cluster_mapping.keys())

    def load_all(self) -> None:
        """Read every artifact into RAM.  Call once at startup."""
        t0 = time.perf_counter()

        with _timed("_load_registry_metadata"):
            self._load_registry_metadata()

        with _timed("_load_metadata"):
            self._load_metadata()

        splits = self.manifest.get("splits_available", ["test"])
        cids = self.cluster_ids

        for split in splits:
            with _timed(f"split={split}"):
                self._load_portfolio(split)
                self._load_cluster_summary(split)
                self._load_cluster_predictions(split, cids)
                self._load_trade_metrics(split)
                self._load_group_correlations(split)
                self._compute_portfolio_analytics(split)
                self._compute_group_summaries(split)

        with _timed("_load_all_cluster_registry_data"):
            self._load_all_cluster_registry_data()

        if not self.graph_stats:
            with _timed("_compute_graph_stats_from_cache"):
                self._compute_graph_stats_from_cache()

        with _timed("_load_registry_versions"):
            self._load_registry_versions()

        logger.info(
            "[cache] COMPLETE: %d clusters, %d trades, %d splits, "
            "~%.1fMB numpy, total=%.1fs",
            len(self.cluster_attributes),
            len(self.trade_cluster_map),
            len(splits),
            self._numpy_bytes() / (1024 * 1024),
            time.perf_counter() - t0,
        )

    # ── Registry metadata (replaces EnsembleSession.load_metadata) ──

    def _load_registry_metadata(self) -> None:
        """Load ensemble config, member versions, cluster mapping from registry."""
        from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

        registry = EnsembleRegistry(self._registry_dir)
        config, member_versions, version = registry.load(self._version_hint)

        self.ensemble_version = version
        self.member_versions = member_versions
        self.cluster_mapping = dict(config.cluster_mapping)
        self.member_configs = dict(config.member_configs)

        self._eval_dir = (
            self.artifacts_dir / "ensemble" / version / "evaluation"
        )

        ca = config.get_cluster_keys_for_router()
        if ca:
            self.cluster_attributes = ca

        logger.info(
            "[cache]   registry resolved version=%s, %d clusters",
            version, len(self.cluster_mapping),
        )

    def _load_all_cluster_registry_data(self) -> None:
        """Eagerly load per-cluster registry blobs (graph, market, display).

        Uses a thread pool because joblib/numpy I/O releases the GIL.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        member_summary: Dict[str, Any] = {}
        summary_path = self._registry_dir / self.ensemble_version / "member_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                member_summary = json.load(f)

        def _load_one(cid: str) -> tuple:
            version = self.member_versions.get(cid)
            if not version:
                return cid, None, None, None, None
            version_dir = self._registry_dir / version
            graph = self._load_cluster_graph(version_dir)
            market = self._load_cluster_market(version_dir)
            display = self._build_cluster_display(
                cid, version, version_dir, member_summary, graph,
            )
            return cid, version, graph, market, display

        n_workers = min(8, len(self.cluster_ids))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_load_one, cid): cid
                for cid in self.cluster_ids
            }
            for future in as_completed(futures):
                cid, version, graph, market, display = future.result()
                if version is None:
                    continue
                self._graph_data[cid] = graph
                self._market_data[cid] = market
                self._cluster_displays[cid] = display
                self._load_cluster_extras(cid, version)

        logger.info(
            "[cache]   loaded registry data for %d clusters (%d workers)",
            len(self.cluster_ids), n_workers,
        )

    def _load_cluster_graph(self, version_dir: Path) -> Dict[str, Any]:
        """Load graph_results, encoder_results, trade_universe for one cluster."""
        import joblib

        data: Dict[str, Any] = {}
        graph_path = version_dir / "graph_results.joblib"
        if graph_path.exists():
            data["graph_results"] = joblib.load(str(graph_path))
        encoder_path = version_dir / "encoder_results.joblib"
        if encoder_path.exists():
            data["encoder_results"] = joblib.load(str(encoder_path))

        tu_path = version_dir / "trade_universe.json"
        data["trade_universe"] = (
            self._read_json_path(tu_path) if tu_path.exists() else {}
        )
        return data

    def _load_cluster_market(self, version_dir: Path) -> Dict[str, Any]:
        """Load market shock arrays for one cluster."""
        import joblib

        assets_path = version_dir / "cluster_assets.joblib"
        if not assets_path.exists():
            return {}

        portfolio = joblib.load(str(assets_path))
        mkt: Dict[str, Any] = {}
        for asset_name, asset in portfolio.items():
            rf_shocks = getattr(asset, "risk_factor_shocks", None)
            if rf_shocks is None:
                continue
            mkt[asset_name] = {}
            for rf_name, shocks in rf_shocks.items():
                try:
                    if isinstance(shocks, dict):
                        arr = np.array(list(shocks.values()), dtype=np.float64)
                    else:
                        arr = np.asarray(shocks, dtype=np.float64)
                    mkt[asset_name][rf_name] = arr
                except (TypeError, ValueError):
                    continue
        return mkt

    def _build_cluster_display(
        self,
        cid: str,
        version: str,
        version_dir: Path,
        member_summary: Dict[str, Any],
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble display payload for one cluster."""
        gd = graph_data if graph_data is not None else self._graph_data.get(cid, {})
        display: Dict[str, Any] = {
            "version": version,
            "version_dir": str(version_dir),
            "trade_universe": gd.get("trade_universe", {}),
            "eval_metrics": {},
        }
        if cid in member_summary:
            display["eval_metrics"]["summary"] = member_summary[cid]

        dc_path = version_dir / "data_config.json"
        if dc_path.exists():
            display["eval_metrics"]["data_config"] = self._read_json_path(dc_path)
        return display

    def _load_cluster_extras(self, cid: str, version: str) -> None:
        """Load convergence PNG bytes and elementary PnL path for one cluster."""
        plot_path = self.artifacts_dir / "training" / version / "training_plots.png"
        if plot_path.exists():
            self._convergence_pngs[cid] = plot_path.read_bytes()

        elem_path = self._registry_dir / version / "elementary_pnl.parquet"
        if elem_path.exists():
            self._elementary_pnl[cid] = str(elem_path)

    def _load_registry_versions(self) -> None:
        """Load version listing for governance comparison dropdown."""
        from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

        try:
            registry = EnsembleRegistry(self._registry_dir)
            self._registry_versions = registry.list_versions()
        except Exception:
            logger.warning("Could not load registry versions for governance")
            self._registry_versions = []

    @staticmethod
    def _to_numpy(v):
        """Convert a value to a numpy array, handling torch tensors."""
        if v is None:
            return None
        try:
            import torch
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                if t.is_sparse:
                    t = t.to_dense()
                return t.numpy()
        except ImportError:
            pass
        return np.asarray(v)

    def _compute_graph_stats_from_cache(self) -> None:
        """Derive graph_stats from already-loaded _graph_data (fallback).

        Mirrors the logic in ``_save_graph_stats`` from the eval pipeline,
        using ``sparse_values`` / ``sparse_shape`` from the graph_results dict.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        default = {"n_nodes": 0, "n_edges": 0, "density": 0.0, "mean_weight": 0.0}
        for cid, gd in self._graph_data.items():
            gr = gd.get("graph_results")
            if not isinstance(gr, dict):
                stats[cid] = default.copy()
                continue
            try:
                values = self._to_numpy(gr.get("sparse_values"))
                shape_raw = gr.get("sparse_shape", [0, 0])
                shape = self._to_numpy(shape_raw) if shape_raw is not None else np.array([0, 0])
                n_nodes = int(shape[0]) if shape[0] > 0 else 0
                nnz = len(values) if values is not None else 0
                if values is not None and values.ndim > 1:
                    nnz = values.shape[-1] if values.ndim == 2 else values.size
                density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
                mean_w = float(np.mean(values)) if values is not None and nnz > 0 else 0
                stats[cid] = {
                    "n_nodes": int(n_nodes),
                    "n_edges": int(nnz),
                    "density": round(density, 6),
                    "mean_weight": round(mean_w, 4),
                }
            except Exception:
                stats[cid] = default.copy()
        self.graph_stats = stats
        logger.info("[cache]   computed graph_stats for %d clusters from cache", len(stats))

    # ── Eval metadata (called by load_all) ───────────────────────

    def _load_metadata(self) -> None:
        self.manifest = self._read_json("manifest.json") or {}

        eval_attrs = self._read_json("cluster_attributes.json") or {}
        if eval_attrs:
            self.cluster_attributes = eval_attrs

        self.trade_cluster_map = self._read_json("trade_cluster_map.json") or {}
        self.graph_stats = self._read_json("graph_stats.json") or {}

        splits = self.manifest.get("splits_available", ["test"])
        for split in splits:
            suffix = "" if split == _PRIMARY_SPLIT else f"_{split}"
            self.ensemble_metrics[split] = (
                self._read_json(f"ensemble_metrics{suffix}.json") or {}
            )
            self.member_metrics[split] = (
                self._read_json(f"per_member_metrics{suffix}.json") or {}
            )

        self._build_trade_catalogue()

    def _load_portfolio(self, split: str) -> None:
        npz = self._read_npz(f"portfolio_summary/{split}.npz")
        if npz is not None:
            self._portfolio[split] = {
                "predictions": npz["predictions"],
                "targets": npz["targets"],
                "n_scenarios": len(npz["predictions"]),
            }

    def _load_cluster_summary(self, split: str) -> None:
        npz = self._read_npz(f"cluster_summary/{split}.npz")
        if npz is None:
            self._cluster_summary[split] = {}
            return

        result: Dict[str, Dict[str, Any]] = {}
        for key, arr in npz.items():
            cid, suffix = key.rsplit("_", 1)
            entry = result.setdefault(cid, {})
            if suffix == "pred":
                entry["predictions"] = arr
            else:
                entry["targets"] = arr
        self._cluster_summary[split] = result
        self._build_group_aggregations(split, result)

    def _load_cluster_predictions(
        self, split: str, cluster_ids: List[str],
    ) -> None:
        split_store: Dict[str, Dict[str, np.ndarray]] = {}
        for cid in cluster_ids:
            npz_path = (
                self._eval_dir / "members" / cid / "predictions" / f"{split}.npz"
            )
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=False)
            split_store[cid] = {
                "predictions": data["predictions"],
                "targets": data["targets"],
            }
        self._cluster_predictions[split] = split_store

    def _load_trade_metrics(self, split: str) -> None:
        self._trade_metrics[split] = (
            self._read_json(f"trade_metrics/{split}.json") or []
        )

    def _load_group_correlations(self, split: str) -> None:
        self._group_correlations[split] = (
            self._read_json(f"group_correlations/{split}.json") or {}
        )

    # ══════════════════════════════════════════════════════════════
    # Public accessors — dict lookups only
    # ══════════════════════════════════════════════════════════════

    def get_portfolio(self, split: str) -> Optional[Dict[str, Any]]:
        """Return ``{"predictions": list, "targets": list, "n_scenarios": int}``."""
        data = self._portfolio.get(split)
        if data is None:
            return None
        return {
            "predictions": data["predictions"].tolist(),
            "targets": data["targets"].tolist(),
            "n_scenarios": data["n_scenarios"],
        }

    def get_cluster_summary(self, split: str) -> Dict[str, Dict[str, Any]]:
        """Return ``{cluster_id: {"predictions": list, "targets": list}}``."""
        raw = self._cluster_summary.get(split, {})
        return {
            cid: {
                "predictions": entry["predictions"].tolist(),
                "targets": entry["targets"].tolist(),
            }
            for cid, entry in raw.items()
        }

    def get_cluster_predictions(
        self, cluster_id: str, split: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return raw numpy ``{"predictions": 2-D, "targets": 2-D}``."""
        return self._cluster_predictions.get(split, {}).get(cluster_id)

    def get_group_aggregations(
        self, split: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return ``{attr_key: {group_value: {predictions, targets, ...}}}``."""
        return self._group_aggregations.get(split, {})

    def get_group_correlations(self, split: str) -> Dict[str, Any]:
        return self._group_correlations.get(split, {})

    def get_trade_metrics(self, split: str) -> List[Dict[str, Any]]:
        return self._trade_metrics.get(split, [])

    def get_portfolio_percentiles(self, split: str) -> List[Dict[str, Any]]:
        return self._portfolio_percentiles.get(split, [])

    def get_worst_scenarios(self, split: str) -> List[Dict[str, Any]]:
        return self._worst_scenarios.get(split, [])

    def get_group_summaries(
        self, split: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return ``{attr_key: {group_val: {mae, rmse, n_clusters, n_trades}}}``."""
        return self._group_summaries.get(split, {})

    # ── Registry data accessors ──────────────────────────────────

    def get_graph_data(self, cluster_id: str) -> Dict[str, Any]:
        """Return graph adjacency, encoder, and trade universe for a cluster."""
        return self._graph_data.get(cluster_id, {})

    def get_market_data(self, cluster_id: str) -> Dict[str, Any]:
        """Return ``{asset_name: {rf_name: np.ndarray}}`` for a cluster."""
        return self._market_data.get(cluster_id, {})

    def get_cluster_display(self, cluster_id: str) -> Dict[str, Any]:
        """Return cluster display state (eval_metrics, trade_universe, version)."""
        return self._cluster_displays.get(cluster_id, {})

    def get_convergence_png(self, cluster_id: str) -> Optional[bytes]:
        """Return raw PNG bytes for a cluster's training convergence plot."""
        return self._convergence_pngs.get(cluster_id)

    def get_elementary_pnl_path(self, cluster_id: str) -> Optional[str]:
        """Return filesystem path to the elementary_pnl parquet file."""
        return self._elementary_pnl.get(cluster_id)

    def get_registry_versions(self) -> List[Dict[str, Any]]:
        """Return version listing for governance."""
        return self._registry_versions

    def get_ensemble_metrics_for_version(
        self, compare_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Load ensemble metrics for an arbitrary version (governance comparison)."""
        compare_dir = (
            self.artifacts_dir / "ensemble" / compare_version / "evaluation"
        )
        compare_path = compare_dir / "ensemble_metrics.json"
        if not compare_path.exists():
            compare_path = compare_dir / "ensemble_metrics_test.json"
        if not compare_path.exists():
            return None
        with open(compare_path) as f:
            return json.load(f)

    # ══════════════════════════════════════════════════════════════
    # Pre-computation helpers
    # ══════════════════════════════════════════════════════════════

    def _build_trade_catalogue(self) -> None:
        rows: List[Dict[str, Any]] = []
        for trade_id, cluster_id in self.trade_cluster_map.items():
            row: Dict[str, Any] = {
                "trade_id": trade_id,
                "cluster_id": cluster_id,
            }
            row.update(self.cluster_attributes.get(cluster_id, {}))
            rows.append(row)
        self.trade_catalogue = rows

    def _build_group_aggregations(
        self,
        split: str,
        cluster_summary: Dict[str, Dict[str, Any]],
    ) -> None:
        """Sum cluster timeseries by each attribute dimension.

        Uses numpy for accumulation, converts to Python lists once at the
        end so ``get_group_aggregations`` returns JSON-ready data.
        """
        attr_keys = {
            k
            for ca in self.cluster_attributes.values()
            for k in ca
            if k != "n_trades"
        }
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for attr_key in sorted(attr_keys):
            groups: Dict[str, Dict[str, Any]] = {}
            for cid, data in cluster_summary.items():
                val = self.cluster_attributes.get(cid, {}).get(attr_key)
                if val is None:
                    continue
                val_str = str(val)
                n_trades = self.cluster_attributes.get(cid, {}).get("n_trades", 0)
                if val_str not in groups:
                    groups[val_str] = {
                        "predictions": data["predictions"].copy(),
                        "targets": data["targets"].copy(),
                        "n_clusters": 1,
                        "n_trades": n_trades,
                    }
                else:
                    grp = groups[val_str]
                    grp["predictions"] = grp["predictions"] + data["predictions"]
                    grp["targets"] = grp["targets"] + data["targets"]
                    grp["n_clusters"] += 1
                    grp["n_trades"] += n_trades

            for group_data in groups.values():
                group_data["predictions"] = group_data["predictions"].tolist()
                group_data["targets"] = group_data["targets"].tolist()

            result[attr_key] = groups

        self._group_aggregations[split] = result

    def _compute_portfolio_analytics(self, split: str) -> None:
        """Pre-compute distribution table and worst scenarios for one split."""
        data = self._portfolio.get(split)
        if data is None:
            return

        preds = data["predictions"]
        targets = data["targets"]
        residuals = preds - targets
        abs_errors = np.abs(residuals)

        percentiles = [1, 5, 25, 50, 75, 95, 99]
        rows: List[Dict[str, Any]] = []
        for p in percentiles:
            rows.append({
                "metric": f"P{p}",
                "predicted": float(np.percentile(preds, p)),
                "target": float(np.percentile(targets, p)),
                "diff": float(np.percentile(preds, p) - np.percentile(targets, p)),
                "abs_error": float(np.percentile(abs_errors, p)),
            })
        rows.append({
            "metric": "Mean",
            "predicted": float(preds.mean()),
            "target": float(targets.mean()),
            "diff": float(residuals.mean()),
            "abs_error": float(abs_errors.mean()),
        })
        rows.append({
            "metric": "Std",
            "predicted": float(preds.std()),
            "target": float(targets.std()),
            "diff": float(residuals.std()),
            "abs_error": float(abs_errors.std()),
        })
        rows.append({
            "metric": "MAE",
            "predicted": None,
            "target": None,
            "diff": None,
            "abs_error": float(abs_errors.mean()),
        })
        rows.append({
            "metric": "RMSE",
            "predicted": None,
            "target": None,
            "diff": None,
            "abs_error": float(np.sqrt(np.mean(residuals ** 2))),
        })
        rows.append({
            "metric": "Max AE",
            "predicted": None,
            "target": None,
            "diff": None,
            "abs_error": float(abs_errors.max()),
        })
        self._portfolio_percentiles[split] = rows

        worst_idx = np.argsort(abs_errors)[::-1][:20]
        self._worst_scenarios[split] = [
            {
                "rank": rank + 1,
                "scenario": int(idx),
                "target": float(targets[idx]),
                "prediction": float(preds[idx]),
                "abs_error": float(abs_errors[idx]),
            }
            for rank, idx in enumerate(worst_idx)
        ]

    def _compute_group_summaries(self, split: str) -> None:
        """Pre-compute scalar metrics (MAE, RMSE) per group per attribute."""
        aggs = self._group_aggregations.get(split, {})
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for attr_key, groups in aggs.items():
            attr_summaries: Dict[str, Dict[str, Any]] = {}
            for group_val, gdata in groups.items():
                p = np.array(gdata["predictions"])
                t = np.array(gdata["targets"])
                residual = p - t
                attr_summaries[group_val] = {
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual ** 2))),
                    "n_clusters": gdata["n_clusters"],
                    "n_trades": gdata["n_trades"],
                }
            result[attr_key] = attr_summaries

        self._group_summaries[split] = result

    # ══════════════════════════════════════════════════════════════
    # Cache management
    # ══════════════════════════════════════════════════════════════

    def reload(self) -> None:
        """Full reload from disk (e.g. after a new eval run)."""
        self._portfolio.clear()
        self._cluster_summary.clear()
        self._cluster_predictions.clear()
        self._group_aggregations.clear()
        self._group_correlations.clear()
        self._trade_metrics.clear()
        self._portfolio_percentiles.clear()
        self._worst_scenarios.clear()
        self._group_summaries.clear()
        self._graph_data.clear()
        self._market_data.clear()
        self._cluster_displays.clear()
        self._convergence_pngs.clear()
        self._elementary_pnl.clear()
        self.load_all()

    @property
    def cached_splits(self) -> List[str]:
        return sorted(self.manifest.get("splits_available", []))

    def _numpy_bytes(self) -> int:
        total = 0
        for split_data in self._portfolio.values():
            for v in split_data.values():
                if isinstance(v, np.ndarray):
                    total += v.nbytes
        for split_data in self._cluster_summary.values():
            for entry in split_data.values():
                for v in entry.values():
                    if isinstance(v, np.ndarray):
                        total += v.nbytes
        for split_data in self._cluster_predictions.values():
            for entry in split_data.values():
                for v in entry.values():
                    if isinstance(v, np.ndarray):
                        total += v.nbytes
        return total

    # ══════════════════════════════════════════════════════════════
    # Disk I/O helpers (used only during load_all / reload)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _read_json_path(path: Path) -> Any:
        """Read JSON from an absolute path."""
        with open(path) as f:
            return json.load(f)

    def _read_json(self, rel_path: str) -> Any:
        path = self._eval_dir / rel_path
        if not path.exists():
            logger.debug("JSON not found: %s", path)
            return None
        return self._read_json_path(path)

    def _read_npz(self, rel_path: str) -> Optional[Dict[str, np.ndarray]]:
        path = self._eval_dir / rel_path
        if not path.exists():
            logger.debug("NPZ not found: %s", path)
            return None
        return dict(np.load(path, allow_pickle=False))
