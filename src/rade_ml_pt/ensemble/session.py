"""
Ensemble session: three-phase lifecycle for UI and programmatic use.

Separates **display state** (metrics, plots, predictions — fast file reads)
from **inference state** (nn.Modules, graph builders, encoders — heavy
deserialization, loaded on demand).

Phases
------
1. **Metadata** (instant) — ensemble config, member versions, trade-cluster
   map, member summary.  Enough for landing page rendering.
2. **Display artifacts** (fast) — per-cluster eval metrics, plot file paths,
   prediction arrays.  Enough for full analytics drill-down.  No model
   loading; just reading JSON/PNG/NPZ from the artifacts directory.
3. **Inference state** (on demand) — per-cluster nn.Module + inference
   context (graph_builder, encoder, scalers).  Loaded lazily when the user
   first runs inference, with optional parallel loading across clusters.

Design
------
- The session never mutates registry or artifacts on disk.
- Inference runs on **copies** of baseline data; cached state stays read-only.
- Display artifacts are loaded eagerly (Phase 2) because they're small files.
- Inference state is loaded lazily (Phase 3) because model.pt + pickle files
  are large; parallel loading with ``ThreadPoolExecutor`` keeps wall-clock
  time proportional to the slowest cluster, not the sum.

Usage
-----
::

    session = EnsembleSession(registry_dir, artifacts_dir)
    session.load_metadata("production")          # Phase 1: instant
    session.load_display_artifacts()             # Phase 2: fast file reads
    session.load_inference_state()               # Phase 3: parallel model load
    result = session.run_inference(mode, ...)    # Uses cached state
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Per-cluster display state (Phase 2) — lightweight file reads
# ------------------------------------------------------------------

@dataclass
class ClusterDisplayState:
    """Pre-saved evaluation artifacts for one cluster (no model objects)."""

    cluster_id: str
    version: str
    version_dir: str

    # {split: {metric_name: value}} — per-split eval metrics
    eval_metrics: Dict[str, Any] = field(default_factory=dict)

    # {"split/plot_name": "/abs/path.png"}
    plot_paths: Dict[str, str] = field(default_factory=dict)

    # from trade_universe.json
    trade_universe: Dict[str, Any] = field(default_factory=dict)

    # from target_attributes.json — per-trade attribute arrays
    target_attributes: Dict[str, Any] = field(default_factory=dict)

    # predictions arrays (loaded from .npz on demand)
    predictions: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)


@dataclass
class EnsembleDisplayState:
    """Ensemble-wide (portfolio-level) evaluation artifacts."""

    # {split: {mae, rmse, ...}}
    ensemble_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # {split: rollup dict}
    member_rollup: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # {split: {cluster_id: metrics}}
    per_member_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # manifest.json contents (trade_ids, cluster_ids, cluster_trade_indices, splits_available)
    manifest: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Global prediction store — unified arrays for cross-cluster slicing
# ------------------------------------------------------------------

@dataclass
class GlobalPredictionStore:
    """Unified prediction/target arrays aligned by global trade order.

    Built lazily from per-member ``.npz`` files.  All arrays share the
    same ``[n_scenarios, n_total_targets]`` shape so callers can slice
    by arbitrary trade subsets without per-cluster bookkeeping.
    """

    predictions: np.ndarray       # [n_scenarios, n_total_targets]
    targets: np.ndarray           # [n_scenarios, n_total_targets]
    trade_ids: List[str]          # length = n_total_targets
    cluster_ids: List[str]        # per-target cluster membership
    split: str                    # "train", "val", or "test"


# ------------------------------------------------------------------
# Per-cluster inference state (Phase 3) — heavy model objects
# ------------------------------------------------------------------

@dataclass
class ClusterInferenceState:
    """Loaded model + inference context for one cluster."""

    cluster_id: str
    model: Any = None                        # nn.Module (eval mode)
    inference_context: Dict[str, Any] = field(default_factory=dict)
    data_config: Any = None                  # HybridGnnRnnDataConfig (or dict)
    baseline_pnl: Optional[np.ndarray] = None


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------

class EnsembleSession:
    """
    Manages the full lifecycle of an ensemble for the UI.

    Parameters
    ----------
    registry_dir : str or Path
        Root directory for model and ensemble registries.
    artifacts_dir : str or Path or None
        Root directory for evaluation artifacts (plots, metrics, predictions).
        If None, display artifact loading is skipped.
    max_workers : int
        Thread pool size for parallel model loading in Phase 3.
    """

    def __init__(
        self,
        registry_dir: Union[str, Path],
        artifacts_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 4,
    ) -> None:
        self.registry_dir = Path(registry_dir)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.max_workers = max_workers

        self._ens_registry = EnsembleRegistry(self.registry_dir)

        # Phase 1: metadata
        self._config: Optional[EnsembleConfig] = None
        self._member_versions: Optional[Dict[str, str]] = None
        self._ensemble_version: Optional[str] = None
        self._trade_cluster_map: Optional[Dict[str, str]] = None
        self._member_summary: Optional[Dict[str, Dict[str, Any]]] = None

        # Phase 2: display artifacts per cluster + ensemble-level
        self._display: Dict[str, ClusterDisplayState] = {}
        self._ensemble_display: Optional[EnsembleDisplayState] = None

        # Phase 2b: cached global prediction stores per split
        self._prediction_stores: Dict[str, GlobalPredictionStore] = {}

        # Phase 2c: cached market / graph data per cluster
        self._market_data_cache: Dict[str, Dict[str, Any]] = {}
        self._graph_data_cache: Dict[str, Dict[str, Any]] = {}

        # Phase 3: inference state per cluster
        self._inference: Dict[str, ClusterInferenceState] = {}

        # Assembled ensemble (built when all inference states are loaded)
        self._ensemble_model: Any = None  # EnsembleModel

    # ==================================================================
    # Phase 1 — Metadata (instant)
    # ==================================================================

    def load_metadata(self, version_or_tag: str = "latest") -> None:
        """
        Load ensemble metadata from the registry.

        After this call the session knows: cluster IDs, member versions,
        trade-cluster map, and per-cluster summary stats.  Enough for
        landing page rendering.
        """
        config, member_versions, version = self._ens_registry.load(version_or_tag)
        self._config = config
        self._member_versions = member_versions
        self._ensemble_version = version

        # trade_cluster_map and member_summary from the ensemble version dir
        ens_version_dir = self._ens_registry.root_dir / version

        tcm_path = ens_version_dir / "trade_cluster_map.json"
        if tcm_path.exists():
            with open(tcm_path, "r") as f:
                self._trade_cluster_map = json.load(f)

        summary_path = ens_version_dir / "member_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                self._member_summary = json.load(f)

        logger.info(
            "Session Phase 1: loaded metadata for ensemble '%s' (%d clusters)",
            version, config.n_members,
        )

    # ==================================================================
    # Phase 2 — Display artifacts (fast file reads)
    # ==================================================================

    def load_display_artifacts(self) -> None:
        """
        Load ensemble-level evaluation artifacts (fast).

        Reads only the portfolio-level JSON files (ensemble metrics,
        per-member rollups, manifest).  Per-cluster display state
        (trade universe, target attributes, plots, data config) is
        loaded lazily on first access via :meth:`load_cluster_display`.
        """
        self._require_metadata()
        self._ensemble_display = self._load_ensemble_display()

        logger.info(
            "Session Phase 2: loaded ensemble display artifacts "
            "(%d clusters available for lazy loading)",
            self._config.n_members,
        )

    def load_cluster_display(self, cluster_id: str) -> ClusterDisplayState:
        """Load display artifacts for a single cluster (for lazy/on-demand drill-down)."""
        self._require_metadata()
        if cluster_id not in self._display:
            self._display[cluster_id] = self._load_cluster_display(cluster_id)
        return self._display[cluster_id]

    def _load_cluster_display(self, cluster_id: str) -> ClusterDisplayState:
        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version

        state = ClusterDisplayState(
            cluster_id=cluster_id,
            version=version,
            version_dir=str(version_dir),
        )

        if self._member_summary and cluster_id in self._member_summary:
            state.eval_metrics["summary"] = dict(self._member_summary[cluster_id])

        universe_path = version_dir / "trade_universe.json"
        if universe_path.exists():
            with open(universe_path, "r") as f:
                state.trade_universe = json.load(f)

        ta_path = version_dir / "target_attributes.json"
        if ta_path.exists():
            with open(ta_path, "r") as f:
                state.target_attributes = json.load(f)

        if self.artifacts_dir is not None:
            for split in ("train", "val", "test"):
                plots_dir = (
                    self.artifacts_dir / "ensemble" / self._ensemble_version
                    / "evaluation" / "plots" / split
                )
                if plots_dir.exists():
                    for p in plots_dir.glob("*.png"):
                        state.plot_paths[f"{split}/{p.stem}"] = str(p)

        dc_path = version_dir / "data_config.json"
        if dc_path.exists():
            with open(dc_path, "r") as f:
                state.eval_metrics["data_config"] = json.load(f)

        return state

    def _load_ensemble_display(self) -> Optional[EnsembleDisplayState]:
        """Load ensemble-wide (portfolio-level) evaluation artifacts."""
        if self.artifacts_dir is None:
            return None

        eval_dir = (
            self.artifacts_dir / "ensemble" / self._ensemble_version / "evaluation"
        )
        if not eval_dir.exists():
            return None

        state = EnsembleDisplayState()

        # Manifest
        manifest_path = eval_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                state.manifest = json.load(f)

        # Per-split: ensemble metrics, rollup, per-member metrics
        splits = state.manifest.get("splits_available", ["test"])
        for split in splits:
            suffix = "" if split == "test" else f"_{split}"

            em_path = eval_dir / f"ensemble_metrics{suffix}.json"
            if em_path.exists():
                with open(em_path, "r") as f:
                    state.ensemble_metrics[split] = json.load(f)

            rollup_path = eval_dir / f"member_rollup{suffix}.json"
            if rollup_path.exists():
                with open(rollup_path, "r") as f:
                    state.member_rollup[split] = json.load(f)

            pm_path = eval_dir / f"per_member_metrics{suffix}.json"
            if pm_path.exists():
                with open(pm_path, "r") as f:
                    state.per_member_metrics[split] = json.load(f)

        return state

    def load_cluster_predictions(
        self, cluster_id: str, split: str = "test",
    ) -> Optional[np.ndarray]:
        """
        Load saved prediction arrays for one cluster + split.

        Only loads the .npz from disk when first requested; caches for
        subsequent calls.
        """
        display = self.load_cluster_display(cluster_id)
        cache_key = f"{split}_predictions"
        if cache_key in display.predictions:
            return display.predictions[cache_key]

        version_dir = Path(display.version_dir)

        # try artifacts_dir first (from eval pipeline)
        if self.artifacts_dir is not None:
            npz_path = (
                self.artifacts_dir / "ensemble" / self._ensemble_version
                / "members" / cluster_id / "predictions" / f"{split}.npz"
            )
            if npz_path.exists():
                data = np.load(str(npz_path))
                arr = data.get("predictions", data.get("arr_0"))
                display.predictions[cache_key] = arr
                return arr

        # fallback: from registry version_dir/datasets
        ds_path = version_dir / "datasets" / f"{split}.pt"
        if ds_path.exists():
            logger.debug("Predictions .npz not found; raw dataset at %s", ds_path)

        display.predictions[cache_key] = None
        return None

    # ==================================================================
    # Phase 3 — Inference state (on demand, parallel)
    # ==================================================================

    def load_inference_state(
        self,
        cluster_ids: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> None:
        """
        Load models + inference contexts for the specified clusters.

        Parameters
        ----------
        cluster_ids : list or None
            Clusters to load.  None = all clusters.
        parallel : bool
            If True, load clusters concurrently via ThreadPoolExecutor.
        """
        self._require_metadata()
        targets = cluster_ids or list(self._config.cluster_ids)
        targets = [cid for cid in targets if cid not in self._inference]

        if not targets:
            logger.info("Session Phase 3: all requested clusters already loaded")
            return

        logger.info(
            "Session Phase 3: loading inference state for %d clusters%s",
            len(targets), " (parallel)" if parallel else "",
        )

        if parallel and len(targets) > 1:
            self._load_parallel(targets)
        else:
            for cid in targets:
                self._inference[cid] = self._load_cluster_inference(cid)

        # Build the assembled EnsembleModel if all clusters are ready
        if set(self._config.cluster_ids) <= set(self._inference.keys()):
            self._build_ensemble_model()

        logger.info(
            "Session Phase 3: inference ready for %d / %d clusters",
            len(self._inference), self._config.n_members,
        )

    def _load_parallel(self, cluster_ids: List[str]) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._load_cluster_inference, cid): cid
                for cid in cluster_ids
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    self._inference[cid] = future.result()
                    logger.info("  loaded inference state: %s", cid)
                except Exception:
                    logger.exception("  FAILED to load inference state: %s", cid)

    def _load_cluster_inference(self, cluster_id: str) -> ClusterInferenceState:
        from src.rade_ml_pt.registry.store import ModelRegistry

        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version

        # Load model
        model_registry = ModelRegistry(self.registry_dir)
        model, _entry = model_registry.load(version)
        model.eval()

        # Load inference context (graph_builder, encoder, scalers) when present.
        # Keep this optional so non-hybrid members can still be loaded and used
        # with externally prepared member inputs.
        context: Dict[str, Any] = {}
        try:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                load_inference_context_from_dir,
            )
            context = load_inference_context_from_dir(version_dir)
        except Exception:
            logger.debug(
                "No model-specific inference context loaded for cluster '%s' "
                "(version '%s'); external/prebuilt member inputs will be required.",
                cluster_id, version,
            )

        # Data config
        data_config = None
        dc_path = version_dir / "data_config.json"
        if dc_path.exists():
            from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
            data_config = HybridGnnRnnDataConfig.from_json(dc_path)

        # Baseline PnL (optional, for new_scenarios default)
        baseline_pnl = None
        pnl_path = version_dir / "elementary_pnl.parquet"
        if pnl_path.exists():
            import pandas as pd
            baseline_pnl = pd.read_parquet(pnl_path).to_numpy().astype(np.float32)

        return ClusterInferenceState(
            cluster_id=cluster_id,
            model=model,
            inference_context=context,
            data_config=data_config,
            baseline_pnl=baseline_pnl,
        )

    def _build_ensemble_model(self) -> None:
        from src.rade_ml_pt.ensemble.model import EnsembleModel
        from src.rade_ml_pt.ensemble.router import TradeRouter
        from src.rade_ml_pt.ensemble.builder import EnsembleBuilder

        members = {cid: state.model for cid, state in self._inference.items()}
        router = TradeRouter(
            self._config.cluster_mapping,
            cluster_keys=self._config.get_cluster_keys_for_router(),
        )
        cluster_trade_indices = EnsembleBuilder._build_cluster_trade_indices(self._config)

        self._ensemble_model = EnsembleModel(
            members=members,
            router=router,
            aggregation=self._config.aggregation,
            weights=self._config.weights,
            cluster_trade_indices=cluster_trade_indices,
            n_total_targets=len(self._config.all_trade_ids),
        )
        logger.info("Assembled EnsembleModel (%d members)", len(members))

    # ==================================================================
    # Inference
    # ==================================================================

    def run_inference(
        self,
        mode: str = "new_scenarios",
        cluster_pnl_histories: Optional[Dict[str, np.ndarray]] = None,
        new_trade_attribs: Optional[Dict[str, Any]] = None,
        member_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference using cached state.

        Parameters
        ----------
        mode : str
            ``"new_scenarios"`` or ``"new_trades"``.
        cluster_pnl_histories : dict or None
            ``{cluster_id: pnl_array [n_scenarios, seq_len, n_elementary]}``.
            Required for new_scenarios; optional for new_trades (uses baseline).
        new_trade_attribs : dict or None
            For new_trades mode: ``{cluster_id: trade_attribs_dict}`` or a single
            dict that will be routed via the TradeRouter.

        Returns
        -------
        dict
            ``predictions`` (np.ndarray), ``per_member`` (dict), ``metadata`` (dict).
        """
        self._require_inference()

        if mode not in {"new_scenarios", "new_trades"}:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: 'new_scenarios', 'new_trades'."
            )

        # Model-agnostic path: caller provides fully prepared per-member inputs.
        if member_inputs:
            combined = self._ensemble_model.predict(member_inputs)
            return {
                "predictions": combined,
                "per_member": {
                    cid: {"n_trades": len(self._config.cluster_mapping.get(cid, []))}
                    for cid in member_inputs.keys()
                },
                "metadata": {
                    "ensemble_version": self._ensemble_version,
                    "mode": mode,
                    "n_scenarios": combined.shape[0],
                    "n_targets": combined.shape[1] if combined.ndim > 1 else 1,
                },
            }

        if mode == "new_trades":
            raise NotImplementedError(
                "new_trades inference is not yet supported. The underlying "
                "HybridGnnRnnInferencePipeline does not implement "
                "_prepare_new_trade_inputs yet."
            )

        if mode != "new_scenarios":
            raise ValueError(f"Unknown mode: {mode}")

        from src.rade_ml_pt.pipelines.ensemble.infer import _dict_to_inference_context
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
            InferenceContext,
        )
        import pandas as pd

        member_inputs_built: Dict[str, Any] = {}
        per_member_meta: Dict[str, Dict[str, Any]] = {}

        for cid in self._config.cluster_ids:
            state = self._inference[cid]

            if not state.inference_context:
                raise ValueError(
                    f"No inference context available for cluster '{cid}'. "
                    f"Provide prebuilt member_inputs to run model-agnostic "
                    f"ensemble inference."
                )

            ctx = state.inference_context
            if not isinstance(ctx, InferenceContext):
                ctx = _dict_to_inference_context(
                    ctx, data_config_override=state.data_config,
                )

            inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
                ctx, mode="new_scenarios",
            )

            # Resolve elementary PnL: caller-provided > baseline from registry.
            # Expects a pd.DataFrame with columns matching the training order
            # (ctx.elementary_pnl.columns). If a raw np.ndarray is provided,
            # wrap it using the stored column names.
            raw_pnl = None
            if cluster_pnl_histories and cid in cluster_pnl_histories:
                raw_pnl = cluster_pnl_histories[cid]
            elif state.baseline_pnl is not None:
                raw_pnl = state.baseline_pnl
            else:
                raise ValueError(
                    f"No pnl_history provided for cluster '{cid}' and no "
                    f"baseline PnL available in the session."
                )

            if isinstance(raw_pnl, pd.DataFrame):
                pnl_df = raw_pnl
            elif ctx.elementary_pnl is not None:
                pnl_df = pd.DataFrame(
                    raw_pnl,
                    columns=ctx.elementary_pnl.columns.tolist(),
                )
            else:
                raise ValueError(
                    f"Cannot wrap raw PnL array for cluster '{cid}': no "
                    f"elementary_pnl reference available for column names."
                )

            # Standardise using the saved scaler (same as training).
            if ctx.elementary_scaler is not None:
                pnl_scaled = HybridGnnRnnInferencePipeline._standardise_pnl(
                    pnl_unscaled=pnl_df, scaler=ctx.elementary_scaler,
                )
                inputs.elementary_pnl = pd.DataFrame(
                    pnl_scaled,
                    columns=pnl_df.columns.tolist(),
                    index=pnl_df.index.tolist(),
                )
            else:
                inputs.elementary_pnl = pnl_df

            elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
                elementary_pnl=inputs.elementary_pnl,
                seq_length=ctx.data_config.seq_length,
                n_targets=len(inputs.target_indices),
            )

            result = HybridGnnRnnInferencePipeline.build_model_inputs(
                elem_seq=elem_seq,
                inputs=inputs,
                seq_length=ctx.data_config.seq_length,
            )

            member_inputs_built[cid] = result["inputs"]

            per_member_meta[cid] = {
                "n_trades": len(inputs.target_indices),
                "n_scenarios": elem_seq.shape[0],
                "has_new_trades": False,
            }

        combined = self._ensemble_model.predict(member_inputs_built)

        return {
            "predictions": combined,
            "per_member": per_member_meta,
            "metadata": {
                "ensemble_version": self._ensemble_version,
                "mode": mode,
                "n_scenarios": combined.shape[0],
                "n_targets": combined.shape[1] if combined.ndim > 1 else 1,
            },
        }

    def _route_new_trades(
        self, new_trade_attribs: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Route new trade attributes to clusters.

        If *new_trade_attribs* is already keyed by cluster_id, use as-is.
        Otherwise treat as a single trade and route via the TradeRouter.
        """
        result: Dict[str, Optional[Dict[str, Any]]] = {
            cid: None for cid in self._config.cluster_ids
        }

        if not new_trade_attribs:
            return result

        first_key = next(iter(new_trade_attribs))
        if first_key in self._config.cluster_ids:
            for cid, attribs in new_trade_attribs.items():
                result[cid] = attribs
            return result

        cid = self._ensemble_model.router.assign_new_trade(new_trade_attribs)
        result[cid] = new_trade_attribs
        return result

    # ==================================================================
    # Properties (UI helpers)
    # ==================================================================

    @property
    def is_metadata_loaded(self) -> bool:
        """True after :meth:`load_metadata` has run (Phase 1 complete)."""
        return self._config is not None

    @property
    def is_display_loaded(self) -> bool:
        """True when ensemble-level display and all per-cluster states are loaded."""
        if not self._config or self._ensemble_display is None:
            return False
        return set(self._config.cluster_ids) <= set(self._display.keys())

    @property
    def inference_ready_clusters(self) -> List[str]:
        """Cluster IDs with loaded :class:`ClusterInferenceState` (Phase 3), sorted."""
        return sorted(self._inference.keys())

    @property
    def all_inference_ready(self) -> bool:
        """True when every cluster in the ensemble has inference state loaded."""
        if not self._config:
            return False
        return set(self._config.cluster_ids) <= set(self._inference.keys())

    @property
    def config(self) -> Optional[EnsembleConfig]:
        """Loaded :class:`EnsembleConfig` from the registry, or ``None`` if Phase 1 not run."""
        return self._config

    @property
    def member_versions(self) -> Optional[Dict[str, str]]:
        """``{cluster_id: member_registry_version}`` from the registry load, or ``None``."""
        return self._member_versions

    @property
    def ensemble_version(self) -> Optional[str]:
        """Resolved ensemble version string (e.g. tag or id), or ``None`` if not loaded."""
        return self._ensemble_version

    @property
    def trade_cluster_map(self) -> Optional[Dict[str, str]]:
        """``{trade_id: cluster_id}`` from ``trade_cluster_map.json``, or ``None``."""
        return self._trade_cluster_map

    @property
    def member_summary(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Per-cluster summary dict from ``member_summary.json``, or ``None``."""
        return self._member_summary

    @property
    def display(self) -> Dict[str, ClusterDisplayState]:
        """Shallow copy of per-cluster display states (metrics, plots, attributes)."""
        return dict(self._display)

    @property
    def ensemble_display(self) -> Optional[EnsembleDisplayState]:
        """Portfolio-level eval artifacts (manifest, rollups, metrics) if Phase 2 ran."""
        return self._ensemble_display

    @property
    def ensemble_model(self) -> Any:
        """Assembled :class:`~src.rade_ml_pt.ensemble.model.EnsembleModel` once all members are loaded."""
        return self._ensemble_model

    @property
    def cluster_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Per-cluster attribute dicts used by routing (e.g. desk, product, ccy).

        Shape: ``{cluster_id: {attribute_name: value, ...}}`` from the config's
        cluster-key materialization. Empty dict if metadata is not loaded.
        """
        if self._config is None:
            return {}
        return self._config.get_cluster_keys_for_router() or {}

    # ==================================================================
    # Data helpers (dashboard-oriented)
    # ==================================================================

    def _ensure_all_cluster_displays(self) -> None:
        """Load display state for all clusters not yet cached, using parallel I/O."""
        missing = [cid for cid in self._config.cluster_ids if cid not in self._display]
        if not missing:
            return

        logger.info("Lazy-loading display state for %d clusters (parallel)", len(missing))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(missing))) as pool:
            futures = {
                pool.submit(self._load_cluster_display, cid): cid
                for cid in missing
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    self._display[cid] = future.result()
                except Exception:
                    logger.exception("Failed to load cluster display: %s", cid)

    def build_global_trade_catalogue(self) -> "pd.DataFrame":
        """Build a DataFrame of all target trades with attributes and cluster membership.

        Joins per-cluster ``target_attributes.json`` with cluster-level
        attributes so the dashboard can filter/aggregate by desk, product,
        ccy, or any other attribute across the entire portfolio.
        """
        import pandas as pd

        self._require_metadata()
        self._ensure_all_cluster_displays()

        rows: List[Dict[str, Any]] = []
        cluster_attrs = self.cluster_attributes

        for cid in self._config.cluster_ids:
            display = self._display.get(cid)
            if display is None:
                continue
            attribs = display.target_attributes

            if not attribs:
                for tid in self._config.cluster_mapping.get(cid, []):
                    row: Dict[str, Any] = {"trade_id": tid, "cluster_id": cid}
                    row.update(cluster_attrs.get(cid, {}))
                    rows.append(row)
                continue

            n = len(next(iter(attribs.values()), []))
            for i in range(n):
                row = {"cluster_id": cid}
                for key, values in attribs.items():
                    row[key] = values[i] if i < len(values) else None
                row.update(cluster_attrs.get(cid, {}))
                rows.append(row)

        return pd.DataFrame(rows)

    def get_prediction_store(self, split: str = "test") -> Optional[GlobalPredictionStore]:
        """Return a unified prediction/target store for *split*, building lazily.

        Reads per-member ``.npz`` files saved by the eval pipeline and
        slots each member's columns into the global trade order defined
        by ``manifest.json``.  Returns ``None`` if the manifest or
        prediction files are not available.
        """
        if split in self._prediction_stores:
            return self._prediction_stores[split]

        store = self._build_prediction_store(split)
        if store is not None:
            self._prediction_stores[split] = store
        return store

    @staticmethod
    def _load_member_npz(npz_path: Path, cid: str):
        """Load a single member's prediction .npz file (thread-safe)."""
        if not npz_path.exists():
            return cid, None, None
        data = np.load(str(npz_path))
        preds = data.get("predictions")
        targets = data.get("targets")
        return cid, preds, targets

    def _build_prediction_store(self, split: str) -> Optional[GlobalPredictionStore]:
        if self._ensemble_display is None or not self._ensemble_display.manifest:
            return None

        manifest = self._ensemble_display.manifest
        trade_ids: List[str] = manifest.get("trade_ids", [])
        cluster_ids_list: List[str] = manifest.get("cluster_ids", [])
        cluster_trade_indices = manifest.get("cluster_trade_indices", {})

        if not trade_ids or self.artifacts_dir is None:
            return None

        eval_dir = (
            self.artifacts_dir / "ensemble" / self._ensemble_version / "evaluation"
        )

        npz_paths = {
            cid: eval_dir / "members" / cid / "predictions" / f"{split}.npz"
            for cid in cluster_ids_list
        }

        loaded: Dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(cluster_ids_list))) as pool:
            futures = {
                pool.submit(self._load_member_npz, path, cid): cid
                for cid, path in npz_paths.items()
            }
            for future in as_completed(futures):
                cid, preds, targets = future.result()
                if preds is not None:
                    loaded[cid] = (preds, targets)

        if not loaded:
            return None

        n_total = len(trade_ids)
        first_preds = next(iter(loaded.values()))[0]
        n_scenarios = first_preds.shape[0]
        preds_global = np.zeros((n_scenarios, n_total), dtype=np.float32)
        targets_global = np.zeros((n_scenarios, n_total), dtype=np.float32)
        cluster_membership: List[str] = [""] * n_total

        for cid, (member_preds, member_targets) in loaded.items():
            indices = cluster_trade_indices.get(cid, [])
            for col_local, col_global in enumerate(indices):
                if col_local < member_preds.shape[1]:
                    preds_global[:, col_global] = member_preds[:, col_local]
                    targets_global[:, col_global] = member_targets[:, col_local]
                    cluster_membership[col_global] = cid

        return GlobalPredictionStore(
            predictions=preds_global,
            targets=targets_global,
            trade_ids=trade_ids,
            cluster_ids=cluster_membership,
            split=split,
        )

    def load_cluster_market_data(self, cluster_id: str) -> Dict[str, Any]:
        """Load cluster asset portfolio / risk-factor shock data.

        Returns ``{asset_name: {rf_name: np.ndarray}}``.  Cached after
        first load per cluster.
        """
        if cluster_id in self._market_data_cache:
            return self._market_data_cache[cluster_id]

        self._require_metadata()
        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version
        assets_path = version_dir / "cluster_assets.joblib"

        if not assets_path.exists():
            self._market_data_cache[cluster_id] = {}
            return {}

        import joblib
        portfolio = joblib.load(str(assets_path))

        result: Dict[str, Any] = {}
        for asset_name, asset in portfolio.items():
            rf_shocks = getattr(asset, "risk_factor_shocks", None)
            if rf_shocks is None:
                continue
            result[asset_name] = {}
            for rf_name, shocks in rf_shocks.items():
                if isinstance(shocks, dict):
                    result[asset_name][rf_name] = np.array(
                        list(shocks.values()), dtype=np.float64,
                    )
                else:
                    result[asset_name][rf_name] = np.asarray(shocks)

        self._market_data_cache[cluster_id] = result
        return result

    def load_cluster_graph_data(self, cluster_id: str) -> Dict[str, Any]:
        """Load graph adjacency and encoder feature data for one cluster.

        Returns dict with ``graph_results``, ``encoder_results``, and
        ``trade_universe`` keys.  Cached after first load per cluster.
        """
        if cluster_id in self._graph_data_cache:
            return self._graph_data_cache[cluster_id]

        self._require_metadata()
        import joblib

        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version

        data: Dict[str, Any] = {}

        graph_path = version_dir / "graph_results.joblib"
        if graph_path.exists():
            data["graph_results"] = joblib.load(str(graph_path))

        encoder_path = version_dir / "encoder_results.joblib"
        if encoder_path.exists():
            data["encoder_results"] = joblib.load(str(encoder_path))

        display = self.load_cluster_display(cluster_id)
        data["trade_universe"] = display.trade_universe

        self._graph_data_cache[cluster_id] = data
        return data

    # ==================================================================
    # Guards
    # ==================================================================

    def _require_metadata(self) -> None:
        if self._config is None:
            raise RuntimeError(
                "Session metadata not loaded. Call load_metadata() first."
            )

    def _require_inference(self) -> None:
        self._require_metadata()
        if not self.all_inference_ready:
            missing = set(self._config.cluster_ids) - set(self._inference.keys())
            raise RuntimeError(
                f"Inference state not loaded for clusters: {sorted(missing)}. "
                f"Call load_inference_state() first."
            )
        if self._ensemble_model is None:
            raise RuntimeError("EnsembleModel not assembled. This is a bug.")
