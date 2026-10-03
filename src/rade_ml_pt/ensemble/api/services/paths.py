"""Path resolution for an ensemble evaluation run.

Canonical layout written by :mod:`rade_ml_pt.pipelines.ensemble.eval`:

.. code-block:: text

    {artifacts_dir}/ensemble/{version}/evaluation/
        manifest.json
        trade_cluster_map.json
        cluster_attributes.parquet
        ensemble_metrics.parquet
        per_member_metrics.parquet
        graph_stats.parquet
        portfolio_summary/
            portfolio_timeseries_{split}.parquet
        cluster_summary/
            cluster_timeseries_{split}.parquet
        trade_metrics/
            trade_metrics_{split}.parquet
        group_correlations/
            group_correlations_{split}.parquet
        quality/
            completeness_{split}.parquet
            feature_summary_{split}.parquet
        members/
            {cluster_id}/
                predictions/
                    {split}.npz                              # scaled space (legacy name)
                    {split}_original.npz                     # original space (Phase 3.1)
                graph_results.joblib
                trade_universe.json
                training_curves.parquet

This module is the single source of truth for those filenames inside the
API.  If the evaluation pipeline ever renames or relocates an artifact,
update this class and the readers follow automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Immutable path resolver for one ``(artifacts_dir, version)`` pair."""

    artifacts_dir: Path
    version: str

    # ── Root ──────────────────────────────────────────────────────
    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "ensemble" / self.version / "evaluation"

    # ── Run metadata ──────────────────────────────────────────────
    @property
    def manifest(self) -> Path:
        return self.eval_dir / "manifest.json"

    @property
    def trade_cluster_map(self) -> Path:
        return self.eval_dir / "trade_cluster_map.json"

    # ── Ensemble-scoped parquets ──────────────────────────────────
    @property
    def cluster_attributes(self) -> Path:
        return self.eval_dir / "cluster_attributes.parquet"

    @property
    def ensemble_metrics(self) -> Path:
        return self.eval_dir / "ensemble_metrics.parquet"

    @property
    def per_member_metrics(self) -> Path:
        return self.eval_dir / "per_member_metrics.parquet"

    @property
    def graph_stats(self) -> Path:
        return self.eval_dir / "graph_stats.parquet"

    # ── Per-split parquets ────────────────────────────────────────
    def portfolio_timeseries(self, split: str) -> Path:
        return (
            self.eval_dir
            / "portfolio_summary"
            / f"portfolio_timeseries_{split}.parquet"
        )

    def cluster_timeseries(self, split: str) -> Path:
        return (
            self.eval_dir
            / "cluster_summary"
            / f"cluster_timeseries_{split}.parquet"
        )

    def trade_metrics(self, split: str) -> Path:
        return self.eval_dir / "trade_metrics" / f"trade_metrics_{split}.parquet"

    def group_correlations(self, split: str) -> Path:
        return (
            self.eval_dir
            / "group_correlations"
            / f"group_correlations_{split}.parquet"
        )

    def completeness(self, split: str) -> Path:
        return self.eval_dir / "quality" / f"completeness_{split}.parquet"

    def feature_summary(self, split: str) -> Path:
        return self.eval_dir / "quality" / f"feature_summary_{split}.parquet"

    # ── Raw per-member shards (NPZ) ───────────────────────────────
    def member_predictions(
        self,
        cluster_id: str,
        split:      str,
        space:      str = "scaled",
    ) -> Path:
        """Per-cluster NPZ shard for one PnL space.

        Filename convention preserves backwards compatibility:

          - ``space="scaled"``    → ``{split}.npz``           (legacy)
          - ``space="original"``  → ``{split}_original.npz``  (Phase 3.1)

        Each NPZ carries four arrays:
          - ``predictions``      float32 ``[n_scenarios, n_trades]``
          - ``targets``          float32 ``[n_scenarios, n_trades]``
          - ``trade_ids``        unicode ``[n_trades]``
          - ``scenario_labels``  unicode ``[n_scenarios]``

        All arrays are pickle-free so consumers can keep using
        ``np.load(path, allow_pickle=False)``.
        """
        member_dir = self.eval_dir / "members" / cluster_id / "predictions"
        filename = (
            f"{split}.npz" if space == "scaled" else f"{split}_{space}.npz"
        )
        return member_dir / filename

    # ── Per-member graph artefacts (staged by the eval pipeline) ──
    def member_graph_results(self, cluster_id: str) -> Path:
        """Sparse adjacency joblib staged from the training registry.

        Used by the Trade-Graph UI tab to render cluster networks.
        The joblib is expected to contain keys ``sparse_indices``,
        ``sparse_values`` and ``sparse_shape`` (see
        ``_save_graph_stats_parquet`` in the eval pipeline for the
        producer side).
        """
        return self.eval_dir / "members" / cluster_id / "graph_results.joblib"

    def member_trade_universe(self, cluster_id: str) -> Path:
        """Trade-universe JSON staged from the training registry.

        Provides ``target_ids`` / ``elementary_ids`` for node colouring
        and trade-attribute lookup in the Selected-Trade card.
        """
        return self.eval_dir / "members" / cluster_id / "trade_universe.json"

    def member_training_curves(self, cluster_id: str) -> Path:
        """Per-epoch training curves parquet staged from the registry.

        Schema (B-level contract, trainer-side §11.15.1):

        ``epoch``      int32    (required — monotonically increasing)
        ``train_loss`` float32  (required — always present)
        *other cols*   float32  (any additional per-epoch series the
                                 trainer emitted — e.g. ``val_loss``,
                                 ``mae``, ``val_mae``).

        Consumed by the Cluster Deep-Dive training-curves chart.
        Column set is not fixed so the UI introspects available
        metrics at render time.
        """
        return self.eval_dir / "members" / cluster_id / "training_curves.parquet"

    def member_elementary_pnl(self, cluster_id: str) -> Path:
        """Elementary-leg PnL parquet staged from the registry.

        Schema:

        * Rows  — one per scenario in this cluster's universe
        * Cols  — one per elementary trade id (raw PnL, ``float32``)

        Elementary trades are *model inputs*, not predictions, so the
        Cluster Deep-Dive "Elementary PnL Explorer" plots these values
        directly without a prediction / target overlay.  The producer
        side of the contract lives in
        :mod:`src.rade_ml_pt.data.hybrid_gnn_rnn.build` (see
        ``elementary_pnl_scaled``); :func:`_stage_member_graph_artefacts`
        in the eval pipeline copies the file alongside
        ``training_curves.parquet`` so the API can read it without
        knowing the training registry layout.
        """
        return self.eval_dir / "members" / cluster_id / "elementary_pnl.parquet"
