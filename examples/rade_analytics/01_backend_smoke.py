"""Phase A exit-criterion smoke test for :class:`RadeBackend`.

Stands up a :class:`RadeApiClient` against a running API server (see
``examples/rade_ml_pt/hybrid_gnn_rnn/12_run_prism_api.py``), wraps it in
a :class:`RadeBackend` with a :class:`NoOpCache`, and exercises every
public method once.  Prints a compact pass/fail line per call so you can
eyeball the whole matrix in < 2 seconds.

Run::

    python examples/rade_analytics/01_backend_smoke.py

Pass criterion: every line starts with ``[ok]`` for the splits + cluster
your current artifact store actually contains.
"""
from __future__ import annotations

import logging

from src.rade_ml_pt.ensemble.api.client import RadeApiClient
from src.ui.apps.rade_analytics.data.backend import NoOpCache, RadeBackend

API_URL = "http://localhost:8000"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rade-backend-smoke")


def _line(label: str, result) -> None:
    if result.ok:
        log.info("  [ok]   %s", label)
    else:
        log.warning(
            "  [fail] %s → %s (%s)",
            label,
            result.error,
            result.status_code,
        )


def main() -> None:
    with RadeApiClient(API_URL, timeout=30.0) as client:
        backend = RadeBackend(client=client, cache=NoOpCache())

        log.info("=== meta ===")
        _line("health()", backend.health())
        versions_res = backend.versions()
        _line("versions()", versions_res)
        if not versions_res.ok:
            log.error("API unreachable — aborting")
            return

        log.info("=== overview + metrics ===")
        overview_res = backend.overview()
        _line("overview()", overview_res)
        _line("ensemble_metrics_df()", backend.ensemble_metrics_df())
        _line("per_member_metrics_df()", backend.per_member_metrics_df())

        splits = overview_res.data.splits_available if overview_res.ok else []
        clusters_res = backend.clusters()
        _line("clusters()", clusters_res)
        _line("clusters_df()", backend.clusters_df())
        cluster_ids = (
            [c.cluster_id for c in clusters_res.data.clusters]
            if clusters_res.ok
            else []
        )

        log.info("=== per-split tabular ===")
        for split in splits:
            _line(f"portfolio_df({split!r})", backend.portfolio_df(split))
            _line(
                f"cluster_timeseries_df({split!r})",
                backend.cluster_timeseries_df(split),
            )
            _line(f"trades_df({split!r})", backend.trades_df(split))
            _line(
                f"group_correlations_df({split!r})",
                backend.group_correlations_df(split),
            )
            _line(f"completeness_df({split!r})", backend.completeness_df(split))
            _line(
                f"feature_summary_df({split!r})",
                backend.feature_summary_df(split),
            )

        log.info("=== graph stats ===")
        _line("graph_stats_df()", backend.graph_stats_df())

        if cluster_ids and splits:
            log.info("=== predictions (binary, uncached) ===")
            cid, split = cluster_ids[0], splits[0]
            pred_res = backend.predictions(cluster_id=cid, split=split)
            _line(f"predictions(cluster_id={cid!r}, split={split!r})", pred_res)
            if pred_res.ok:
                log.info(
                    "    predictions=%s, targets=%s",
                    pred_res.data["predictions"].shape,
                    pred_res.data["targets"].shape,
                )

        log.info("=== failure-path spot-check ===")
        bad = backend.trades_df("no-such-split")
        _line(
            "trades_df('no-such-split') → should be [fail]",
            bad if not bad.ok else bad,
        )


if __name__ == "__main__":
    main()
