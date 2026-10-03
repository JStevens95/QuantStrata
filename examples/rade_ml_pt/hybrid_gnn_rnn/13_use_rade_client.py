"""Smoke-test script for the typed Rade API client.

Point this at a running PRISM / Rade FastAPI server (see
``12_run_prism_api.py``) and it will hit every endpoint exactly once,
printing a one-line summary per call.  Useful for:

* verifying every route on a freshly-rebuilt server,
* confirming the Pydantic round-trip (server → wire → client) works
  for every artifact family,
* sanity-checking the binary NPZ stream decodes into numpy arrays of
  the expected shape.

Run:

    python examples/rade_ml_pt/hybrid_gnn_rnn/13_use_rade_client.py
"""
from __future__ import annotations

import logging

from src.rade_ml_pt.ensemble.api.client import RadeApiClient, RadeApiError

BASE_URL = "http://localhost:8000"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rade-smoke")


def _try(label: str, fn):
    """Run *fn* and log a compact success / failure line."""
    try:
        result = fn()
    except RadeApiError as e:
        log.warning("  [skip] %s: %s %s", label, e.status_code, e.detail)
        return None
    log.info("  [ok]   %s", label)
    return result


def main() -> None:
    with RadeApiClient(BASE_URL, timeout=30.0) as c:
        log.info("=== meta ===")
        health = _try("health", c.health)
        versions = _try("versions", c.versions)

        if not health or not versions:
            log.error("server not healthy — aborting")
            return

        log.info("  active version: %s", versions.active)
        log.info("  available:      %s", versions.available)

        log.info("=== overview ===")
        overview = _try("overview", c.overview)
        if overview is None:
            return
        splits = overview.splits_available
        cluster_ids: list[str] = []

        log.info("=== clusters ===")
        clusters = _try("clusters()", c.clusters)
        if clusters is not None:
            cluster_ids = [c_.cluster_id for c_ in clusters.clusters]
            log.info("  found %d clusters", len(cluster_ids))

        log.info("=== metrics ===")
        _try("ensemble_metrics", c.ensemble_metrics)
        _try("per_member_metrics()", c.per_member_metrics)
        for split in splits:
            _try(f"per_member_metrics(split={split!r})",
                 lambda split=split: c.per_member_metrics(split=split))

        log.info("=== per-split data-family endpoints ===")
        for split in splits:
            _try(f"portfolio(split={split!r})",
                 lambda split=split: c.portfolio(split))
            _try(f"cluster_timeseries(split={split!r})",
                 lambda split=split: c.cluster_timeseries(split))
            _try(f"trades(split={split!r})",
                 lambda split=split: c.trades(split))
            _try(f"group_correlations(split={split!r})",
                 lambda split=split: c.group_correlations(split))
            _try(f"completeness(split={split!r})",
                 lambda split=split: c.completeness(split))
            _try(f"feature_summary(split={split!r})",
                 lambda split=split: c.feature_summary(split))

        log.info("=== graph stats ===")
        _try("graph_stats()", c.graph_stats)

        if cluster_ids and splits:
            log.info("=== predictions (NPZ) ===")
            cid, split = cluster_ids[0], splits[0]
            result = _try(
                f"predictions(cluster_id={cid!r}, split={split!r})",
                lambda: c.predictions(cluster_id=cid, split=split),
            )
            if result is not None:
                log.info(
                    "  predictions shape=%s, targets shape=%s",
                    result["predictions"].shape,
                    result["targets"].shape,
                )


if __name__ == "__main__":
    main()
