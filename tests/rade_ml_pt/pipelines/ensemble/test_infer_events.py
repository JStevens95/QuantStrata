"""Unit tests for the lifecycle event emitter (``infer_events.py``).

Covers the standalone helpers:
:func:`event` factory shape, :func:`noop_emit` is a no-op,
:class:`EventCollector` accumulates and is thread-safe, and the
:class:`TypedActivityEntry` ↔ wire-dict round-trip preserves every
field.

The previous pipeline-integration suite (``TestPipelineEmits``) was
removed when the pre-built ``member_inputs`` short-circuit was
deleted from :class:`EnsembleInferencePipeline`.  Fresh integration
tests against the staged path (``load_scenarios`` →
``validate_scenarios`` → ``run_inference``) will be added once the
synthetic-staged-path fixture lands.
"""
from __future__ import annotations

import threading
from typing import List

from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    STAGE_INFERENCE,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_RUNNING,
    EventCollector,
    TypedActivityEntry,
    event,
    noop_emit,
)


# ─────────────────────────────────────────────────────────────────────
# Unit tests — helpers in infer_events.py
# ─────────────────────────────────────────────────────────────────────


class TestEventFactory:
    def test_required_keys_present(self):
        e = event(STAGE_INFERENCE, "Forward pass complete")
        assert set(e.keys()) >= {"id", "stage", "phase", "status", "ts"}
        assert e["stage"] == STAGE_INFERENCE
        assert e["phase"] == "Forward pass complete"
        assert e["status"] == STATUS_OK   # default

    def test_optional_keys_omitted_when_none(self):
        e = event(STAGE_INFERENCE, "Loaded")
        assert "target" not in e
        assert "detail" not in e

    def test_optional_keys_present_when_passed(self):
        e = event(
            STAGE_INFERENCE, "Failed",
            status=STATUS_FAIL, target="cluster_2", detail="OOM",
        )
        assert e["target"] == "cluster_2"
        assert e["detail"] == "OOM"

    def test_id_unique_across_calls(self):
        ids = {event(STAGE_INFERENCE, "x")["id"] for _ in range(50)}
        assert len(ids) == 50

    def test_ts_is_iso_8601_utc(self):
        e = event(STAGE_INFERENCE, "x")
        assert e["ts"].endswith("+00:00")  # UTC marker
        # second precision: never microseconds
        assert "." not in e["ts"]


class TestTypedActivityEntryRoundTrip:
    def test_round_trip_preserves_all_fields(self):
        entry = TypedActivityEntry(
            stage=STAGE_INFERENCE, phase="x", status=STATUS_OK,
            target="t", detail="d",
        )
        d = entry.to_dict()
        assert d["stage"]  == "inference"
        assert d["phase"]  == "x"
        assert d["status"] == "ok"
        assert d["target"] == "t"
        assert d["detail"] == "d"
        assert "id" in d and "ts" in d


class TestNoopEmit:
    def test_returns_none(self):
        assert noop_emit({"phase": "x"}) is None

    def test_accepts_any_dict(self):
        # No exception even on garbage input — noop is total.
        noop_emit({})
        noop_emit({"unrelated": [1, 2, 3]})


class TestEventCollector:
    def test_call_appends(self):
        coll = EventCollector()
        coll(event(STAGE_INFERENCE, "a"))
        coll(event(STAGE_INFERENCE, "b"))
        assert len(coll) == 2
        assert [e["phase"] for e in coll.snapshot()] == ["a", "b"]

    def test_snapshot_is_a_copy(self):
        coll = EventCollector()
        coll(event(STAGE_INFERENCE, "a"))
        snap = coll.snapshot()
        snap.append({"injected": True})
        assert len(coll) == 1   # original unchanged

    def test_clear_empties_buffer(self):
        coll = EventCollector()
        coll(event(STAGE_INFERENCE, "a"))
        coll.clear()
        assert len(coll) == 0
        assert coll.snapshot() == []

    def test_thread_safe_under_concurrent_appends(self):
        coll = EventCollector()
        n_threads = 8
        n_per_thread = 250

        def worker(tid: int) -> None:
            for i in range(n_per_thread):
                coll(event(STAGE_INFERENCE, f"t{tid}-{i}"))

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(coll) == n_threads * n_per_thread
        # No duplicates / no losses — each event has a unique id.
        ids = {e["id"] for e in coll.snapshot()}
        assert len(ids) == n_threads * n_per_thread

