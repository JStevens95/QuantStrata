"""
Tests for experiment tracking.

One rule governs this module: tracking is never load-bearing. A model that
trained successfully but could not reach a tracking server has still trained
successfully, and the bundle on disk is the system of record. So the tests
here spend most of their effort on failure paths -- a read-only directory, an
unserialisable value -- confirming that none of them can take a run down.

The second thing under test is that the default does nothing. If the baseline
behaviour required a backend, every test, notebook and quick experiment would
need one configured, and tracking would be something you disable rather than
something you add.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rade_xl.core.runtime.context import Tracker
from src.rade_xl.storage.tracker import JsonlTracker, NullTracker


@pytest.fixture
def tracker(tmp_path):
    """
    Provide a file-backed tracker.

    Returns
    -------
    JsonlTracker
        The tracker.
    """
    return JsonlTracker(tmp_path / "events.jsonl")


def _events(tracker):
    """
    Read back every event a tracker has written.

    Parameters
    ----------
    tracker
        The tracker.

    Returns
    -------
    list of dict
        Decoded events, in order.
    """
    text = tracker.path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestTheDefaultDoesNothing:
    """``NullTracker`` is a working implementation of "do not track"."""

    def test_it_satisfies_the_protocol(self):
        """
        So it substitutes for a real tracker with no special-casing.

        A pipeline that had to check for ``None`` before every call would
        put the "is tracking configured?" question in a dozen places.
        """
        assert isinstance(NullTracker(), Tracker)

    def test_no_method_raises(self, tmp_path):
        """
        Empty, not ``NotImplementedError``.

        This is a complete implementation of a real decision, not a stub
        waiting to be filled in.
        """
        tracker = NullTracker()
        tracker.log_params({"seed": 1})
        tracker.log_metrics({"mae": 0.1}, step=0)
        tracker.log_artifact(tmp_path / "figure.png", name="curve")
        tracker.finish(succeeded=True)

    def test_it_writes_nothing(self, tmp_path):
        """A run with default tracking leaves no stray files behind."""
        NullTracker().log_params({"seed": 1})
        assert list(tmp_path.iterdir()) == []


class TestEventRecording:
    """What the file-backed tracker writes."""

    def test_it_satisfies_the_protocol(self, tracker):
        """Both implementations have to be interchangeable."""
        assert isinstance(tracker, Tracker)

    def test_parameters_are_recorded(self, tracker):
        """The configuration a run was launched with."""
        tracker.log_params({"seed": 3, "model": "demo"})
        assert _events(tracker)[0]["params"]["seed"] == 3

    def test_parameters_are_canonicalised(self, tracker):
        """
        Through the same canonical form the spec digest uses.

        So two runs whose configurations differ only in key order produce
        identical parameter records, and a diff between them is meaningful.
        """
        tracker.log_params({"b": 1, "a": 2})
        assert list(_events(tracker)[0]["params"]) == ["a", "b"]

    def test_metrics_carry_their_step(self, tracker):
        """
        Which is what makes a sequence of events a curve.

        Without it, fifty epoch records are fifty unordered points.
        """
        tracker.log_metrics({"loss": 0.5}, step=7)
        assert _events(tracker)[0]["step"] == 7

    def test_a_final_metric_has_no_step(self, tracker):
        """
        ``None`` distinguishes a summary metric from an epoch metric.

        A test-set score plotted as epoch zero would be actively misleading.
        """
        tracker.log_metrics({"mae": 0.25})
        assert _events(tracker)[0]["step"] is None

    def test_an_artifact_defaults_to_its_file_name(self, tracker, tmp_path):
        """A sensible name beats an empty one in a listing."""
        tracker.log_artifact(tmp_path / "training_curve.png")
        assert _events(tracker)[0]["name"] == "training_curve.png"

    def test_an_artifact_can_be_named_explicitly(self, tracker, tmp_path):
        """
        Because the file name is an implementation detail.

        ``metrics_test.png`` means less to a reader than "test metrics".
        """
        tracker.log_artifact(tmp_path / "f.png", name="test metrics")
        assert _events(tracker)[0]["name"] == "test metrics"

    def test_the_terminal_event_records_the_outcome(self, tracker):
        """
        So an interrupted run is distinguishable from a finished one.

        A run whose events simply stop could be either, and treating a
        crashed run's last metric as its final metric is how a bad model
        gets promoted.
        """
        tracker.finish(succeeded=False)
        assert _events(tracker)[0] == {"event": "finish", "succeeded": False}

    def test_events_are_appended_in_order(self, tracker):
        """
        One line per event, so ``tail -f`` follows a run in progress.

        And a crashed run still leaves everything written before the crash.
        """
        tracker.log_params({"seed": 1})
        tracker.log_metrics({"loss": 1.0}, step=0)
        tracker.finish(succeeded=True)
        assert [event["event"] for event in _events(tracker)] == [
            "params",
            "metrics",
            "finish",
        ]

    def test_the_parent_directory_is_created(self, tmp_path):
        """
        A run's output directory may not exist when the tracker is built.

        Failing at construction would make tracking load-bearing at the
        earliest possible moment.
        """
        JsonlTracker(tmp_path / "nested" / "deeper" / "events.jsonl").finish(succeeded=True)
        assert (tmp_path / "nested" / "deeper" / "events.jsonl").is_file()


class TestTrackingIsNeverLoadBearing:
    """Every failure path, confirmed harmless."""

    def test_an_unwritable_path_does_not_raise(self, tmp_path):
        """
        The central guarantee.

        A full disk, a revoked permission or an unmounted share must not
        fail a run that has otherwise completed.
        """
        directory = tmp_path / "events.jsonl"
        directory.mkdir()
        JsonlTracker(directory).log_metrics({"mae": 0.1})

    def test_a_failure_is_warned_about_rather_than_hidden(self, tmp_path, caplog):
        """
        Swallowed is not the same as silent.

        Someone has to be able to find out why their dashboard is empty.
        """
        directory = tmp_path / "events.jsonl"
        directory.mkdir()
        with caplog.at_level("WARNING"):
            JsonlTracker(directory).finish(succeeded=True)
        assert "tracking event" in caplog.text

    def test_a_run_can_finish_after_a_failed_write(self, tmp_path):
        """
        One failure must not poison the tracker.

        Otherwise a transient error partway through a run would silently
        stop recording everything after it.
        """
        directory = tmp_path / "events.jsonl"
        directory.mkdir()
        tracker = JsonlTracker(directory)
        tracker.log_params({"seed": 1})
        tracker.finish(succeeded=True)

    def test_an_unserialisable_value_is_coerced_rather_than_raised(self, tracker):
        """
        A spec may legitimately carry a path or an enum.

        Refusing to record it, or worse raising, would make tracking fail on
        exactly the configurations worth recording.
        """
        tracker.log_artifact(Path("/tmp/figure.png"), name="curve")
        assert _events(tracker)[0]["path"].endswith("figure.png")
