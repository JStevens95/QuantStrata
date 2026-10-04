"""
Tests for the run context.

Three behaviours are the point of this module.

A **failing hook never fails a run**, and neither does a failing tracker. Both
are observers; the bundle on disk is the system of record. A broken tracking
credential must not destroy four hours of training.

A **job's seed is derived by hashing**, so re-running one failed member of a
job set alone reproduces exactly what it would have produced inside the set.

The **context is frozen**, so a stage cannot reconfigure the run it is part of.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from src.rade_qnet.core.lifecycle.context import Catalog, RunContext, Tracker
from src.rade_qnet.core.lifecycle.hooks import PipelineHook
from src.rade_qnet.core.provenance.logging import configure_logging, current_context
from src.rade_qnet.storage.runs.catalog import InMemoryCatalog
from src.rade_qnet.storage.runs.tracker import NullTracker
from src.rade_qnet.testkit.fixtures import RecordingHook


@pytest.fixture
def context(run_directory):
    """
    Provide a context with one recording hook attached.

    Parameters
    ----------
    run_directory
        Per-test working directory.

    Returns
    -------
    RunContext
        A context whose single hook records every event.
    """
    return RunContext(
        run_id="r-001",
        spec_digest="d" * 64,
        output_directory=run_directory,
        seed=100,
        hooks=(RecordingHook(),),
    )


class TestImmutability:
    """A stage cannot reconfigure its own run."""

    def test_the_context_is_frozen(self, context):
        """
        Assignment raises.

        A stage that could change the output directory or the seed would make
        the bundle a record of something that did not happen.
        """
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.seed = 999


class TestDirectories:
    """Derived paths are consistent, so two stages agree on where to write."""

    def test_reports_and_bundles_sit_under_the_output_directory(self, context):
        """Everything a run writes is beneath one root."""
        assert context.reports_directory.parent == context.output_directory
        assert context.bundles_directory.parent == context.output_directory

    def test_the_paths_are_stable(self, context):
        """
        Two calls return the same path.

        A derived path that varied -- by timestamp, say -- would mean the
        writer and the reader of a report disagree about where it is.
        """
        assert context.reports_directory == context.reports_directory


class TestJobDerivation:
    """Per-job contexts are derived without mutating the parent."""

    def test_two_jobs_get_different_seeds(self, context):
        """Members of a job set must be independent."""
        assert context.for_job("EURUSD").seed != context.for_job("USDJPY").seed

    def test_a_job_seed_is_reproducible(self, context):
        """
        Deriving twice gives the same seed.

        This is what lets a single failed job be re-run on its own and
        reproduce what it would have done in the full set.
        """
        assert context.for_job("EURUSD").seed == context.for_job("EURUSD").seed

    def test_job_seeds_do_not_depend_on_order(self, context):
        """
        Derivation is by hash, not by position.

        With ``base + index`` seeding, re-running a subset of jobs -- or
        reordering them -- would change every member's seed, so a job set
        would not be reproducible job by job.
        """
        in_order = [context.for_job(job).seed for job in ("a", "b", "c")]
        reversed_order = [context.for_job(job).seed for job in ("c", "b", "a")]
        assert in_order == list(reversed(reversed_order))

    def test_each_job_gets_its_own_directory(self, context):
        """Two members writing to one directory would overwrite each other."""
        assert context.for_job("EURUSD").output_directory != (
            context.for_job("USDJPY").output_directory
        )

    def test_the_directory_can_be_overridden(self, context, tmp_path):
        """A caller may place a job's output wherever it likes."""
        elsewhere = tmp_path / "elsewhere"
        assert context.for_job("EURUSD", output_directory=elsewhere).output_directory == elsewhere

    def test_observers_and_stores_are_inherited(self, context):
        """A job's events reach the same hooks as the run's."""
        derived = context.for_job("EURUSD")
        assert derived.hooks == context.hooks
        assert derived.run_id == context.run_id
        assert derived.spec_digest == context.spec_digest

    def test_the_parent_is_unchanged(self, context):
        """Derivation returns a new context rather than mutating this one."""
        context.for_job("EURUSD")
        assert context.job_id is None


class TestActivate:
    """Activation binds the identifiers used to attribute log lines."""

    def test_identifiers_bind_inside_the_block(self, context):
        """A log line emitted in a stage carries the run and the stage."""
        with context.activate(stage="fit"):
            assert current_context() == {"run_id": "r-001", "stage": "fit"}

    def test_the_job_identifier_binds_when_present(self, context):
        """A job set's logs must be attributable to the job."""
        with context.for_job("EURUSD").activate():
            assert current_context()["job_id"] == "EURUSD"

    def test_identifiers_are_released_on_exit(self, context):
        """Nothing leaks into the next stage."""
        with context.activate(stage="fit"):
            pass
        assert current_context() == {}


class TestNotify:
    """Hook fan-out is tolerant of a broken hook."""

    def test_every_hook_is_called(self, run_directory):
        """Observers see events in the order they were registered."""
        first, second = RecordingHook(), RecordingHook()
        context = RunContext(
            run_id="r",
            spec_digest="d",
            output_directory=run_directory,
            hooks=(first, second),
        )
        context.notify(lambda hook: hook.on_stage_start("fit"), description="fit start")
        assert first.names() == ("stage_start",)
        assert second.names() == ("stage_start",)

    def test_a_raising_hook_does_not_propagate(self, run_directory, caplog):
        """
        The central rule: an observer never fails a run.

        A broken dashboard or an expired tracking credential must not destroy
        a completed training run.
        """

        class Broken(PipelineHook):
            """Raises on every stage start."""

            def on_stage_start(self, stage: str) -> None:
                """Raise, to prove the failure is contained."""
                message = "hook is broken"
                raise RuntimeError(message)

        context = RunContext(
            run_id="r", spec_digest="d", output_directory=run_directory, hooks=(Broken(),)
        )
        with caplog.at_level(logging.WARNING):
            context.notify(lambda hook: hook.on_stage_start("fit"), description="fit start")
        assert "Broken" in caplog.text

    def test_a_raising_hook_does_not_silence_the_others(self, run_directory):
        """
        One broken observer must not suppress the working ones.

        Otherwise registration order would decide whether a dashboard
        receives events.
        """

        class Broken(PipelineHook):
            """Raises on every stage start."""

            def on_stage_start(self, stage: str) -> None:
                """Raise."""
                message = "hook is broken"
                raise RuntimeError(message)

        working = RecordingHook()
        context = RunContext(
            run_id="r",
            spec_digest="d",
            output_directory=run_directory,
            hooks=(Broken(), working),
        )
        context.notify(lambda hook: hook.on_stage_start("fit"), description="fit start")
        assert working.names() == ("stage_start",)


class TestTracking:
    """Tracking is optional and never load-bearing."""

    def test_metrics_without_a_tracker_are_a_no_op(self, context):
        """A run with no tracker configured still runs."""
        context.track_metrics({"mae": 0.1})

    def test_artifacts_without_a_tracker_are_a_no_op(self, context):
        """Same, for artifacts."""
        context.track_artifact(Path("/tmp/x"))

    def test_a_failing_tracker_is_tolerated(self, run_directory, caplog):
        """
        An unreachable tracking server does not end the run.

        The bundle on disk is the system of record; the tracker is a
        convenience.
        """

        class Offline:
            """A tracker that cannot reach its server."""

            def log_params(self, params):
                """Raise."""
                raise ConnectionError

            def log_metrics(self, metrics, *, step=None):
                """Raise."""
                raise ConnectionError

            def log_artifact(self, path, *, name=None):
                """Raise."""
                raise ConnectionError

            def finish(self, *, succeeded):
                """Raise."""
                raise ConnectionError

        context = RunContext(
            run_id="r", spec_digest="d", output_directory=run_directory, tracker=Offline()
        )
        with caplog.at_level(logging.WARNING):
            context.track_metrics({"mae": 0.1})
            context.track_artifact(Path("/tmp/x"))
        assert "tracker failed" in caplog.text


class TestProtocols:
    """Storage satisfies the protocols core declares, structurally."""

    def test_the_in_memory_catalog_satisfies_the_catalog_protocol(self):
        """
        ``core`` declares the interface; ``storage`` implements it.

        The dependency may not run the other way, which is why the protocol
        lives in ``core`` at all.
        """
        assert isinstance(InMemoryCatalog(), Catalog)

    def test_the_null_tracker_satisfies_the_tracker_protocol(self):
        """The default tracker is a real implementation of "do not track"."""
        assert isinstance(NullTracker(), Tracker)

    def test_an_unrelated_object_satisfies_neither(self):
        """The checks discriminate rather than accepting anything."""
        assert not isinstance(object(), Catalog)
        assert not isinstance(object(), Tracker)


class TestDescribe:
    """The opening log line states what the run was configured with."""

    def test_every_salient_field_is_described(self, context):
        """
        A run's own log says how it was set up.

        When a result is questioned months later, this is usually the first
        thing anyone reads.
        """
        configure_logging(level=logging.CRITICAL, force=True)
        rendered = "\n".join(context.describe())
        assert "r-001" in rendered
        assert "100" in rendered
        assert "RecordingHook" in rendered
