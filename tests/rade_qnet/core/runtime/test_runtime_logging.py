"""
Tests for contextual logging.

Two properties are load-bearing. First, configuration never happens at import:
a library that installs a root handler when imported takes over the logging of
every application that uses it. Second, the run, job and stage identifiers are
restored on exit, so a nested stage cannot leak its name to its parent -- which
would make a job set's logs attribute work to the wrong job.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from src.rade_qnet.core.runtime.logging import (
    ROOT_LOGGER_NAME,
    apply_context_payload,
    bound_context,
    configure_logging,
    context_payload,
    current_context,
    get_logger,
)


@pytest.fixture(autouse=True)
def _clear_context():
    """
    Ensure each test starts and ends with no bound identifiers.

    The identifiers live in context variables, which are process-wide. A test
    that left one bound would silently change what a later test observes.
    """
    yield
    apply_context_payload({})


class TestLoggerNaming:
    """Logger names place every record under one hierarchy."""

    def test_loggers_live_under_the_framework_root(self):
        """One root means an application can configure the framework alone."""
        assert get_logger("rade_qnet.core.runtime.pipeline").name.startswith(ROOT_LOGGER_NAME)

    def test_the_src_prefix_is_stripped(self):
        """
        A module imported as ``src.rade_qnet...`` still logs as ``rade_qnet...``.

        The repository imports through ``src``, an installed distribution does
        not. Without stripping, the same module logs under two different names
        depending on how it was imported, and a log filter configured for one
        silently misses the other.
        """
        assert get_logger("src.rade_qnet.storage.bundle").name == "rade_qnet.storage.bundle"


class TestConfiguration:
    """Configuration is explicit, and does not happen on import."""

    def test_importing_installs_no_handler(self):
        """
        The framework root has no handler until asked.

        If this fails, importing ``rade_qnet`` changes the logging behaviour of
        the importing application, which is not ours to change.
        """
        fresh = logging.getLogger("rade_qnet.test_no_autoconfig")
        assert not fresh.handlers

    def test_configure_writes_to_the_given_stream(self):
        """Records reach the stream the caller nominated."""
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        get_logger("rade_qnet.test").info("hello from the test")
        assert "hello from the test" in stream.getvalue()

    def test_configure_is_idempotent_without_force(self):
        """
        Calling twice does not double every record.

        A pipeline and a job-set driver may both configure logging; duplicated
        handlers would emit each line as many times as configure was called.
        """
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        configure_logging(level=logging.INFO, stream=stream)
        get_logger("rade_qnet.test").info("once")
        assert stream.getvalue().count("once") == 1


class TestBoundContext:
    """Identifiers bind for a block and are restored afterwards."""

    def test_identifiers_are_visible_inside_the_block(self):
        """A stage's logs can be attributed while the stage runs."""
        with bound_context(run_id="r-1", job_id="EURUSD", stage="fit"):
            assert current_context() == {"run_id": "r-1", "job_id": "EURUSD", "stage": "fit"}

    def test_identifiers_are_restored_on_exit(self):
        """Nothing leaks past the block."""
        with bound_context(run_id="r-1", stage="fit"):
            pass
        assert current_context() == {}

    def test_identifiers_are_restored_after_an_exception(self):
        """
        A failing stage does not leave its name bound.

        Otherwise every subsequent log line in the run would be attributed to
        the stage that already failed.
        """
        with pytest.raises(RuntimeError), bound_context(run_id="r-1", stage="fit"):
            raise RuntimeError("stage failed")
        assert current_context() == {}

    def test_nesting_restores_the_outer_stage(self):
        """
        An inner stage's name does not outlive it.

        This is what lets a pipeline call a sub-pipeline and still have its
        own later stages logged under its own name.
        """
        with bound_context(run_id="r-1", stage="outer"):
            with bound_context(stage="inner"):
                assert current_context()["stage"] == "inner"
            assert current_context()["stage"] == "outer"

    def test_partial_binding_leaves_other_identifiers_alone(self):
        """Binding a stage does not clear the run it belongs to."""
        with bound_context(run_id="r-1", job_id="EURUSD"), bound_context(stage="fit"):
            assert current_context() == {"run_id": "r-1", "job_id": "EURUSD", "stage": "fit"}


class TestRecordDecoration:
    """Bound identifiers reach the emitted record."""

    def test_identifiers_appear_in_the_output(self):
        """
        A log line from arbitrary code carries the run it came from.

        This is the whole point of using context variables: a model's own
        logging is attributed without the model knowing the framework exists.
        """
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        with bound_context(run_id="r-42", job_id="USDJPY", stage="build_data"):
            get_logger("rade_qnet.test").info("building")
        output = stream.getvalue()
        assert "r-42" in output
        assert "USDJPY" in output
        assert "build_data" in output

    def test_records_without_a_context_still_emit(self):
        """
        An unbound record is not dropped and does not raise.

        Logging happens during import and during teardown, when no run is
        active. A filter that required the identifiers would lose exactly the
        records emitted when something went wrong early.
        """
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        get_logger("rade_qnet.test").info("no context here")
        assert "no context here" in stream.getvalue()


class TestContextPayload:
    """The payload is how identifiers cross a process boundary."""

    def test_payload_round_trips(self):
        """
        Identifiers survive export and re-application.

        A worker process inherits nothing, so the parent exports a payload and
        the worker applies it. Without this, every log line from a parallel
        job set is unattributable.
        """
        with bound_context(run_id="r-9", job_id="GBPUSD", stage="fit"):
            payload = context_payload()
        assert current_context() == {}
        apply_context_payload(payload)
        assert current_context() == {"run_id": "r-9", "job_id": "GBPUSD", "stage": "fit"}

    def test_an_empty_payload_clears_the_context(self):
        """Applying nothing is how a worker resets between jobs."""
        apply_context_payload({"run_id": "r-1"})
        apply_context_payload({})
        assert current_context() == {}

    def test_applying_replaces_rather_than_merges(self):
        """
        A reused worker does not retain a finished job's identifier.

        A process pool reuses workers. If applying a payload merged into the
        existing context, a worker that handled ``EURUSD`` and was then given
        a run with no job identifier would keep logging ``EURUSD`` --
        attributing work to a job that had already completed.
        """
        apply_context_payload({"run_id": "r-1", "job_id": "EURUSD", "stage": "fit"})
        apply_context_payload({"run_id": "r-2"})
        assert current_context() == {"run_id": "r-2"}

    def test_payload_is_json_encodable(self):
        """
        The payload is plain strings.

        It travels through whatever a process pool uses to serialise, so it
        must not contain anything exotic.
        """
        with bound_context(run_id="r-1", job_id="EURUSD"):
            assert json.loads(json.dumps(context_payload())) == context_payload()
