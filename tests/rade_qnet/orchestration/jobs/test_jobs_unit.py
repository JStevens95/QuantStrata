"""
Tests for the unit of work a job set dispatches.

``run_job`` is the one function a job set sends to a worker, so the decisions
under test are mostly about what survives the crossing into another process
and what has to be rebuilt once it arrives.

Three of those decisions are load-bearing:

- **The payload is plain data.** No catalog handle, no open file, no live
  model. A handle is meaningless in another process, and the symptom of
  sending one is a pickling error at dispatch time -- or worse, a handle that
  pickles and then refers to nothing.
- **The seed is derived from the job identifier**, not assigned by position.
  Position would mean reordering a job set changed every model in it, and
  adding a job at the top would change all the ones below.
- **Registrations are replayed before any name is resolved.** A spawned
  worker starts with a bare interpreter; see `ARCHITECTURE.md` defect 12.
"""

from __future__ import annotations

import pickle

import pytest

from src.rade_qnet.core.lifecycle.components import engine as register_engine
from src.rade_qnet.core.lifecycle.components import model as register_model
from src.rade_qnet.core.lifecycle.errors import ComponentError
from src.rade_qnet.core.spec.jobs import parse_job_set_spec
from src.rade_qnet.orchestration.jobs.set import JobSetRunner
from src.rade_qnet.orchestration.jobs.unit import JobOutcome, JobPayload, run_job
from src.rade_qnet.testkit.fixtures import SyntheticEngine, isolated_registries

from .support import (
    ENGINE_TAG,
    MODEL_NAME,
    SyntheticSupervisedModel,
    job_set_payload,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Isolate the component registries for every test here.

    Autouse because every test registers the same engine and model, and a
    leaked registration fails whichever test happens to run next rather
    than the one at fault.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        register_engine(ENGINE_TAG)(SyntheticEngine)
        register_model(MODEL_NAME, engine=ENGINE_TAG)(SyntheticSupervisedModel)
        yield


@pytest.fixture
def payloads(tmp_path):
    """
    Provide the payloads for a two-job set, without running anything.

    Returns
    -------
    dict
        Job identifier to :class:`JobPayload`.
    """
    dataset = write_linear_dataset(tmp_path / "linear.csv")
    spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out"))
    return JobSetRunner(spec).payloads()


class TestThePayloadCrossesAProcessBoundary:
    """
    What a job carries has to be plain data.

    The constraint is not stylistic: an open handle either refuses to pickle
    or pickles into something that refers to nothing in the receiving
    process, and the second failure mode is far harder to diagnose.
    """

    def test_a_payload_pickles(self, payloads):
        """
        The whole payload round-trips, specification included.

        The specification is a pydantic model, so it pickles by value and
        arrives identical to the one validated in the parent -- which is
        what lets validation happen once, up front, for the whole set.
        """
        payload = payloads["a"]

        restored = pickle.loads(pickle.dumps(payload))

        assert restored.job_id == payload.job_id
        assert restored.spec == payload.spec

    def test_the_payload_carries_a_catalog_path_not_a_catalog(self, payloads):
        """
        A path, because a catalog holds a lock file handle.

        Each worker opens its own against the same path, which is what the
        single-writer catalog is built for. Sending the catalog itself
        would send a handle to a lock this process holds.
        """
        assert isinstance(payloads["a"].catalog_root, type(payloads["a"].output_directory))

    def test_the_payload_names_the_modules_a_worker_must_import(self, payloads):
        """
        Registration modules travel as strings.

        Resolved in the parent, so an unregistered name fails where the
        error can list the alternatives. See defect 12.
        """
        modules = payloads["a"].registration_modules

        assert SyntheticSupervisedModel.__module__ in modules
        assert all(isinstance(module, str) for module in modules)


class TestSeedsAreDerivedFromIdentifiers:
    """A job's seed comes from its name, not from where it sits in the list."""

    def test_two_jobs_get_different_seeds(self, payloads):
        """
        Otherwise every job in a set would train the same model.

        The set would still look healthy: forty runs, forty bundles, forty
        identical scores that nobody reads as a bug.
        """
        assert payloads["a"].context().seed != payloads["b"].context().seed

    def test_the_same_identifier_gives_the_same_seed(self, tmp_path):
        """
        Re-running one job alone reproduces what it produced in the set.

        This is what makes a failed job debuggable: it can be re-run on its
        own and behave identically.
        """
        dataset = write_linear_dataset(tmp_path / "linear.csv")
        spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out"))

        first = JobSetRunner(spec).payloads()["a"].context().seed
        second = JobSetRunner(spec).payloads()["a"].context().seed

        assert first == second

    def test_reordering_the_jobs_does_not_change_their_seeds(self, tmp_path):
        """
        Order-independence, which a positional seed would not give.

        With positional seeds, inserting a job at the top of a file would
        silently retrain every job below it with different weights, and the
        diff that caused it would be one added line.
        """
        dataset = write_linear_dataset(tmp_path / "linear.csv")
        forward = parse_job_set_spec(
            job_set_payload(dataset, tmp_path / "out", jobs=[{"id": "a"}, {"id": "b"}])
        )
        reversed_ = parse_job_set_spec(
            job_set_payload(dataset, tmp_path / "out", jobs=[{"id": "b"}, {"id": "a"}])
        )

        assert (
            JobSetRunner(forward).payloads()["a"].context().seed
            == JobSetRunner(reversed_).payloads()["a"].context().seed
        )


class TestRunningOneJob:
    """The function a worker actually calls."""

    def test_a_job_produces_an_outcome(self, payloads):
        """
        Metrics, a seed, a bundle location -- and never a model.

        The outcome crosses back through a pickle, and sending trained
        weights through it would make a forty-job set's return value the
        size of forty models for no reason: the weights are already on
        disk, and the record says where.
        """
        outcome = run_job(payloads["a"])

        assert isinstance(outcome, JobOutcome)
        assert outcome.job_id == "a"
        assert outcome.model_name == MODEL_NAME
        assert outcome.bundle_directory is not None

    def test_the_exact_fit_is_recovered(self, payloads):
        """
        An r-squared of one, because the data is exactly linear.

        Asserting the value rather than a tolerance is what distinguishes a
        job that trained from one that merely completed.
        """
        outcome = run_job(payloads["a"])

        assert outcome.metric("test", "r2") == pytest.approx(1.0)

    def test_the_outcome_records_the_seed_that_was_applied(self, payloads):
        """
        The derived seed, not the set's base seed.

        Recorded so a job can be reproduced from its manifest row alone,
        without re-deriving anything.
        """
        payload = payloads["a"]

        assert run_job(payload).seed == payload.context().seed

    def test_the_job_writes_beneath_its_own_directory(self, payloads):
        """
        Each job owns a directory, so two jobs cannot overwrite each other.

        The layout is decided in the parent rather than the worker, so a
        set has one place that knows where things go.
        """
        payload = payloads["a"]

        outcome = run_job(payload)

        assert outcome.bundle_directory.is_relative_to(payload.output_directory)

    def test_an_outcome_pickles(self, payloads):
        """The return trip is as constrained as the outbound one."""
        outcome = run_job(payloads["a"])

        assert pickle.loads(pickle.dumps(outcome)) == outcome


class TestWhatAJobRefuses:
    """Failures that should happen in the worker, clearly."""

    def test_an_unresolvable_model_names_the_alternatives(self, payloads):
        """
        The registry's error, listing what it does know.

        A worker that cannot resolve a name has to say so clearly, because
        the usual cause is a registration that did not travel across the
        process boundary rather than a typo in the file.
        """
        payload = payloads["a"]
        absent = payload.spec.model_copy(
            update={"model": payload.spec.model.model_copy(update={"name": "absent"})}
        )

        with pytest.raises(ComponentError, match="no model named 'absent'"):
            run_job(
                JobPayload(
                    job_id=payload.job_id,
                    spec=absent,
                    run_id=payload.run_id,
                    spec_digest=payload.spec_digest,
                    output_directory=payload.output_directory,
                )
            )
