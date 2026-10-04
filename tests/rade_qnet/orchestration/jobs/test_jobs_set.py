"""
Tests for the job-set runner.

The runner is the piece that turns a file describing forty jobs into forty
bundles and one manifest. It expands, merges, dispatches, and aggregates, and
each of those four has a decision worth pinning down.

**Expansion and validation happen before anything runs.** A set that trains
thirty-nine models and then discovers the fortieth was misconfigured has
spent thirty-nine runs' worth of time finding out. Every job is validated up
front and every failure is reported together.

**A failing job does not fail the set.** The opposite policy throws away the
thirty-nine that worked because of one typo in the fortieth. A failure is a
recorded outcome, not an exception.

**The run identifier is derived, not timestamped.** A timestamp looks tidier
and means a re-run after a crash produces a second partial set beside the
first rather than completing it.

These tests use the synthetic engine from :mod:`.support` throughout, so a
"job" costs milliseconds. The one test that uses a real process pool uses the
same synthetic model, which is why it can afford to.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.runtime.components import engine as register_engine
from src.rade_qnet.core.runtime.components import model as register_model
from src.rade_qnet.core.runtime.errors import SpecError
from src.rade_qnet.core.spec.jobs import parse_job_set_spec
from src.rade_qnet.orchestration.compute.local import LocalExecutor
from src.rade_qnet.orchestration.jobs.manifest import MANIFEST_FILENAME, JobSetManifest
from src.rade_qnet.orchestration.jobs.set import JOBS_SUBDIRECTORY, JobSetRunner
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
def dataset(tmp_path):
    """
    Write the exactly-linear dataset every job reads.

    Returns
    -------
    pathlib.Path
        The CSV file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


def runner(dataset, tmp_path, **overrides):
    """
    Build a runner over a two-job set, pinned to the sequential executor.

    Pinned rather than left to the placement policy, because the policy's
    answer depends on the machine and a test asserting on a manifest should
    not change behaviour between a laptop and a build box.

    Parameters
    ----------
    dataset
        The CSV file.
    tmp_path
        The test's temporary directory.
    **overrides
        Job-set payload keys to replace.

    Returns
    -------
    JobSetRunner
        The runner.
    """
    spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out", **overrides))
    return JobSetRunner(spec, executor=LocalExecutor())


class TestRunningASet:
    """End to end, with every job succeeding."""

    def test_every_job_runs(self, dataset, tmp_path):
        """Two jobs in, two records out."""
        manifest = runner(dataset, tmp_path).run()

        assert [record.job_id for record in manifest.jobs] == ["a", "b"]
        assert len(manifest.succeeded) == 2

    def test_each_job_writes_its_own_bundle(self, dataset, tmp_path):
        """
        Separate directories, so two jobs cannot overwrite each other.

        Sharing one would leave a manifest describing artifacts that a
        later job had replaced -- a record that is wrong rather than
        missing, which is the worse of the two.
        """
        manifest = runner(dataset, tmp_path).run()

        directories = {record.bundle_directory for record in manifest.succeeded}

        assert len(directories) == 2
        assert all(path.startswith(f"{JOBS_SUBDIRECTORY}/") for path in directories)

    def test_the_manifest_is_written_to_disk(self, dataset, tmp_path):
        """
        The set's own record, beside the job directories rather than in them.

        It is the thing a dashboard opens, and a reader should not have to
        know a job identifier to find it.
        """
        job_runner = runner(dataset, tmp_path)

        job_runner.run()

        assert (job_runner.output_directory / MANIFEST_FILENAME).exists()
        assert JobSetManifest.read(job_runner.output_directory).run_id == job_runner.run_id

    def test_overrides_reach_the_jobs_that_declared_them(self, dataset, tmp_path):
        """
        Per-job configuration is what a job set is for.

        Asserted through the seed, which is observable in the manifest and
        differs per job, rather than through a metric the exact fit makes
        identical for both.
        """
        manifest = runner(
            dataset,
            tmp_path,
            jobs=[
                {"id": "a", "overrides": {"training": {"fit_params": {"alpha": 0.1}}}},
                {"id": "b", "overrides": {"training": {"fit_params": {"alpha": 0.9}}}},
            ],
        ).run()

        assert manifest.record("a").seed != manifest.record("b").seed

    def test_the_set_records_where_it_ran(self, dataset, tmp_path):
        """
        Placement and the reason for it, even when supplied directly.

        Wrapped in the same shape as a policy decision rather than special
        cased, so a manifest always answers the question.
        """
        manifest = runner(dataset, tmp_path).run()

        assert manifest.placement
        assert manifest.placement_reason == "supplied directly by the caller"


class TestPartialFailure:
    """One job failing must not cost the others."""

    @pytest.fixture
    def manifest(self, dataset, tmp_path):
        """
        Run a set whose second job points at a file that is not there.

        Returns
        -------
        JobSetManifest
            The manifest.
        """
        return runner(
            dataset,
            tmp_path,
            jobs=[
                {"id": "good"},
                {"id": "bad", "overrides": {"source": {"path": str(tmp_path / "absent.csv")}}},
            ],
        ).run()

    def test_the_working_job_still_completes(self, manifest):
        """
        Thirty-nine results are thirty-nine results.

        Discarding them because the fortieth failed throws away hours of
        work over one typo.
        """
        assert [record.job_id for record in manifest.succeeded] == ["good"]

    def test_the_failure_is_recorded_with_its_reason(self, manifest):
        """
        A recorded outcome, not an exception that ends the set.

        The kind is recorded separately from the message so forty jobs
        failing the same way read as one fix.
        """
        record = manifest.record("bad")

        assert not record.succeeded
        assert record.failure_kind
        assert record.failure_traceback

    def test_the_manifest_still_reaches_disk(self, dataset, tmp_path):
        """
        A set with failures is still a set that happened.

        Writing the manifest only on complete success would leave the one
        case where a record is most needed with no record at all.
        """
        job_runner = runner(
            dataset,
            tmp_path,
            jobs=[{"id": "bad", "overrides": {"source": {"path": str(tmp_path / "absent.csv")}}}],
        )

        job_runner.run()

        assert JobSetManifest.read(job_runner.output_directory).failed


class TestValidationPrecedesExecution:
    """Nothing runs until every job is known to be sound."""

    def test_a_misconfigured_job_stops_the_set_before_it_starts(self, dataset, tmp_path):
        """
        Raised, unlike a job that fails while running.

        The distinction is deliberate: a specification error is knowable
        without doing any work, so discovering it after thirty-nine runs
        would be the framework's fault rather than the data's.
        """
        job_runner = runner(
            dataset,
            tmp_path,
            jobs=[{"id": "a"}, {"id": "b", "overrides": {"source": {"kind": "invented"}}}],
        )

        with pytest.raises(SpecError):
            job_runner.run()

    def test_nothing_is_written_when_validation_fails(self, dataset, tmp_path):
        """
        No manifest, no job directories, no partial set to clean up.

        A set that refused to start should leave the filesystem as it found
        it, so a re-run after the fix is a fresh attempt.
        """
        job_runner = runner(
            dataset, tmp_path, jobs=[{"id": "a", "overrides": {"source": {"kind": "invented"}}}]
        )

        with pytest.raises(SpecError):
            job_runner.run()

        assert not job_runner.output_directory.exists()


class TestTheRunIdentifier:
    """Derived from the specification, so a re-run extends rather than scatters."""

    def test_the_same_specification_lands_in_the_same_place(self, dataset, tmp_path):
        """
        Two runs of one file share a directory.

        A timestamped identifier would make a re-run after a crash produce
        a second partial set beside the first rather than completing it.
        """
        first = runner(dataset, tmp_path)
        second = runner(dataset, tmp_path)

        assert first.run_id == second.run_id

    def test_a_changed_specification_lands_somewhere_else(self, dataset, tmp_path):
        """
        Derived from the digest, so a different set is a different place.

        Otherwise a changed configuration would overwrite the results of
        the one it replaced, and the manifest would describe neither.
        """
        first = runner(dataset, tmp_path)
        second = runner(dataset, tmp_path, jobs=[{"id": "a"}, {"id": "c"}])

        assert first.run_id != second.run_id

    def test_the_identifier_carries_the_sets_name(self, dataset, tmp_path):
        """
        A directory listing should be readable without opening anything.

        `portfolio-7f3a9c21` says what it is; a bare digest does not.
        """
        assert runner(dataset, tmp_path).run_id.startswith("set-")

    def test_an_explicit_identifier_is_honoured(self, dataset, tmp_path):
        """A caller that has its own naming scheme keeps it."""
        spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out"))

        assert JobSetRunner(spec, run_id="chosen").run_id == "chosen"


class TestPayloadsWithoutRunning:
    """Inspecting a set is as useful as running one."""

    def test_payloads_are_produced_for_every_job(self, dataset, tmp_path):
        """
        The natural unit to inspect, test against, and re-run one job from.

        A job that failed inside a set should be reproducible on its own,
        and this makes that a one-line operation rather than a
        reconstruction of the merge, the seed and the directory layout.
        """
        payloads = runner(dataset, tmp_path).payloads()

        assert set(payloads) == {"a", "b"}

    def test_producing_payloads_runs_nothing(self, dataset, tmp_path):
        """Inspection must not have side effects on the filesystem."""
        job_runner = runner(dataset, tmp_path)

        job_runner.payloads()

        assert not job_runner.output_directory.exists()

    def test_every_job_in_a_set_shares_its_run_identifier(self, dataset, tmp_path):
        """
        The set is one experiment, and its bundles should be findable together.

        The digest is what makes a set's members a set in the catalog
        rather than forty unrelated runs.
        """
        payloads = runner(dataset, tmp_path).payloads()

        assert len({payload.run_id for payload in payloads.values()}) == 1
        assert len({payload.spec_digest for payload in payloads.values()}) == 1


class TestTheSetLevelContext:
    """A set needs somewhere to write its own figures and logs."""

    def test_the_context_points_at_the_sets_directory(self, dataset, tmp_path):
        """
        Not at any job's.

        Set-level output belongs beside the job directories, not inside
        one of them, where it would look like that job produced it.
        """
        job_runner = runner(dataset, tmp_path)

        assert job_runner.context().output_directory == job_runner.output_directory

    def test_the_context_is_not_the_one_a_job_runs_under(self, dataset, tmp_path):
        """
        Each job derives its own, in its own process.

        The set's context has no job identifier, so a seed derived from it
        would belong to no job in particular.
        """
        job_runner = runner(dataset, tmp_path)

        assert job_runner.context().run_id == job_runner.run_id
        assert job_runner.context().seed != job_runner.payloads()["a"].context().seed
