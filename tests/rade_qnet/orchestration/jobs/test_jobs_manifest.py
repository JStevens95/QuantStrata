"""
Tests for the set-level manifest.

Each bundle is recorded in the catalog on its own. This is the record of the
*set*: which jobs there were, which succeeded, and for the ones that did not,
why.

Two properties dominate what is worth testing here.

**It is read while it is being written.** A forty-job set takes hours, and a
dashboard or a watching script will open this file at some point during them.
A half-written manifest must read as the previous version rather than as a
parse error, which is what the write-and-rename does -- and the temporary
file has to be in the same directory, because a rename is atomic only within
a filesystem.

**It has to survive being moved.** A manifest is archived, copied to another
machine, and read months later. Bundle paths are therefore recorded relative
to the set's directory: an absolute path is correct exactly once, on the
machine that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rade_qnet.orchestration.compute.base import WorkFailure, WorkResult
from src.rade_qnet.orchestration.jobs.manifest import (
    MANIFEST_FILENAME,
    JobRecord,
    JobSetManifest,
)
from src.rade_qnet.orchestration.jobs.unit import JobOutcome


def outcome(job_id: str = "a", **overrides: object) -> JobOutcome:
    """
    Build a job outcome.

    Parameters
    ----------
    job_id
        The job's identifier.
    **overrides
        Fields to replace.

    Returns
    -------
    JobOutcome
        The outcome.
    """
    fields: dict[str, object] = {
        "job_id": job_id,
        "metrics": {"test": {"r2": 0.5, "mse": 1.25}},
        "seed": 7,
        "epochs": 3,
        "stopped_early": False,
        "bundle_directory": Path("/sets/run/jobs") / job_id / "bundle",
        "bundle_version": 1,
        "model_name": "synthetic_tabular",
    }
    fields.update(overrides)
    return JobOutcome(**fields)  # type: ignore[arg-type]


def success(job_id: str = "a", **overrides: object) -> WorkResult[JobOutcome]:
    """
    Build a successful executor result.

    Parameters
    ----------
    job_id
        The job's identifier.
    **overrides
        Outcome fields to replace.

    Returns
    -------
    WorkResult
        The result.
    """
    return WorkResult(key=job_id, value=outcome(job_id, **overrides), wall_seconds=2.0)


def failure(job_id: str = "b") -> WorkResult[JobOutcome]:
    """
    Build a failed executor result.

    Parameters
    ----------
    job_id
        The job's identifier.

    Returns
    -------
    WorkResult
        The result.
    """
    return WorkResult(
        key=job_id,
        failure=WorkFailure(kind="StageError", message="boom", traceback_text="Traceback..."),
        wall_seconds=0.5,
    )


#: The root every bundle path in these tests sits beneath.
ROOT = Path("/sets/run")


class TestBuildingARecordFromAResult:
    """What one executor result becomes."""

    def test_a_success_carries_its_numbers(self):
        """Metrics, seed, epochs and the bundle's version all survive."""
        record = JobRecord.of(success(), relative_to=ROOT)

        assert record.succeeded
        assert record.metric("test", "r2") == 0.5
        assert record.seed == 7
        assert record.epochs == 3
        assert record.bundle_version == 1

    def test_a_failure_carries_its_reason(self):
        """
        Kind, message and traceback, captured in whichever process failed.

        The kind is separate from the message so a set that failed the same
        way forty times is visible as one fix rather than forty.
        """
        record = JobRecord.of(failure(), relative_to=ROOT)

        assert not record.succeeded
        assert record.failure_kind == "StageError"
        assert record.failure_message == "boom"
        assert record.failure_traceback

    def test_a_failure_still_records_how_long_it_took(self):
        """
        "It failed after four hours" and "it failed immediately" differ.

        The two call for completely different responses, and a record that
        dropped the duration on failure would lose the distinction exactly
        when it matters most.
        """
        assert JobRecord.of(failure(), relative_to=ROOT).wall_seconds == 0.5

    def test_a_bundle_path_is_recorded_relative_to_the_set(self):
        """
        So the record survives the directory being copied or archived.

        An absolute path is correct on exactly one machine.
        """
        record = JobRecord.of(success(), relative_to=ROOT)

        assert record.bundle_directory == "jobs/a/bundle"

    def test_a_bundle_outside_the_set_keeps_its_absolute_path(self):
        """
        Unusual but legal, and recording it beats failing to record it.

        A job given its own `output_root` writes outside the set, and the
        manifest should still say where.
        """
        elsewhere = Path("/somewhere/else/bundle")

        record = JobRecord.of(success(bundle_directory=elsewhere), relative_to=ROOT)

        assert record.bundle_directory == str(elsewhere)

    def test_a_missing_metric_reads_as_none(self):
        """
        Not zero.

        A zero would plot, and would be indistinguishable from a genuinely
        zero score.
        """
        record = JobRecord.of(success(), relative_to=ROOT)

        assert record.metric("test", "absent") is None
        assert record.metric("absent", "r2") is None


class TestWhatASetReports:
    """The summaries a manifest exists to serve."""

    @pytest.fixture
    def manifest(self):
        """
        Provide a manifest with one success and one failure.

        Returns
        -------
        JobSetManifest
            The manifest.
        """
        return JobSetManifest(
            run_id="run",
            spec_digest="digest",
            wall_seconds=12.0,
            jobs=(
                JobRecord.of(success("a"), relative_to=ROOT),
                JobRecord.of(failure("b"), relative_to=ROOT),
            ),
        )

    def test_successes_and_failures_are_separable(self, manifest):
        """The two partitions are what every summary starts from."""
        assert [record.job_id for record in manifest.succeeded] == ["a"]
        assert [record.job_id for record in manifest.failed] == ["b"]

    def test_a_record_is_retrievable_by_identifier(self, manifest):
        """Looking one job up should not need a scan at the call site."""
        assert manifest.record("a").job_id == "a"
        assert manifest.record("absent") is None

    def test_a_metric_can_be_gathered_across_jobs(self, manifest):
        """
        The shape every ranking and figure needs.

        Jobs without the metric are omitted rather than given a
        placeholder, for the same reason a missing metric reads as `None`.
        """
        assert manifest.metric_by_job("test", "r2") == {"a": 0.5}

    def test_the_summary_counts_what_succeeded(self, manifest):
        """One line, for a log and a terminal."""
        assert manifest.summary() == "1 of 2 job(s) succeeded in 12.0s"

    def test_partial_failure_is_a_recorded_outcome_not_an_exception(self, manifest):
        """
        A set with a failing job still produces a manifest.

        Thirty-nine jobs that trained are thirty-nine results, and
        discarding them because the fortieth failed would throw away hours
        of work over one typo.
        """
        assert len(manifest.jobs) == 2
        assert len(manifest.succeeded) == 1


class TestWritingAndReading:
    """The file itself."""

    @pytest.fixture
    def manifest(self):
        """
        Provide a one-job manifest.

        Returns
        -------
        JobSetManifest
            The manifest.
        """
        return JobSetManifest(
            run_id="run",
            spec_digest="digest",
            name="set",
            placement="local (sequential, in-process)",
            placement_reason="one job",
            jobs=(JobRecord.of(success("a"), relative_to=ROOT),),
        )

    def test_a_manifest_round_trips(self, manifest, tmp_path):
        """What is written is what is read."""
        manifest.write(tmp_path)

        assert JobSetManifest.read(tmp_path) == manifest

    def test_the_directory_is_created(self, manifest, tmp_path):
        """A set writes its manifest before anyone has made its directory."""
        directory = tmp_path / "not" / "yet"

        assert manifest.write(directory) == directory / MANIFEST_FILENAME

    def test_the_temporary_file_does_not_survive(self, manifest, tmp_path):
        """
        A leftover temporary file is evidence of a crash mid-write.

        It is deliberately visible in a directory listing for that reason,
        so a successful write must not leave one behind.
        """
        manifest.write(tmp_path)

        assert [path.name for path in tmp_path.iterdir()] == [MANIFEST_FILENAME]

    def test_a_rewrite_replaces_rather_than_appends(self, manifest, tmp_path):
        """
        Re-running a set overwrites its manifest in place.

        The run identifier is derived from the specification, so the same
        file lands in the same place -- which is only safe if the second
        write is a replacement.
        """
        manifest.write(tmp_path)
        manifest.model_copy(update={"wall_seconds": 99.0}).write(tmp_path)

        assert JobSetManifest.read(tmp_path).wall_seconds == 99.0

    def test_the_file_is_readable_json(self, manifest, tmp_path):
        """
        Indented JSON, because a human reads this during a long run.

        A dashboard is not the only consumer; the first thing anyone does
        with a stalled set is open its manifest in an editor.
        """
        path = manifest.write(tmp_path)

        assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run"

    def test_placement_and_its_reason_are_both_recorded(self, manifest, tmp_path):
        """
        A set that chose its placement and one that was told differ.

        Six months later this is the only thing that remembers which it
        was, and the two are not the same experiment.
        """
        manifest.write(tmp_path)

        reloaded = JobSetManifest.read(tmp_path)

        assert reloaded.placement.startswith("local")
        assert reloaded.placement_reason == "one job"

    def test_an_unknown_field_is_refused_on_read(self, tmp_path):
        """
        A manifest from a newer version fails loudly here.

        Silently dropping a field somebody downstream relies on is the
        worse outcome: the data is simply absent, with nothing to say so.
        """
        path = tmp_path / MANIFEST_FILENAME
        path.write_text(json.dumps({"run_id": "r", "spec_digest": "d", "invented": 1}))

        with pytest.raises(Exception, match=r"(?i)extra|invented"):
            JobSetManifest.read(tmp_path)

    def test_reading_a_missing_manifest_says_so(self, tmp_path):
        """A set that never finished has no manifest, and that is the answer."""
        with pytest.raises(FileNotFoundError):
            JobSetManifest.read(tmp_path)

    def test_the_framework_version_is_recorded(self, manifest, tmp_path):
        """
        Which version produced the record.

        A manifest outlives the code that wrote it, and "which version was
        this" is the first question asked of a result that looks wrong.
        """
        manifest.write(tmp_path)

        assert JobSetManifest.read(tmp_path).framework_version
