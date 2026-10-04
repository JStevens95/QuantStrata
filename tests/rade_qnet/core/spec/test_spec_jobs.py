"""
Tests for job-set specifications.

A job set is a shared set of defaults plus a list of per-job overrides. The
decision these tests pin down is that the overrides are merged as **raw
mappings**, before validation, rather than as validated specifications.

Merging validated specs would be simpler to write and silently wrong: once a
mapping has been through the schema, a field the user never mentioned is
indistinguishable from one they set, because both are present and populated.
A job overriding a single learning rate would then carry its own copy of every
default alongside it, and those copies would overwrite the set's defaults. The
symptom is that a job set behaves as though its ``defaults`` block were half
ignored, in a way that depends on which fields happen to have schema defaults.

The other recurring theme is **where an error is raised**. A set of forty jobs
that trains thirty-nine and then discovers the fortieth was misconfigured has
wasted thirty-nine runs to find out, so every job is validated before any job
starts, and every failure is reported together.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.core.spec.jobs import (
    JobSetSpec,
    JobSpec,
    PlacementSpec,
    dump_job_set_spec,
    load_job_set_spec,
    parse_job_set_spec,
)

#: A minimal set of defaults that validates on its own. Kept deliberately
#: small: these tests are about merging and placement, and a realistic
#: specification would bury the field under test in noise.
DEFAULTS: dict[str, object] = {
    "model": {"name": "hybrid_gnn_rnn"},
    "source": {"kind": "model", "params": {"directory": "."}},
    "training": {"engine": "torch", "epochs": 3, "learning_rate": 0.01},
}


def payload(**overrides: object) -> dict[str, object]:
    """
    Build a job-set payload with two jobs.

    Parameters
    ----------
    **overrides
        Top-level keys to replace.

    Returns
    -------
    dict
        A payload suitable for :func:`parse_job_set_spec`.
    """
    base: dict[str, object] = {
        "name": "set",
        "defaults": dict(DEFAULTS),
        "jobs": [{"id": "a"}, {"id": "b"}],
    }
    base.update(overrides)
    return base


class TestMergingDefaultsWithOverrides:
    """
    What a job inherits, and what it replaces.

    The property under test throughout is that overriding one field leaves
    its siblings alone, at every depth.
    """

    def test_a_job_inherits_every_default(self):
        """A job with no overrides is the defaults."""
        spec = parse_job_set_spec(payload())

        run = spec.run_spec_for("a")

        assert run.training.epochs == 3
        assert run.training.learning_rate == 0.01

    def test_an_override_replaces_only_the_field_named(self):
        """
        Siblings of an overridden field survive.

        This is the merge behaviour that merging validated specs would
        destroy: `epochs` is not mentioned by the job, so it must still be
        the set's 3 rather than the schema's default.
        """
        spec = parse_job_set_spec(
            payload(jobs=[{"id": "a", "overrides": {"training": {"learning_rate": 0.5}}}])
        )

        run = spec.run_spec_for("a")

        assert run.training.learning_rate == 0.5
        assert run.training.epochs == 3

    def test_one_jobs_override_does_not_reach_another(self):
        """
        Jobs are merged independently from the same defaults.

        A shared mutable defaults mapping would let the first job's merge
        leak into the second, and the symptom would depend on declaration
        order -- which is close to undebuggable in a set of forty.
        """
        spec = parse_job_set_spec(
            payload(
                jobs=[
                    {"id": "a", "overrides": {"training": {"epochs": 99}}},
                    {"id": "b"},
                ]
            )
        )

        assert spec.run_spec_for("a").training.epochs == 99
        assert spec.run_spec_for("b").training.epochs == 3

    def test_the_model_shorthand_merges_rather_than_being_replaced(self):
        """
        A top-level `model:` string survives a job overriding model params.

        Stored as the bare string the user wrote, it would not deep-merge:
        a mapping over a string replaces rather than recurses, and the
        model's name would vanish. The failure reads "model.name: Field
        required" from a file whose only mention of a name is at the top
        level and plainly there.
        """
        spec = parse_job_set_spec(
            {
                "model": "hybrid_gnn_rnn",
                "defaults": {
                    "source": {"kind": "model", "params": {"directory": "."}},
                    "training": {"engine": "torch"},
                },
                "jobs": [{"id": "a", "overrides": {"model": {"params": {"units": 8}}}}],
            }
        )

        run = spec.run_spec_for("a")

        assert run.model.name == "hybrid_gnn_rnn"
        assert run.model.params["units"] == 8


class TestHowAJobIsWritten:
    """
    The two accepted spellings of a job, and the one that is refused.

    A job may nest its overrides under `overrides:` or write them at the top
    level beside `id:`. Both are common in hand-written files and both are
    unambiguous on their own.
    """

    def test_top_level_keys_are_folded_into_overrides(self):
        """The terse spelling means the same as the explicit one."""
        terse = JobSpec.model_validate({"id": "a", "training": {"epochs": 7}})
        explicit = JobSpec.model_validate({"id": "a", "overrides": {"training": {"epochs": 7}}})

        assert terse.overrides == explicit.overrides

    def test_mixing_the_two_spellings_is_refused(self):
        """
        Refused rather than merged, because the intent is unclear.

        A file using both has most likely been edited by two people with
        different habits, and guessing which wins would make the answer
        depend on an implementation detail no reader can see.
        """
        with pytest.raises((SpecError, ValueError)):
            JobSpec.model_validate(
                {"id": "a", "overrides": {"training": {"epochs": 7}}, "training": {"epochs": 9}}
            )

    def test_duplicate_job_identifiers_are_refused(self):
        """
        Identifiers name output directories and manifest rows.

        Two jobs sharing one would have the second overwrite the first's
        bundle, and the manifest would describe a run whose artifacts had
        been replaced by a different run's.
        """
        with pytest.raises((SpecError, ValueError), match=r"(?i)duplicate|unique"):
            parse_job_set_spec(payload(jobs=[{"id": "a"}, {"id": "a"}]))


class TestValidationHappensBeforeAnyJobRuns:
    """Every job is checked up front, and every failure reported together."""

    def test_all_jobs_validate_when_the_set_is_sound(self):
        """A sound set yields one run specification per job."""
        spec = parse_job_set_spec(payload())

        assert set(spec.validate_jobs()) == {"a", "b"}

    def test_every_failing_job_is_named_at_once(self):
        """
        Not just the first.

        Fixing one typo, rerunning, and finding the next is a slow loop when
        each attempt costs a validation pass over forty jobs' worth of file.
        """
        spec = parse_job_set_spec(
            payload(
                jobs=[
                    {"id": "a", "overrides": {"training": {"epochs": -1}}},
                    {"id": "b", "overrides": {"training": {"epochs": -2}}},
                ]
            )
        )

        with pytest.raises(SpecError) as caught:
            spec.validate_jobs()

        message = str(caught.value)
        assert "a:" in message
        assert "b:" in message

    def test_an_unknown_job_identifier_is_an_error(self):
        """Asking for a job that is not in the set names the mistake."""
        spec = parse_job_set_spec(payload())

        with pytest.raises((SpecError, KeyError)):
            spec.job("absent")


class TestWhereFieldsBelong:
    """
    Run-specification fields at the top level are a reasonable mistake.

    It is where they go in a single-run file. The message therefore says
    where they belong rather than reporting an unexpected key.
    """

    def test_a_misplaced_run_field_says_where_it_belongs(self):
        """`training:` at the top level is told about `defaults:`."""
        with pytest.raises(SpecError, match="defaults"):
            parse_job_set_spec(
                {"defaults": dict(DEFAULTS), "jobs": [{"id": "a"}], "training": {"epochs": 1}}
            )

    def test_model_cannot_be_given_twice(self):
        """
        The top-level shorthand and `defaults.model` are the same setting.

        Accepting both would need a precedence rule that the file does not
        show.
        """
        with pytest.raises(SpecError, match=r"(?i)both"):
            parse_job_set_spec(
                {"model": "hybrid_gnn_rnn", "defaults": dict(DEFAULTS), "jobs": [{"id": "a"}]}
            )


class TestPlacement:
    """
    Placement is declared, defaulted, and never changes results.

    Its fields are an operational choice: how many processes, how many
    threads each. See `PHASE_4_JOB_SETS.md` §8.1 for why the thread budget
    is *not* among them for reproducibility purposes.
    """

    def test_placement_defaults_to_automatic(self):
        """A set that says nothing about placement gets `auto`."""
        assert parse_job_set_spec(payload()).placement.executor == "auto"

    def test_placement_is_read_from_the_file(self):
        """Declared placement is preserved verbatim."""
        spec = parse_job_set_spec(
            payload(placement={"executor": "processes", "workers": 4, "threads_per_worker": 2})
        )

        assert spec.placement == PlacementSpec(
            executor="processes", workers=4, threads_per_worker=2
        )

    def test_an_unknown_executor_is_refused(self):
        """
        The set of executors is closed.

        A typo would otherwise fall through to whatever the policy chose,
        and the run would silently not use what was asked for.
        """
        with pytest.raises((SpecError, ValueError)):
            parse_job_set_spec(payload(placement={"executor": "dask"}))

    def test_spawn_is_accepted_on_every_platform(self):
        """
        The default start method must work everywhere.

        Stated as a test because the whole point of defaulting to ``spawn``
        is that a specification written on one platform runs on another.
        """
        spec = parse_job_set_spec(payload(placement={"start_method": "spawn"}))

        assert spec.placement.start_method == "spawn"

    @pytest.mark.skipif(sys.platform != "win32", reason="forkserver is available off Windows")
    def test_forkserver_is_refused_on_windows(self):
        """
        A POSIX-only start method fails at parse time, not mid-run.

        Without the check, ``multiprocessing.get_context('forkserver')``
        raises from inside the executor -- after the data build, with a
        message naming neither the setting nor the file it came from.
        """
        with pytest.raises((SpecError, ValueError), match="not available on Windows"):
            parse_job_set_spec(payload(placement={"start_method": "forkserver"}))

    @pytest.mark.skipif(sys.platform == "win32", reason="forkserver is genuinely absent on Windows")
    def test_forkserver_is_accepted_off_windows(self):
        """
        The check rejects only what the platform actually lacks.

        A guard that refused ``forkserver`` everywhere would be a silent
        feature removal for the platforms that have it.
        """
        spec = parse_job_set_spec(payload(placement={"start_method": "forkserver"}))

        assert spec.placement.start_method == "forkserver"


class TestRoundTripThroughYaml:
    """A set loaded from a file and dumped back describes the same run."""

    def test_load_reads_a_file(self, tmp_path):
        """`load_job_set_spec` parses what `parse_job_set_spec` would."""
        path = tmp_path / "set.yaml"
        path.write_text(yaml.safe_dump(payload()))

        assert load_job_set_spec(path).job_ids == ("a", "b")

    def test_dump_then_load_preserves_the_jobs(self, tmp_path):
        """
        The round trip is lossless for everything a run depends on.

        A job set is written into a run's output as a record of what was
        asked for, so a dump that dropped a field would make the record
        disagree with the run it describes.
        """
        original = parse_job_set_spec(
            payload(jobs=[{"id": "a", "overrides": {"training": {"epochs": 11}}}])
        )
        path = tmp_path / "set.yaml"
        dump_job_set_spec(original, path)

        reloaded = load_job_set_spec(path)

        assert reloaded.job_ids == original.job_ids
        assert reloaded.run_spec_for("a").training.epochs == 11


class TestTheSpecIsNotASharedMutable:
    """
    Nothing a caller does to a merged result can reach the set.

    A merged specification is handed to a job and travels to a worker. If it
    shared a mapping with the defaults, a job that mutated its own
    configuration would change every later job's.
    """

    def test_merged_specs_are_independent_objects(self):
        """Two jobs' merged results are not the same object."""
        spec = parse_job_set_spec(payload())

        assert spec.run_spec_for("a") is not spec.run_spec_for("b")

    def test_a_job_set_is_frozen(self):
        """
        The set itself cannot be edited after loading.

        What was loaded is what is recorded in the manifest, so the two
        cannot drift.
        """
        spec = parse_job_set_spec(payload())

        with pytest.raises((ValueError, AttributeError, TypeError)):
            spec.name = "renamed"  # type: ignore[misc]

    def test_a_set_reports_its_jobs_in_declaration_order(self):
        """
        Order is the file's, not a sort.

        Manifest rows, log lines and figures all follow it, and a user
        reading them is looking for the order they wrote.
        """
        spec = parse_job_set_spec(payload(jobs=[{"id": "z"}, {"id": "m"}, {"id": "a"}]))

        assert spec.job_ids == ("z", "m", "a")

    def test_the_jobs_tuple_is_the_declared_jobs(self):
        """Each entry is a `JobSpec`, not a raw mapping."""
        spec = parse_job_set_spec(payload())

        assert all(isinstance(job, JobSpec) for job in spec.jobs)
        assert isinstance(spec, JobSetSpec)
