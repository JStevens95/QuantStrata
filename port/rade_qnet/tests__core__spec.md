# `tests/rade_qnet/core/spec`

8 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 27 | 1075 | `4cf0eee3c800d080` |
| 2 | `test_spec_data.py` | 277 | 10239 | `cf59fa4befee87a3` |
| 3 | `test_spec_hardware.py` | 153 | 5495 | `7beadfe09ad37b73` |
| 4 | `test_spec_jobs.py` | 404 | 14709 | `73e15a3dde448b19` |
| 5 | `test_spec_merge.py` | 316 | 12557 | `4f2d2383798d852e` |
| 6 | `test_spec_reports.py` | 134 | 4694 | `6abb6cbdee9201df` |
| 7 | `test_spec_run.py` | 348 | 12919 | `950ffd0f989e98d5` |
| 8 | `test_spec_training.py` | 249 | 9181 | `0f30bc07b387cda5` |

---

## 1. `tests/rade_qnet/core/spec/__init__.py`

1075 bytes · SHA-256 `4cf0eee3c800d080`

```python
"""
Tests for ``rade_qnet.core.spec`` -- configuration schemas.

The properties under test are the ones users depend on without knowing it: an
unknown key is rejected rather than ignored, an invalid combination fails at
load time rather than three hours into a run, and a spec round-trips through
``model_dump`` and back to an equal object -- which is what makes a persisted
spec a faithful record of how a model was produced.

Planned modules
---------------
``test_spec_run.py``
    ``RunSpec`` validation and discrimination on ``task``.  [Phase 1]
``test_spec_data.py``
    Source, split and loader specs.  [Phase 1]
``test_spec_training.py``
    Engine-discriminated training specs.  [Phase 1]
``test_spec_hardware.py``
    Device, precision and determinism resolution.  [Phase 1]
``test_spec_reports.py``
    Report selection.  [Phase 1]
``test_spec_jobs.py``
    Job and job-set specs.  [Phase 4]
``test_spec_merge.py``
    Deep merge of defaults with per-job overrides, including the cases where a
    nested override must not discard sibling defaults.  [Phase 4]
"""
```

---

## 2. `tests/rade_qnet/core/spec/test_spec_data.py`

10239 bytes · SHA-256 `cf59fa4befee87a3`

```python
"""
Tests for the data specification.

Two things here are worth more attention than the rest.

``SplitSpec`` is a **discriminated union** on ``kind``. Without the
discriminator, pydantic tries each member in turn and reports a failure against
whichever happened to get furthest -- so a typo in a chronological split is
reported as a failure of the grouped split, which sends the reader somewhere
irrelevant.

``ReductionSpec.fit_on`` defaults to ``train``, and the ``all`` setting is the
flag for the ninth diagnosed defect: basis selection fitted across every
scenario leaks the held-out period's structure into training.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from src.rade_qnet.core.spec.data import (
    CacheSpec,
    ChronologicalSplitSpec,
    ExplicitSplitSpec,
    GroupedSplitSpec,
    LoaderSpec,
    PurgedKFoldSplitSpec,
    ReductionSpec,
    ScalingSpec,
    SequenceSpec,
    SourceSpec,
    SplitSpec,
    TabularSourceSpec,
    TransformsSpec,
)

_SPLIT_ADAPTER = TypeAdapter(SplitSpec)
_SOURCE_ADAPTER = TypeAdapter(SourceSpec)


class TestChronologicalSplit:
    """The default split, and the one a time-series problem needs."""

    def test_it_constructs_with_defaults(self):
        """Reached through a ``default_factory``, so it must be bare."""
        assert ChronologicalSplitSpec() is not None

    def test_fractions_summing_to_one_or_more_are_rejected(self):
        """
        There must be something left to train on.

        Fractions summing to one leave an empty training split, which trains
        without error and learns nothing.
        """
        with pytest.raises(ValidationError, match="sum"):
            ChronologicalSplitSpec(validation_fraction=0.5, test_fraction=0.5)

    def test_fractions_summing_to_less_than_one_are_accepted(self):
        """The normal case."""
        assert ChronologicalSplitSpec(validation_fraction=0.15, test_fraction=0.15) is not None

    def test_a_negative_fraction_is_rejected(self):
        """A negative held-out fraction is meaningless."""
        with pytest.raises(ValidationError):
            ChronologicalSplitSpec(validation_fraction=-0.1)

    def test_a_gap_is_accepted(self):
        """
        A gap between splits is how window straddling is prevented.

        With a sequence length above one, a window ending just after the split
        boundary would contain training scenarios -- so the boundary needs
        clearance.
        """
        assert ChronologicalSplitSpec(gap_scenarios=20).gap_scenarios == 20


class TestExplicitSplit:
    """Caller-supplied indices, validated for disjointness."""

    def test_indices_are_required(self):
        """
        An explicit split with no indices is not a split.

        There is no sensible default here: the whole point is that the caller
        is supplying them.
        """
        with pytest.raises(ValidationError):
            ExplicitSplitSpec()

    def test_overlapping_indices_are_rejected(self):
        """
        The most expensive error available in this file.

        An overlapping split produces an encouraging validation score and a
        model that fails in production, with nothing in between to warn you.
        """
        with pytest.raises(ValidationError, match="split"):
            ExplicitSplitSpec(train=[0, 1, 2], validation=[2, 3], test=[4, 5])

    def test_disjoint_indices_are_accepted(self):
        """The legitimate form."""
        spec = ExplicitSplitSpec(train=[0, 1, 2], validation=[3, 4], test=[5, 6])
        assert spec.train == (0, 1, 2)


class TestSplitUnion:
    """The union discriminates on ``kind``."""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("chronological", ChronologicalSplitSpec),
            ("purged_kfold", PurgedKFoldSplitSpec),
        ],
    )
    def test_the_kind_selects_the_member(self, kind, expected):
        """A spec resolves to the type its ``kind`` names."""
        assert isinstance(_SPLIT_ADAPTER.validate_python({"kind": kind}), expected)

    def test_a_grouped_split_requires_its_key(self):
        """
        Grouping by nothing is not grouping.

        Required rather than defaulted, because any default would silently
        group by the wrong column.
        """
        with pytest.raises(ValidationError):
            _SPLIT_ADAPTER.validate_python({"kind": "grouped"})

    def test_a_grouped_split_with_a_key_is_accepted(self):
        """The legitimate form."""
        spec = _SPLIT_ADAPTER.validate_python({"kind": "grouped", "group_key": "instrument"})
        assert isinstance(spec, GroupedSplitSpec)

    def test_an_unknown_kind_is_rejected(self):
        """A misspelled kind fails at load."""
        with pytest.raises(ValidationError):
            _SPLIT_ADAPTER.validate_python({"kind": "chronologica"})

    def test_an_error_is_reported_against_the_named_member(self):
        """
        The discriminator's real payoff.

        Without it, a bad chronological fraction is reported as a failure of
        every union member at once, and the reader has to work out which one
        was meant.
        """
        with pytest.raises(ValidationError) as caught:
            _SPLIT_ADAPTER.validate_python(
                {"kind": "chronological", "validation_fraction": 0.9, "test_fraction": 0.9}
            )
        assert "chronological" in str(caught.value)


class TestLoaderSpec:
    """Batch construction, kept separate from what a split is."""

    def test_it_constructs_with_defaults(self):
        """Reached through a ``default_factory``."""
        assert LoaderSpec() is not None

    def test_persistent_workers_without_workers_is_rejected(self):
        """
        Persisting zero workers is a contradiction.

        Silently ignored, it reads as a performance setting that does nothing
        -- so a user tunes it and sees no effect, repeatedly.
        """
        with pytest.raises(ValidationError, match="persistent_workers"):
            LoaderSpec(num_workers=0, persistent_workers=True)

    def test_persistent_workers_with_workers_is_accepted(self):
        """The legitimate form."""
        assert LoaderSpec(num_workers=4, persistent_workers=True) is not None

    def test_a_zero_batch_size_is_rejected(self):
        """A batch of nothing produces no gradient."""
        with pytest.raises(ValidationError):
            LoaderSpec(batch_size=0)

    def test_shuffle_belongs_to_the_loader(self):
        """
        Batch order is a loader concern, not a split concern.

        The third diagnosed defect was one ``shuffle`` flag driving both the
        split and the batch order, so turning off shuffling to get a
        chronological split also disabled shuffling within an epoch.
        """
        assert "shuffle" in LoaderSpec.model_fields


class TestTransforms:
    """Transform settings, including the leakage flag."""

    def test_transforms_construct_with_defaults(self):
        """Every nested transform spec must be bare-constructible."""
        assert TransformsSpec() is not None

    @pytest.mark.parametrize("spec_type", [ScalingSpec, ReductionSpec, SequenceSpec, CacheSpec])
    def test_each_transform_constructs_with_defaults(self, spec_type):
        """The refined rule: anything defaulted elsewhere must be bare."""
        assert spec_type() is not None

    def test_reduction_fits_on_training_data_by_default(self):
        """
        The safe default for the ninth diagnosed defect.

        Basis selection fitted across every scenario leaks the held-out
        period's structure into training. Defaulting to ``train`` means the
        leak has to be opted into.
        """
        assert ReductionSpec().fit_on == "train"

    def test_fitting_on_everything_must_be_requested_explicitly(self):
        """
        The unsafe setting exists, but is named.

        There are legitimate uses -- a final refit on all data with no
        held-out claim -- so it is available rather than removed, and visible
        in the spec recorded in the bundle.
        """
        assert ReductionSpec(fit_on="all").fit_on == "all"

    def test_an_unknown_fit_target_is_rejected(self):
        """Only the two defined behaviours exist."""
        with pytest.raises(ValidationError):
            ReductionSpec(fit_on="validation")

    def test_a_sequence_length_below_one_is_rejected(self):
        """A window of zero scenarios carries no history."""
        with pytest.raises(ValidationError):
            SequenceSpec(length=0)


class TestSourceUnion:
    """The source union discriminates on ``kind`` as well."""

    def test_a_tabular_source_resolves(self):
        """The simple case, requiring no model-specific code."""
        spec = _SOURCE_ADAPTER.validate_python({"kind": "tabular", "path": "data.csv"})
        assert isinstance(spec, TabularSourceSpec)

    def test_a_model_source_resolves(self):
        """The case where the model builds its own data."""
        spec = _SOURCE_ADAPTER.validate_python({"kind": "model"})
        assert spec.kind == "model"

    def test_an_unknown_source_kind_is_rejected(self):
        """A typo fails at load."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python({"kind": "tabula"})


class TestRoundTrip:
    """Every spec in this module survives serialisation exactly."""

    def test_a_source_spec_round_trips(self):
        """
        The first diagnosed defect, checked on the most nested spec.

        A source spec carries splits, loader settings and transforms, so it
        is the deepest round trip in the framework.
        """
        spec = _SOURCE_ADAPTER.validate_python(
            {
                "kind": "tabular",
                "path": "data.csv",
                "split": {"kind": "chronological", "validation_fraction": 0.2},
                "loader": {"batch_size": 64, "num_workers": 2, "persistent_workers": True},
                "transforms": {"reduction": {"fit_on": "all"}, "sequence": {"length": 20}},
            }
        )
        assert (
            _SOURCE_ADAPTER.validate_python(_SOURCE_ADAPTER.dump_python(spec, mode="json")) == spec
        )
```

---

## 3. `tests/rade_qnet/core/spec/test_spec_hardware.py`

5495 bytes · SHA-256 `7beadfe09ad37b73`

```python
"""
Tests for the hardware specification.

The design under test is that ``determinism`` is three-valued rather than a
boolean. Forcing deterministic kernels costs performance, and some operations
have no deterministic implementation at all, so "on or off" cannot express the
real choice. ``warn`` is the setting that says "prefer determinism, tell me
where it was not available" -- which is the honest default for research work.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.spec.hardware import HardwareSpec


class TestDefaults:
    """The spec is constructible bare, because other specs default it."""

    def test_it_constructs_with_no_arguments(self):
        """
        A nested spec reached through a ``default_factory`` must be bare.

        If this raised, a ``RunSpec`` could not be built without naming every
        hardware field -- the second of the ten diagnosed defects.
        """
        assert HardwareSpec() is not None

    def test_the_defaults_are_the_portable_choices(self):
        """
        Out of the box, a spec runs anywhere.

        Automatic device selection, full precision, no compilation, no
        distribution: nothing that requires particular hardware to be present.
        """
        spec = HardwareSpec()
        assert spec.device == "auto"
        assert spec.precision == "fp32"
        assert spec.compile_model is False
        assert spec.distributed == "none"
        assert spec.determinism == "off"


class TestDeterminism:
    """Determinism is a three-valued choice, not a flag."""

    @pytest.mark.parametrize("level", ["off", "warn", "strict"])
    def test_every_level_is_accepted(self, level):
        """All three are valid settings."""
        assert HardwareSpec(determinism=level).determinism == level

    def test_a_boolean_is_rejected(self):
        """
        ``determinism: true`` is not a valid setting.

        The rejection is the point: a boolean cannot distinguish "try, and
        warn me" from "fail if you cannot", and conflating them is how a run
        ends up claiming determinism it does not have.
        """
        with pytest.raises(ValidationError):
            HardwareSpec(determinism=True)

    def test_an_unknown_level_is_rejected(self):
        """A misspelled level fails at load, not at the first kernel."""
        with pytest.raises(ValidationError):
            HardwareSpec(determinism="deterministic")


class TestImpossibleCombinations:
    """Combinations that cannot work are rejected at load time."""

    def test_half_precision_on_cpu_is_rejected(self):
        """
        fp16 on a CPU is not merely slow.

        Most CPU kernels have no fp16 implementation, so the run fails part
        way through -- after the data build. Rejecting it at load turns hours
        into milliseconds.
        """
        with pytest.raises(ValidationError, match="fp16"):
            HardwareSpec(device="cpu", precision="fp16")

    def test_a_device_index_without_a_device_is_rejected(self):
        """
        Asking for device 3 of an unspecified device is meaningless.

        Silently ignoring the index would be worse: a user who asked for a
        particular GPU would get an arbitrary one and no warning.
        """
        with pytest.raises(ValidationError, match="device_index"):
            HardwareSpec(device="auto", device_index=3)

    def test_a_device_index_with_a_named_device_is_accepted(self):
        """The legitimate form of the same request."""
        assert HardwareSpec(device="cuda", device_index=3).device_index == 3

    def test_bfloat16_on_cpu_is_accepted(self):
        """
        bf16 has CPU support, unlike fp16.

        Rejecting both because they are "half precision" would block a
        configuration that works.
        """
        assert HardwareSpec(device="cpu", precision="bf16").precision == "bf16"


class TestStrictness:
    """Typos fail at load, and a spec cannot be mutated afterwards."""

    def test_an_unknown_key_is_rejected(self):
        """
        ``extra="forbid"`` in action.

        A misspelled key that was ignored would mean the run trains with a
        default, reports plausible numbers, and nobody finds out.
        """
        with pytest.raises(ValidationError):
            HardwareSpec(devise="cuda")

    def test_the_spec_is_frozen(self):
        """A stage must not be able to change the hardware mid-run."""
        spec = HardwareSpec()
        with pytest.raises(ValidationError):
            spec.device = "cuda"


class TestRoundTrip:
    """A spec survives serialisation exactly."""

    def test_dump_and_reload_is_exact(self):
        """
        The first of the ten diagnosed defects.

        A spec that does not round-trip cannot be recorded in a bundle,
        because the recorded configuration would not reproduce the run.
        """
        spec = HardwareSpec(
            device="cuda",
            device_index=1,
            precision="bf16",
            compile_model=True,
            distributed="ddp",
            determinism="strict",
            threads_per_worker=4,
        )
        assert HardwareSpec.model_validate(spec.model_dump()) == spec

    def test_json_round_trip_is_exact(self):
        """The form a bundle actually stores."""
        spec = HardwareSpec(device="mps", precision="fp32", determinism="warn")
        assert HardwareSpec.model_validate_json(spec.model_dump_json()) == spec
```

---

## 4. `tests/rade_qnet/core/spec/test_spec_jobs.py`

14709 bytes · SHA-256 `73e15a3dde448b19`

```python
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
```

---

## 5. `tests/rade_qnet/core/spec/test_spec_merge.py`

12557 bytes · SHA-256 `4f2d2383798d852e`

```python
"""
Tests for the defaults-and-overrides merge.

"Deep merge" means at least three different things in common usage, and the
one this framework needs is the one whose failure is silent: an override of a
nested field that discards its siblings produces a job configured plausibly
and wrongly, with no error and no symptom beyond a model that underperforms.

So every rule in `merge.py`'s table gets a test, and the nesting rule gets one
per depth rather than one in general. A merge that is correct at one level
and wrong at three is a real and common implementation, because the recursion
is usually the part that is written last.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.spec.merge import deep_merge, merge_all


class TestSiblingsSurvive:
    """The rule the whole module exists for, at each nesting depth."""

    def test_at_one_level(self):
        """An override of one key leaves the others alone."""
        merged = deep_merge({"units": 256, "layers": 3}, {"layers": 1})
        assert merged == {"units": 256, "layers": 1}

    def test_at_two_levels(self):
        """
        The depth at which a naive implementation starts to fail.

        A merge written as a single `dict.update` is correct at one level and
        wrong here: it replaces the whole `model` mapping, taking `units`
        with it.
        """
        merged = deep_merge(
            {"model": {"units": 256, "layers": 3}},
            {"model": {"layers": 1}},
        )
        assert merged == {"model": {"units": 256, "layers": 1}}

    def test_at_three_levels(self):
        """
        Because a recursion that only recurses once is a real implementation.

        The specification tree genuinely goes this deep --
        `source.transforms.sequence.length` is four -- so this is the shape
        of a real override rather than a contrived one.
        """
        merged = deep_merge(
            {"source": {"transforms": {"sequence": {"length": 20, "stride": 1}}}},
            {"source": {"transforms": {"sequence": {"length": 5}}}},
        )
        assert merged == {"source": {"transforms": {"sequence": {"length": 5, "stride": 1}}}}

    def test_siblings_survive_at_every_level_at_once(self):
        """
        One override, three levels, a sibling at each.

        The three tests above could all pass against an implementation that
        recursed correctly but dropped keys on the way back up. This one
        cannot.
        """
        merged = deep_merge(
            {
                "seed": 7,
                "source": {
                    "kind": "model",
                    "transforms": {"sequence": {"length": 20}, "scaling": {"method": "standard"}},
                },
            },
            {"source": {"transforms": {"sequence": {"length": 5}}}},
        )
        assert merged == {
            "seed": 7,
            "source": {
                "kind": "model",
                "transforms": {"sequence": {"length": 5}, "scaling": {"method": "standard"}},
            },
        }


class TestWhatReplaces:
    """Everything that is not two mappings."""

    def test_a_scalar_replaces(self):
        """There is nothing to merge."""
        assert deep_merge({"seed": 0}, {"seed": 7}) == {"seed": 7}

    def test_a_sequence_replaces_rather_than_concatenating(self):
        """
        The rule most often assumed to be the other way round.

        There is no identity to merge elements on, and the case that matters
        argues the same way: a job asking for `reports: [summary]` means
        those reports, not those plus whatever the defaults wanted.
        """
        merged = deep_merge({"reports": ["summary", "curves"]}, {"reports": ["summary"]})
        assert merged == {"reports": ["summary"]}

    def test_an_empty_sequence_replaces_too(self):
        """
        Which is how a job switches every report off.

        An implementation that treated an empty override as "no opinion"
        would make that impossible to express.
        """
        assert deep_merge({"reports": ["summary"]}, {"reports": []}) == {"reports": []}

    def test_an_explicit_none_overrides(self):
        """
        Because `None` is a value, not an absence.

        Several settings use `None` to mean "let the framework decide" --
        `device_index`, `threads_per_worker`, `workers`. A job resetting one
        to `None` is asking for that behaviour, and a merge that read `None`
        as "no opinion" would leave no way to ask.
        """
        assert deep_merge({"workers": 8}, {"workers": None}) == {"workers": None}

    def test_a_mapping_replaces_a_scalar(self):
        """
        Rather than being merged into it somehow.

        Guessing a resolution here would hide a configuration mistake; the
        merged value is the one the user most recently wrote, and validation
        downstream is what reports it.
        """
        assert deep_merge({"model": "ridge"}, {"model": {"name": "ridge"}}) == {
            "model": {"name": "ridge"}
        }

    def test_a_scalar_replaces_a_mapping(self):
        """The same rule, from the other side."""
        assert deep_merge({"model": {"name": "ridge"}}, {"model": "ridge"}) == {"model": "ridge"}


class TestKeySets:
    """Which keys end up in the result."""

    def test_a_key_only_in_the_override_is_added(self):
        """A job may configure something the defaults never mentioned."""
        assert deep_merge({"seed": 0}, {"name": "run"}) == {"seed": 0, "name": "run"}

    def test_a_key_only_in_the_base_is_kept(self):
        """Which is the point of having defaults at all."""
        assert deep_merge({"seed": 0, "name": "run"}, {"seed": 7}) == {"seed": 7, "name": "run"}

    def test_an_empty_override_changes_nothing(self):
        """A job with no overrides is the shared configuration exactly."""
        base = {"model": {"units": 256}, "seed": 0}
        assert deep_merge(base, {}) == base

    def test_an_empty_base_yields_the_override(self):
        """So a job set with no defaults still works."""
        assert deep_merge({}, {"seed": 7}) == {"seed": 7}


class TestIsolation:
    """
    The result shares no mutable container with either input.

    This matters more here than it usually would. The base is a job set's
    shared defaults and is merged once per job, so a result that aliased a
    nested mapping would let one job's later mutation reach every other job's
    configuration -- a bug that appears only with more than one job and only
    sometimes.
    """

    def test_the_base_is_not_modified(self):
        """Not even at depth."""
        base = {"model": {"units": 256}}
        deep_merge(base, {"model": {"units": 64}})
        assert base == {"model": {"units": 256}}

    def test_the_override_is_not_modified(self):
        """The same, from the other side."""
        override = {"model": {"units": 64}}
        deep_merge({"model": {"units": 256}}, override)
        assert override == {"model": {"units": 64}}

    def test_a_nested_mapping_is_not_shared_with_the_base(self):
        """
        An untouched branch is copied, not aliased.

        The subtle case: `source` is not mentioned by the override, so a
        shallow copy of the base would carry the base's own dictionary
        straight into the result.
        """
        base = {"source": {"kind": "model"}}
        merged = deep_merge(base, {"seed": 7})
        merged["source"]["kind"] = "tabular"
        assert base["source"]["kind"] == "model"

    def test_a_nested_sequence_is_not_shared_with_the_override(self):
        """The same property for the container type that replaces."""
        override = {"reports": ["summary"]}
        merged = deep_merge({}, override)
        merged["reports"].append("curves")
        assert override["reports"] == ["summary"]

    def test_a_string_survives_as_a_string(self):
        """
        A string is not treated as a sequence of characters.

        It is one, and copying it element-wise would turn a model name into
        a list of single-character strings.
        """
        assert deep_merge({}, {"model": "ridge"}) == {"model": "ridge"}


class TestMergeAll:
    """Folding several layers, so precedence reads in argument order."""

    def test_the_rightmost_layer_wins(self):
        """Framework defaults, then set defaults, then the job."""
        assert merge_all({"a": 1, "b": 1}, {"b": 2, "c": 2}, {"c": 3}) == {"a": 1, "b": 2, "c": 3}

    def test_nesting_is_respected_across_layers(self):
        """
        Each layer merges deeply, not only the last.

        A fold implemented with `dict.update` would pass the test above and
        fail this one.
        """
        merged = merge_all(
            {"model": {"units": 256, "layers": 3, "dropout": 0.1}},
            {"model": {"layers": 1}},
            {"model": {"dropout": 0.0}},
        )
        assert merged == {"model": {"units": 256, "layers": 1, "dropout": 0.0}}

    def test_no_layers_is_an_empty_mapping(self):
        """Rather than an error, so a caller need not special-case it."""
        assert merge_all() == {}

    def test_one_layer_is_a_copy_of_it(self):
        """Equal, and not the same object."""
        layer = {"model": {"units": 256}}
        merged = merge_all(layer)
        assert merged == layer
        assert merged["model"] is not layer["model"]


class TestAgainstRealSpecFragments:
    """
    The case the module was written for, end to end.

    Asserted as a merge rather than through validation, because this module
    may not import the run spec -- but the shapes are the real ones, so a
    change to the schema that broke this would be visible here.
    """

    @pytest.fixture
    def defaults(self):
        """Return a job set's shared defaults, in the real schema's shape."""
        return {
            "model": {"name": "hybrid_gnn_rnn", "units": 256, "gnn_layers": 3},
            "source": {
                "kind": "model",
                "transforms": {
                    "sequence": {"length": 20},
                    "reduction": {"method": "basis_selection", "fit_on": "train"},
                },
            },
            "training": {"engine": "torch", "epochs": 200},
            "reports": {"enabled": ["summary", "curves"]},
        }

    def test_a_sparse_cluster_gets_a_smaller_model_and_keeps_everything_else(self, defaults):
        """
        The motivating case from the architecture document.

        A cluster with thin history is given a narrower, shallower network.
        Every other decision -- the window, the basis policy, the epoch
        budget, the reports -- must come through untouched, because the user
        said nothing about them.
        """
        merged = deep_merge(defaults, {"model": {"units": 64, "gnn_layers": 1}})

        assert merged["model"] == {"name": "hybrid_gnn_rnn", "units": 64, "gnn_layers": 1}
        assert merged["source"]["transforms"]["sequence"]["length"] == 20
        assert merged["source"]["transforms"]["reduction"]["fit_on"] == "train"
        assert merged["training"] == {"engine": "torch", "epochs": 200}

    def test_a_job_pointing_at_its_own_data_keeps_the_shared_transforms(self, defaults):
        """
        The override every job has, and the one most likely to be mishandled.

        `source.params` is a sibling of `source.transforms`, so setting the
        former must not disturb the latter -- and a job set where every job
        silently lost its transforms would still train, and would train on
        unscaled, unreduced data.
        """
        merged = deep_merge(defaults, {"source": {"params": {"directory": "clusters/EURUSD"}}})

        assert merged["source"]["params"] == {"directory": "clusters/EURUSD"}
        assert merged["source"]["kind"] == "model"
        assert merged["source"]["transforms"]["sequence"]["length"] == 20

    def test_the_defaults_are_reusable_across_jobs(self, defaults):
        """
        Merged once per job, so the first job must not disturb the second.

        The aliasing bug this guards against is invisible with one job and
        intermittent with forty, which is the worst combination there is.
        """
        first = deep_merge(defaults, {"model": {"units": 64}})
        second = deep_merge(defaults, {"model": {"gnn_layers": 1}})

        assert first["model"]["units"] == 64
        assert second["model"]["units"] == 256
        assert second["model"]["gnn_layers"] == 1
        assert defaults["model"]["units"] == 256
```

---

## 6. `tests/rade_qnet/core/spec/test_spec_reports.py`

4694 bytes · SHA-256 `6abb6cbdee9201df`

```python
"""
Tests for the reports specification.

``fail_fast`` exists so that "a report never fails a run" is the default rather
than the only behaviour. Defaulting it to false is deliberate: reporting code
is the least-tested code in any pipeline, and a figure that cannot render
should not discard four hours of training. A caller who genuinely wants a
report failure to be fatal has to say so, which makes it reviewable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.spec.reports import ReportsSpec


class TestDefaults:
    """Defaults produce a useful run with no configuration."""

    def test_it_constructs_with_no_arguments(self):
        """Reached through a ``default_factory``, so it must be bare."""
        assert ReportsSpec() is not None

    def test_the_summary_report_is_enabled_by_default(self):
        """
        A run with no report configuration still explains itself.

        The summary needs only the bundle, so it works for every model and
        every engine -- which is what makes it safe as the default.
        """
        assert ReportsSpec().enabled == ("summary",)

    def test_report_failures_are_not_fatal_by_default(self):
        """The rule stated in the module docstring."""
        assert ReportsSpec().fail_fast is False


class TestEnabledReports:
    """The enabled list is validated, not merely accepted."""

    def test_an_empty_list_is_accepted(self):
        """
        Disabling reporting entirely is legitimate.

        A tuning sweep of four hundred trials does not want four hundred
        summary pages.
        """
        assert ReportsSpec(enabled=()).enabled == ()

    def test_duplicates_are_rejected(self):
        """
        A report named twice would run twice and overwrite its own output.

        The second run's figures replace the first's, so the duplicate is
        pure waste -- and it usually means the user merged two configurations
        by hand.
        """
        with pytest.raises(ValidationError, match="summary"):
            ReportsSpec(enabled=("summary", "curves", "summary"))

    def test_several_distinct_reports_are_accepted(self):
        """The normal case."""
        assert ReportsSpec(enabled=("summary", "curves")).enabled == ("summary", "curves")

    def test_order_is_preserved(self):
        """
        Reports run in the order given.

        Not sorted, because a user may want the expensive one last so the
        cheap ones have already been written if it fails.
        """
        assert ReportsSpec(enabled=("curves", "summary")).enabled == ("curves", "summary")


class TestFigureSettings:
    """Figure settings are bounded, so a typo cannot produce a useless file."""

    @pytest.mark.parametrize("figure_format", ["png", "svg", "pdf"])
    def test_supported_formats_are_accepted(self, figure_format):
        """All three are lossless or vector."""
        assert ReportsSpec(figure_format=figure_format).figure_format == figure_format

    def test_a_lossy_format_is_rejected(self):
        """
        A figure that will be read carefully should not be a JPEG.

        Restricting the set at the spec level means no report has to defend
        against it.
        """
        with pytest.raises(ValidationError):
            ReportsSpec(figure_format="jpg")

    def test_a_sensible_resolution_is_accepted(self):
        """The normal case."""
        assert ReportsSpec(figure_dpi=300).figure_dpi == 300

    @pytest.mark.parametrize("dpi", [10, 1200])
    def test_an_unusable_resolution_is_rejected(self, dpi):
        """
        Bounded on both sides.

        Ten is illegible; twelve hundred produces a hundred-megabyte PNG that
        nothing will open. Neither is what the user meant.
        """
        with pytest.raises(ValidationError):
            ReportsSpec(figure_dpi=dpi)


class TestStrictness:
    """The usual spec guarantees."""

    def test_an_unknown_key_is_rejected(self):
        """A typo fails at load."""
        with pytest.raises(ValidationError):
            ReportsSpec(enable=("summary",))

    def test_the_spec_is_frozen(self):
        """A report cannot enable itself mid-run."""
        spec = ReportsSpec()
        with pytest.raises(ValidationError):
            spec.fail_fast = True

    def test_round_trip_is_exact(self):
        """So the bundle records what actually ran."""
        spec = ReportsSpec(
            enabled=("summary", "curves"),
            directory_name="figures",
            fail_fast=True,
            figure_format="svg",
            figure_dpi=200,
        )
        assert ReportsSpec.model_validate(spec.model_dump()) == spec
```

---

## 7. `tests/rade_qnet/core/spec/test_spec_run.py`

12919 bytes · SHA-256 `950ffd0f989e98d5`

```python
"""
Tests for the run specification and its single parsing entry point.

``parse_run_spec`` is the only way configuration enters the framework, and that
is what makes two things possible: the ``task`` discriminator can be defaulted
in one place rather than relying on schema magic, and a pydantic
``ValidationError`` can be converted into one ``SpecError`` whose message is
actionable. A user who mistyped a YAML key should see one framework error, not
a pydantic traceback.
"""

from __future__ import annotations

import json

import pytest

from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.core.provenance.hashing import digest_spec
from src.rade_qnet.core.spec.run import (
    ComponentRef,
    ReinforcementRunSpec,
    SupervisedRunSpec,
    dump_run_spec,
    load_run_spec,
    parse_run_spec,
)


class TestComponentRef:
    """Both natural YAML spellings are accepted."""

    def test_a_bare_string_is_accepted(self):
        """
        ``model: ridge`` is how a user writes a model with no parameters.

        Requiring ``{name: ridge}`` would be correct and unpleasant.
        """
        spec = parse_run_spec({"model": "ridge"}, origin="test")
        assert spec.model.name == "ridge"
        assert spec.model.params == {}

    def test_inline_parameters_are_collected(self):
        """
        ``{name: ridge, alpha: 1.0}`` keeps parameters at the top level.

        Nesting them under ``params`` is how the data is stored, not how a
        user wants to type it.
        """
        spec = parse_run_spec({"model": {"name": "ridge", "alpha": 1.0}}, origin="test")
        assert spec.model.params == {"alpha": 1.0}

    def test_the_explicit_form_is_still_accepted(self):
        """The normalised form must survive a round trip through itself."""
        spec = parse_run_spec({"model": {"name": "ridge", "params": {"alpha": 1.0}}}, origin="test")
        assert spec.model.params == {"alpha": 1.0}

    def test_a_mapping_without_a_name_is_rejected(self):
        """There is no way to guess which component was meant."""
        with pytest.raises(SpecError, match="name"):
            parse_run_spec({"model": {"alpha": 1.0}}, origin="test")

    def test_describe_renders_without_the_pydantic_repr(self):
        """
        A component reference appears in reports a human reads.

        The default repr renders as ``name='ridge' params={}``, which reads as
        a debugging artefact in a summary page.
        """
        assert ComponentRef(name="ridge").describe() == "ridge"
        assert "alpha=1.0" in ComponentRef(name="ridge", params={"alpha": 1.0}).describe()


class TestTaskDiscrimination:
    """The task selects which run spec is built."""

    def test_the_task_defaults_to_supervised(self):
        """
        Defaulted at the entry point, not in the schema.

        A discriminated union cannot take a default tag from a member's field
        default, so the single parsing entry point supplies it -- which keeps
        the behaviour in one readable place rather than in schema machinery.
        """
        assert isinstance(parse_run_spec({"model": "demo"}, origin="test"), SupervisedRunSpec)

    def test_the_reinforcement_task_resolves(self):
        """The other member of the union."""
        spec = parse_run_spec(
            {"model": "demo", "task": "reinforcement", "environment": "hedging"}, origin="test"
        )
        assert isinstance(spec, ReinforcementRunSpec)

    def test_an_unknown_task_is_rejected(self):
        """A misspelled task fails at load."""
        with pytest.raises(SpecError):
            parse_run_spec({"model": "demo", "task": "supervize"}, origin="test")


class TestRequiredFields:
    """A top-level spec legitimately requires what it cannot default."""

    def test_the_model_is_required(self):
        """
        There is no sensible default model.

        This is the refinement to the "every spec constructs with defaults"
        rule: it applies to specs reached through a ``default_factory``, not
        to the top-level spec, which must name what to run.
        """
        with pytest.raises(SpecError, match="model"):
            parse_run_spec({}, origin="test")

    def test_everything_else_defaults(self):
        """
        Naming a model is enough to get a complete, valid spec.

        Which is what makes the first customisation tier -- spec only -- real.
        """
        spec = parse_run_spec({"model": "demo"}, origin="test")
        assert spec.hardware is not None
        assert spec.reports is not None
        assert spec.source is not None
        assert spec.training is not None


class TestValidationMessages:
    """Every failure arrives as one SpecError naming the field."""

    def test_an_unknown_top_level_key_is_rejected(self):
        """
        The second diagnosed defect's counterpart.

        A misspelled key that was ignored means the run trains with a default
        and reports plausible numbers.
        """
        with pytest.raises(SpecError, match="sed"):
            parse_run_spec({"model": "demo", "sed": 3}, origin="test")

    def test_the_error_names_the_offending_field_path(self):
        """
        A dotted path, so a deep failure is locatable.

        Without it, "split fractions must sum to less than one" does not say
        which of several nested splits is at fault.
        """
        with pytest.raises(SpecError) as caught:
            parse_run_spec(
                {
                    "model": "demo",
                    "source": {
                        "kind": "tabular",
                        "path": "x.csv",
                        "split": {
                            "kind": "chronological",
                            "validation_fraction": 0.8,
                            "test_fraction": 0.8,
                        },
                    },
                },
                origin="test",
            )
        assert "split" in str(caught.value)

    def test_the_origin_appears_in_the_message(self):
        """
        The message says which file was wrong.

        A job set loads many specs; an error that does not name its source
        sends the reader to the wrong file.
        """
        with pytest.raises(SpecError, match=r"my_config\.yaml"):
            parse_run_spec({}, origin="my_config.yaml")

    def test_a_non_mapping_payload_is_rejected(self):
        """
        A YAML file containing a list is a common mistake.

        Reported as a spec error, because the fix is the user's.
        """
        with pytest.raises(SpecError):
            parse_run_spec(["model", "demo"], origin="test")


class TestCrossFieldValidation:
    """Combinations that cannot work are rejected at load."""

    def test_a_sequence_window_with_an_explicit_split_is_rejected(self):
        """
        The framework cannot insert a boundary gap into caller-given indices.

        With a window longer than one scenario, a sample ending just after a
        split boundary contains training scenarios. For a generated split the
        framework adds clearance; for an explicit split only the caller can.
        """
        with pytest.raises(SpecError, match="sequence"):
            parse_run_spec(
                {
                    "model": "demo",
                    "source": {
                        "kind": "tabular",
                        "path": "x.csv",
                        "split": {
                            "kind": "explicit",
                            "train": [0, 1, 2],
                            "validation": [3],
                            "test": [4],
                        },
                        "transforms": {"sequence": {"length": 20}},
                    },
                },
                origin="test",
            )

    def test_a_single_scenario_window_with_an_explicit_split_is_accepted(self):
        """
        A window of one cannot straddle a boundary.

        So the restriction applies only where it is needed.
        """
        spec = parse_run_spec(
            {
                "model": "demo",
                "source": {
                    "kind": "tabular",
                    "path": "x.csv",
                    "split": {
                        "kind": "explicit",
                        "train": [0, 1, 2],
                        "validation": [3],
                        "test": [4],
                    },
                    "transforms": {"sequence": {"length": 1}},
                },
            },
            origin="test",
        )
        assert spec.source.transforms.sequence.length == 1


class TestImmutabilityAndRoundTrip:
    """A spec is frozen and survives serialisation exactly."""

    def test_the_spec_is_frozen(self):
        """So the bundle's record is a record of what happened."""
        spec = parse_run_spec({"model": "demo"}, origin="test")
        with pytest.raises(Exception, match="frozen"):
            spec.seed = 99

    def test_round_trip_is_exact(self):
        """
        The first diagnosed defect, at the top level.

        A spec that does not round-trip cannot be recorded in a bundle.
        """
        spec = parse_run_spec(
            {
                "model": {"name": "demo", "alpha": 1.0},
                "seed": 99,
                "tags": ["nightly"],
                "hardware": {"device": "cpu", "determinism": "warn"},
                "training": {"engine": "torch", "epochs": 7},
            },
            origin="test",
        )
        reloaded = parse_run_spec(json.loads(spec.model_dump_json()), origin="test")
        assert reloaded == spec

    def test_the_digest_is_stable_across_reparse(self):
        """
        A re-parsed spec recognises itself.

        Which is what lets a bundle claim to record the configuration that
        produced it, and what makes a step cache hit.
        """
        payload = {"model": "demo", "seed": 3}
        assert digest_spec(parse_run_spec(payload, origin="a")) == digest_spec(
            parse_run_spec(payload, origin="b")
        )

    def test_key_order_does_not_change_the_digest(self):
        """Reordering two keys in a YAML file is not a change to the run."""
        first = parse_run_spec({"model": "demo", "seed": 3}, origin="test")
        second = parse_run_spec({"seed": 3, "model": "demo"}, origin="test")
        assert digest_spec(first) == digest_spec(second)


class TestFileIo:
    """Loading and dumping, with every user-facing failure named."""

    def test_a_yaml_file_round_trips(self, tmp_path):
        """The normal path."""
        path = tmp_path / "run.yaml"
        spec = parse_run_spec({"model": "demo", "seed": 5}, origin="test")
        dump_run_spec(spec, path)
        assert load_run_spec(path) == spec

    def test_a_json_file_round_trips(self, tmp_path):
        """
        JSON is accepted as well as YAML.

        A bundle stores JSON, so reloading a bundle's spec uses the same
        reader as loading a hand-written configuration.
        """
        path = tmp_path / "run.json"
        spec = parse_run_spec({"model": "demo", "seed": 5}, origin="test")
        dump_run_spec(spec, path)
        assert load_run_spec(path) == spec

    def test_a_missing_file_is_a_spec_error(self):
        """
        Every "you pointed us at the wrong file" case is the user's to fix.

        So all of them report as a spec error rather than as an OSError.
        """
        with pytest.raises(SpecError):
            load_run_spec("/nonexistent/path/run.yaml")

    def test_a_file_containing_a_list_is_a_spec_error(self, tmp_path):
        """A common YAML mistake, reported in terms the user can act on."""
        path = tmp_path / "run.yaml"
        path.write_text("- model: demo\n", encoding="utf-8")
        with pytest.raises(SpecError):
            load_run_spec(path)

    def test_malformed_yaml_is_a_spec_error(self, tmp_path):
        """Not a yaml library exception leaking through."""
        path = tmp_path / "run.yaml"
        path.write_text("model: [unclosed\n", encoding="utf-8")
        with pytest.raises(SpecError):
            load_run_spec(path)

    def test_an_unknown_suffix_is_rejected_on_dump(self, tmp_path):
        """
        Writing a spec to an unrecognised format would be unreadable later.

        Better to refuse than to write something that cannot be loaded.
        """
        spec = parse_run_spec({"model": "demo"}, origin="test")
        with pytest.raises(SpecError):
            dump_run_spec(spec, tmp_path / "run.txt")

    def test_the_loaded_file_name_appears_in_a_validation_error(self, tmp_path):
        """A job set's failure must name the file that caused it."""
        path = tmp_path / "broken.yaml"
        path.write_text("seed: 3\n", encoding="utf-8")
        with pytest.raises(SpecError, match=r"broken\.yaml"):
            load_run_spec(path)
```

---

## 8. `tests/rade_qnet/core/spec/test_spec_training.py`

9181 bytes · SHA-256 `0f30bc07b387cda5`

```python
"""
Tests for the training specification.

Two structural decisions are pinned down here.

``TrainingSpec`` discriminates on ``engine``, so asking for ``epochs`` with the
XGBoost engine fails at load rather than being ignored. The alternative -- one
training spec with every backend's fields on it -- means most fields are
meaningless for any given run, and nothing can tell you which.

``RlTrainingSpec`` is deliberately **not** in that union. It is selected by
``task`` one level up, because reinforcement learning differs in what it
*means* rather than in which library runs it: an epoch over a fixed dataset has
no counterpart when experience is generated as training proceeds.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from src.rade_qnet.core.spec.training import (
    CheckpointSpec,
    EarlyStoppingSpec,
    RlTrainingSpec,
    SchedulerSpec,
    SklearnTrainingSpec,
    TorchTrainingSpec,
    TrainingSpec,
    XGBoostTrainingSpec,
)

_TRAINING_ADAPTER = TypeAdapter(TrainingSpec)


class TestDefaults:
    """Nested specs are bare-constructible; the top level is not."""

    @pytest.mark.parametrize(
        "spec_type", [EarlyStoppingSpec, CheckpointSpec, SchedulerSpec, TorchTrainingSpec]
    )
    def test_each_spec_constructs_with_defaults(self, spec_type):
        """
        The refined rule from the phase definition of done.

        Anything reached through another spec's ``default_factory`` must be
        constructible bare, or the outer spec cannot be built at all.
        """
        assert spec_type() is not None

    def test_early_stopping_is_off_by_default(self):
        """
        Stopping early is a choice, not a default.

        A user who did not ask for it should get the epoch budget they
        configured.
        """
        assert EarlyStoppingSpec().enabled is False

    def test_the_best_checkpoint_is_restored_by_default(self):
        """
        The safe default.

        Without restoring, the reported metrics and the saved weights
        describe different models -- which is why ``FitOutcome`` records
        whether it happened.
        """
        assert CheckpointSpec().restore_best is True


class TestEngineUnion:
    """The union discriminates on ``engine``."""

    @pytest.mark.parametrize(
        ("engine", "expected"),
        [
            ("torch", TorchTrainingSpec),
            ("xgboost", XGBoostTrainingSpec),
            ("sklearn", SklearnTrainingSpec),
        ],
    )
    def test_the_engine_selects_the_member(self, engine, expected):
        """Each backend gets its own fields, and only its own."""
        assert isinstance(_TRAINING_ADAPTER.validate_python({"engine": engine}), expected)

    def test_an_unknown_engine_is_rejected(self):
        """A misspelled engine fails at load."""
        with pytest.raises(ValidationError):
            _TRAINING_ADAPTER.validate_python({"engine": "pytorch"})

    def test_a_torch_field_is_rejected_for_xgboost(self):
        """
        The payoff of discriminating.

        ``epochs`` means nothing to a boosted-tree fit. Accepting and ignoring
        it would let a user tune a field with no effect and never find out.
        """
        with pytest.raises(ValidationError):
            _TRAINING_ADAPTER.validate_python({"engine": "xgboost", "epochs": 50})

    def test_batch_size_is_not_a_training_field(self):
        """
        Batch size belongs to the loader.

        It describes how data is delivered, not how the model is optimised,
        and putting it in both places is how the two end up disagreeing.
        """
        assert "batch_size" not in TorchTrainingSpec.model_fields


class TestTorchTraining:
    """Field-level and cross-field validation for the gradient engine."""

    def test_zero_epochs_is_rejected(self):
        """A run that trains for no epochs produces an untrained model."""
        with pytest.raises(ValidationError):
            TorchTrainingSpec(epochs=0)

    def test_a_non_positive_learning_rate_is_rejected(self):
        """
        Zero learns nothing; negative ascends the loss.

        Both complete without error, which is exactly why they are rejected
        here instead.
        """
        with pytest.raises(ValidationError):
            TorchTrainingSpec(learning_rate=0.0)

    def test_a_clashing_monitor_direction_is_rejected(self):
        """
        Early stopping and checkpointing must agree on what "better" means.

        If one minimises and the other maximises the same metric, they select
        different epochs as best -- so training stops on one judgement and
        the saved weights reflect the other.
        """
        with pytest.raises(ValidationError, match="monitor"):
            TorchTrainingSpec(
                early_stopping={"enabled": True, "monitor": "val_loss", "mode": "min"},
                checkpoint={"monitor": "val_loss", "mode": "max"},
            )

    def test_agreeing_monitor_directions_are_accepted(self):
        """The legitimate configuration."""
        spec = TorchTrainingSpec(
            early_stopping={"enabled": True, "monitor": "val_loss", "mode": "min"},
            checkpoint={"monitor": "val_loss", "mode": "min"},
        )
        assert spec.early_stopping.enabled is True

    def test_different_metrics_may_use_different_directions(self):
        """
        The check is per metric, not global.

        Stopping on a loss while checkpointing on an accuracy is coherent,
        and a blanket rule would reject it.
        """
        spec = TorchTrainingSpec(
            early_stopping={"enabled": True, "monitor": "val_loss", "mode": "min"},
            checkpoint={"monitor": "val_accuracy", "mode": "max"},
        )
        assert spec.checkpoint.mode == "max"


class TestReinforcementTraining:
    """RL is selected by task, not by engine."""

    def test_it_is_not_a_member_of_the_engine_union(self):
        """
        Asking for it by engine does not work, and should not.

        It is selected one level up by ``task``, because the difference is
        what training *means*, not which library performs it.
        """
        with pytest.raises(ValidationError):
            _TRAINING_ADAPTER.validate_python({"engine": "rl"})

    def test_it_constructs_with_defaults(self):
        """Reached through the reinforcement run spec's default factory."""
        assert RlTrainingSpec() is not None

    def test_an_on_policy_batch_larger_than_the_update_is_rejected(self):
        """
        An on-policy learner must not reuse stale experience.

        A batch larger than the experience collected between updates can only
        be filled by sampling data the algorithm assumes is fresh, which
        breaks its correctness guarantee without any error.
        """
        with pytest.raises(ValidationError, match="batch_size"):
            RlTrainingSpec(learner="ppo", steps_per_update=128, batch_size=256)

    def test_an_on_policy_batch_within_the_update_is_accepted(self):
        """The legitimate configuration."""
        assert RlTrainingSpec(learner="ppo", steps_per_update=2048, batch_size=64) is not None

    def test_an_off_policy_learner_may_exceed_the_update_size(self):
        """
        The check applies only to on-policy learners.

        An off-policy learner samples from a replay buffer by design, so a
        batch larger than one update's collection is normal rather than a
        mistake.
        """
        assert RlTrainingSpec(learner="dqn", steps_per_update=1, batch_size=256) is not None


class TestStrictness:
    """The usual spec guarantees."""

    def test_an_unknown_key_is_rejected(self):
        """A typo fails at load."""
        with pytest.raises(ValidationError):
            TorchTrainingSpec(epoch=50)

    def test_the_spec_is_frozen(self):
        """A tuning trial must build a new spec rather than mutate one."""
        spec = TorchTrainingSpec()
        with pytest.raises(ValidationError):
            spec.epochs = 10

    def test_round_trip_is_exact(self):
        """
        Including the nested callback specs.

        The seventh diagnosed defect was a tuning loop using
        ``dataclasses.replace`` on a spec that was not a dataclass; a spec
        that round-trips can be rebuilt instead.
        """
        spec = TorchTrainingSpec(
            epochs=25,
            learning_rate=3e-4,
            early_stopping={"enabled": True, "patience": 5},
            checkpoint={"restore_best": False},
            scheduler={"kind": "cosine"},
        )
        assert TorchTrainingSpec.model_validate(spec.model_dump()) == spec

    def test_the_union_round_trips_through_json(self):
        """
        The discriminator survives serialisation.

        If ``engine`` were dropped on dump, the reloaded spec could not be
        resolved back to its member, and the bundle's record of the run would
        be unusable.
        """
        spec = _TRAINING_ADAPTER.validate_python({"engine": "xgboost"})
        payload = _TRAINING_ADAPTER.dump_python(spec, mode="json")
        assert _TRAINING_ADAPTER.validate_python(payload) == spec
```

