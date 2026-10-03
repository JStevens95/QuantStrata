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

from src.rade_xl.core.spec.data import (
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
