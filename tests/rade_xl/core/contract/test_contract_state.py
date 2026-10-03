"""
Tests for fitted state.

The decisive property is that ``inverse_transform_targets`` is **abstract**. A
model that scales its target and cannot invert the scaling reports a mean
absolute error in standardised space -- a number nobody can act on, and one
that looks perfectly reasonable. Making the method abstract means that failure
has to be chosen rather than reached by omission.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.rade_xl.core.contract.state import FittedState, IdentityFittedState
from src.rade_xl.testkit.fixtures import StandardisingState


class TestAbstractInterface:
    """The base demands exactly the three things a state must do."""

    def test_the_base_cannot_be_instantiated(self):
        """An incomplete state is not a usable state."""
        with pytest.raises(TypeError):
            FittedState()

    @pytest.mark.parametrize("method", ["save", "load", "inverse_transform_targets"])
    def test_each_required_method_is_abstract(self, method):
        """
        All three are required, and the inverse especially.

        A default inverse would be an identity, so a model that scales its
        target would silently report metrics in the wrong units.
        """
        assert getattr(FittedState, method).__isabstractmethod__

    def test_a_subclass_missing_the_inverse_cannot_be_instantiated(self):
        """The guarantee, demonstrated."""

        class Incomplete(FittedState):
            """Implements saving but not the inverse."""

            def save(self, directory):
                """Write nothing."""

            @classmethod
            def load(cls, directory):
                """Read nothing."""
                return cls()

        with pytest.raises(TypeError, match="inverse_transform_targets"):
            Incomplete()

    def test_describe_is_not_abstract(self):
        """
        Describing is optional, unlike the other three.

        A state that cannot describe itself produces a less informative
        report, not a wrong one.
        """
        assert not getattr(FittedState.describe, "__isabstractmethod__", False)

    def test_save_receives_a_directory_not_a_file(self):
        """
        Directory-based, because a state is naturally several arrays.

        Forcing a scaler's mean and scale, a graph's three arrays and a
        selected feature list through one pickle reintroduces the fragility
        this class exists to remove.
        """
        assert "directory" in inspect.signature(FittedState.save).parameters


class TestIdentityState:
    """The explicit "this model transforms nothing" state."""

    def test_the_inverse_returns_the_input_unchanged(self):
        """The honest behaviour for an untransformed target."""
        values = np.array([1.0, -2.0, 3.5])
        assert np.array_equal(IdentityFittedState().inverse_transform_targets(values), values)

    def test_it_round_trips_through_save_and_load(self, tmp_path):
        """Every state must survive a bundle write and read."""
        IdentityFittedState().save(tmp_path)
        assert IdentityFittedState.load(tmp_path) == IdentityFittedState()

    def test_saving_writes_a_marker_rather_than_nothing(self, tmp_path):
        """
        An empty directory is indistinguishable from a failed write.

        A marker lets bundle verification confirm the state was saved
        deliberately.
        """
        IdentityFittedState().save(tmp_path)
        assert list(tmp_path.iterdir())

    def test_instances_compare_equal(self):
        """
        All instances are equivalent, so they compare so.

        Needed for a bundle round-trip test to be able to assert equality.
        """
        assert IdentityFittedState() == IdentityFittedState()

    def test_it_is_hashable(self):
        """
        Defining equality without a hash would make it unhashable.

        A state can legitimately end up in a set or as a dictionary key while
        a pipeline is assembling a bundle.
        """
        assert len({IdentityFittedState(), IdentityFittedState()}) == 1

    def test_it_does_not_compare_equal_to_another_state(self):
        """
        An identity state is not a standardiser.

        If it compared equal, a bundle round-trip test could pass while
        having loaded the wrong state entirely.
        """
        assert IdentityFittedState() != StandardisingState(mean=0.0, scale=1.0)

    def test_describe_names_its_type(self):
        """Reported in the run summary, so it must identify itself."""
        assert IdentityFittedState().describe()["type"] == "IdentityFittedState"


class TestStandardisingState:
    """A state with a real inverse, used throughout the test suite."""

    def test_the_inverse_undoes_the_transform(self):
        """
        The property every state must have and this one demonstrates.

        If this round trip did not hold, every metric computed downstream
        would be in the wrong units.
        """
        state = StandardisingState.fit(np.array([1.0, 2.0, 3.0, 4.0]))
        values = np.array([1.0, 2.5, 4.0])
        assert np.allclose(state.inverse_transform_targets(state.transform(values)), values)

    def test_it_fits_only_what_it_is_given(self):
        """
        Fitted on the training split alone.

        The signature takes ``train_targets`` precisely to make fitting on
        everything awkward -- fitting on all scenarios leaks the held-out
        period's distribution into training.
        """
        state = StandardisingState.fit(np.array([0.0, 2.0]))
        assert state.mean == pytest.approx(1.0)

    def test_a_constant_target_does_not_produce_a_zero_scale(self):
        """
        A zero scale would make the transform divide by zero.

        A constant target legitimately occurs in a short window, so the scale
        falls back to one rather than raising.
        """
        assert StandardisingState.fit(np.array([3.0, 3.0, 3.0])).scale == 1.0

    def test_a_non_positive_scale_is_rejected_on_construction(self):
        """Constructed directly, an invalid scale is caught immediately."""
        with pytest.raises(ValueError, match="scale"):
            StandardisingState(mean=0.0, scale=0.0)

    def test_it_round_trips_through_save_and_load(self, tmp_path):
        """
        The statistics survive a bundle write.

        Which is what lets a six-month-old bundle invert its target scaling
        without the original run.
        """
        state = StandardisingState(mean=1.5, scale=2.5)
        state.save(tmp_path)
        assert StandardisingState.load(tmp_path) == state

    def test_describe_reports_the_fitted_statistics(self):
        """
        What was fitted appears in the run summary.

        Often the first place an anomaly -- a scale of 1e-9, say -- becomes
        visible.
        """
        described = StandardisingState(mean=1.5, scale=2.5).describe()
        assert described["mean"] == 1.5
        assert described["scale"] == 2.5

    def test_states_with_different_statistics_differ(self):
        """So a bundle round trip cannot pass by loading the wrong values."""
        assert StandardisingState(mean=1.0, scale=1.0) != StandardisingState(mean=2.0, scale=1.0)


class TestSubclassRecognition:
    """A concrete state is recognised as a FittedState."""

    @pytest.mark.parametrize("state", [IdentityFittedState(), StandardisingState()])
    def test_concrete_states_are_instances_of_the_base(self, state):
        """
        Checked with ``isinstance``, which the conformance suite relies on.

        Unlike the capability protocols, this is nominal: a state must
        inherit, because the framework calls ``load`` as a classmethod on the
        type the caller supplies.
        """
        assert isinstance(state, FittedState)
