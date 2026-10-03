"""
Tests for the composed fitted state.

A bundle carries one fitted state, not three. That matters because the state
is what an inference run must *reuse* rather than refit: a standardiser refitted
on a serving window standardises against that window's own statistics, which
is a different transform from the one the model was trained through, and the
predictions come out plausible and wrong.

The decision worth testing hardest is target-inverse ownership. Exactly one
part may own the inverse, because two parts both claiming it would mean the
order of un-scaling decides the answer, and a state with none would hand back
predictions in scaled units while reporting them as original ones. A mean
absolute error of 0.03 is excellent in original units and meaningless in
standardised ones, and nothing in the number says which it is.

A scaler fitted with ``scale_target=False`` does *not* own the inverse even
though it is present, because it never touched the target -- claiming
ownership would be technically true and actively misleading in the bundle.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_xl.core.runtime.errors import ContractError
from src.rade_xl.core.spec.data import ScalingSpec
from src.rade_xl.sources.dataset.transforms.composite import CompositeState, DatasetState
from src.rade_xl.sources.dataset.transforms.encoding import EncodingState
from src.rade_xl.sources.dataset.transforms.scaling import ScalingState


def scaler(*, scale_target: bool = True) -> ScalingState:
    """
    Fit a scaler over a small training window.

    Parameters
    ----------
    scale_target
        Whether the scaler transforms the target as well as the features.

    Returns
    -------
    ScalingState
        The fitted scaler.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    target = rng.normal(loc=10.0, scale=4.0, size=100)
    return ScalingState.fit(
        features,
        target,
        spec=ScalingSpec(method="standard", scale_target=scale_target),
        train_indices=np.arange(70),
    )


class TestComposition:
    """Whichever parts were fitted, answered in one place."""

    def test_only_the_supplied_parts_are_present(self):
        """
        So a tabular run carries a scaler and nothing else.

        Placeholder parts would mean every bundle claims an entity encoder,
        and a reader could not tell a problem with no entities from one whose
        encoder was never fitted.
        """
        state = DatasetState.of(scaling=scaler())
        assert state.has_part("scaling")
        assert not state.has_part("encoding")

    def test_several_parts_compose(self):
        """The multi-entity case, which is the flagship model's shape."""
        state = DatasetState.of(scaling=scaler(), encoding=EncodingState.fit(["EURUSD", "GBPUSD"]))
        assert state.has_part("scaling")
        assert state.has_part("encoding")

    def test_an_empty_composition_is_valid(self):
        """
        Because a model may need no fitted state at all.

        A tree model over raw features fits nothing, and requiring a state
        would make it carry an empty one for the sake of the interface.
        """
        assert not DatasetState.of().has_part("scaling")

    def test_requesting_an_absent_part_is_refused(self):
        """
        Naming what is present, since the cause is usually a typo.

        Returning ``None`` would push the failure to the first use of the
        result, where it is an attribute error on ``None`` with no mention of
        the part that was missing.
        """
        with pytest.raises(ContractError):
            DatasetState.of(scaling=scaler()).part("encoding")

    def test_a_present_part_is_returned_as_itself(self):
        """
        Not a copy, so the composition is a view rather than a clone.

        A copy would double the memory of a reduction basis, and the two
        copies could then be loaded and saved independently.
        """
        fitted = scaler()
        assert DatasetState.of(scaling=fitted).part("scaling") is fitted


class TestTargetInverseOwnership:
    """Exactly one owner, because zero and two both give wrong answers."""

    def test_the_scaler_owns_the_inverse_when_it_scaled_the_target(self):
        """
        So predictions come back in the units the data arrived in.

        Reported without inverting, a mean absolute error of 0.03 is in
        standardised units and says nothing about the size of the error in
        the units anyone cares about.
        """
        state = DatasetState.of(scaling=scaler(scale_target=True))
        predictions = np.zeros(5)
        assert not np.array_equal(state.inverse_transform_targets(predictions), predictions)

    def test_a_scaler_that_left_the_target_alone_does_not_own_it(self):
        """
        Even though it is present, which is the subtle half.

        Claiming ownership would be technically true -- the part is there --
        and actively misleading in the bundle, because the inverse would be
        the identity while the bundle advertised a transform.
        """
        state = DatasetState.of(scaling=scaler(scale_target=False))
        predictions = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(state.inverse_transform_targets(predictions), predictions)

    def test_a_composition_with_no_owner_passes_predictions_through(self):
        """
        Unchanged, which is correct when nothing transformed the target.

        The alternative -- raising -- would make every model that does not
        scale its target unable to report a metric.
        """
        predictions = np.array([1.0, 2.0])
        assert np.array_equal(DatasetState.of().inverse_transform_targets(predictions), predictions)

    def test_the_inverse_round_trips_the_training_target(self):
        """
        Which is the property a reported metric depends on.

        Asserted on real numbers rather than on the presence of a method: an
        inverse that applied the mean and the scale in the wrong order is
        still an inverse-shaped function, and still wrong.
        """
        rng = np.random.default_rng(1)
        target = rng.normal(loc=10.0, scale=4.0, size=100)
        features = rng.normal(size=(100, 3))
        fitted = ScalingState.fit(
            features,
            target,
            spec=ScalingSpec(method="standard", scale_target=True),
            train_indices=np.arange(70),
        )
        state = DatasetState.of(scaling=fitted)
        scaled = fitted.transform_targets(target)
        assert np.allclose(state.inverse_transform_targets(scaled), target)

    def test_only_one_part_can_ever_own_the_inverse(self):
        """
        Enforced by the shape of the field rather than by a check.

        ``target_owner`` is a single name, so two parts cannot both claim the
        inverse -- which would be a silent correctness bug, returning
        predictions in units that depend on the order the parts were
        un-applied in.
        """
        annotation = CompositeState.__init__.__annotations__["target_owner"]
        assert annotation == "str | None"

    def test_an_owner_that_was_not_supplied_is_refused(self):
        """
        Because the inverse would have nothing to delegate to.

        Accepted, it would produce a state that advertises a target transform
        and returns predictions untouched -- which is the "no owner" bug
        wearing the opposite label.
        """
        with pytest.raises(ContractError, match="target_owner"):
            DatasetState(parts={"scaling": scaler()}, target_owner="encoding")


class TestPersistence:
    """One directory per part, which is what a bundle writes."""

    def test_every_part_survives_a_round_trip(self, tmp_path):
        """
        Because the bundle is the only record of the fitted transform.

        An inference run that cannot reload it has to refit, which is the
        failure the whole state exists to prevent.
        """
        state = DatasetState.of(scaling=scaler(), encoding=EncodingState.fit(["EURUSD", "GBPUSD"]))
        state.save(tmp_path)
        reloaded = DatasetState.load(tmp_path)
        assert reloaded.has_part("scaling")
        assert reloaded.has_part("encoding")

    def test_the_reloaded_inverse_behaves_identically(self, tmp_path):
        """
        The claim that matters, as distinct from the parts being present.

        Comparing the stored arrays is a proxy; comparing the inverse's
        output is what an inference run actually depends on.
        """
        state = DatasetState.of(scaling=scaler())
        state.save(tmp_path)
        predictions = np.array([0.0, 1.0, -1.0])
        assert np.allclose(
            DatasetState.load(tmp_path).inverse_transform_targets(predictions),
            state.inverse_transform_targets(predictions),
        )

    def test_target_ownership_survives_the_round_trip(self, tmp_path):
        """
        Otherwise a reloaded state reports metrics in the wrong units.

        And it would do so silently, since the predictions are plausible
        numbers either way.
        """
        state = DatasetState.of(scaling=scaler(scale_target=False))
        state.save(tmp_path)
        predictions = np.array([1.0, 2.0])
        assert np.array_equal(
            DatasetState.load(tmp_path).inverse_transform_targets(predictions), predictions
        )

    def test_an_absent_part_is_not_reloaded(self, tmp_path):
        """
        So a reloaded state claims exactly what was fitted.

        A part reconstructed from an empty directory would be an identity
        transform advertised as a fitted one.
        """
        DatasetState.of(scaling=scaler()).save(tmp_path)
        assert not DatasetState.load(tmp_path).has_part("encoding")

    def test_an_empty_composition_round_trips(self, tmp_path):
        """
        Because a model that fits nothing still writes a bundle.

        And the bundle reader must not need to know in advance whether there
        was anything to read.
        """
        DatasetState.of().save(tmp_path)
        assert not DatasetState.load(tmp_path).has_part("scaling")


class TestDescription:
    """What the bundle records about the state, for a person to read."""

    def test_every_part_is_described(self):
        """
        So a bundle says what was fitted without being loaded.

        Which is the only way to tell whether a stored model expects scaled
        inputs before handing it any.
        """
        described = DatasetState.of(
            scaling=scaler(), encoding=EncodingState.fit(["EURUSD"])
        ).describe()
        assert "scaling" in str(described)
        assert "encoding" in str(described)

    def test_the_target_owner_is_named(self):
        """
        Because it decides the units every reported metric is in.

        A reader comparing two runs' mean absolute errors needs to know
        whether both were inverted, and this is where that is recorded.
        """
        described = DatasetState.of(scaling=scaler()).describe()
        assert "scaling" in str(described)
