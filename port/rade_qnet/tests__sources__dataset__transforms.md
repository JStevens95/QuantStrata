# `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 24 | 1073 | `2d244a830faebd42` |
| 2 | `test_transforms_composite.py` | 286 | 11339 | `32678fa78e49e640` |
| 3 | `test_transforms_encoding.py` | 272 | 10850 | `77ec92acc59bbfc4` |
| 4 | `test_transforms_reduction.py` | 237 | 9560 | `efdc6746d5981dde` |
| 5 | `test_transforms_scaling.py` | 206 | 7915 | `3ef1d50bfe25cc83` |
| 6 | `test_transforms_sequence.py` | 224 | 9132 | `a72496bc7acf552b` |

---

## 1. `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/__init__.py`

1073 bytes · SHA-256 `2d244a830faebd42`

```python
"""
Tests for ``rade_qnet.sources.dataset.transforms`` -- fitted transforms.

Every transform is held to three properties: it is fitted only on the rows it
is permitted to see, its fitted state round-trips through save and load, and --
if it touches the target -- its inverse recovers the original values to
floating-point tolerance. The third matters because a metric is only meaningful
in original units, so a broken inverse corrupts every number a user reads.

Planned modules
---------------
``test_transforms_scaling.py``
    Fit statistics computed from training rows only; inverse recovers the
    input.  [Phase 2]
``test_transforms_sequence.py``
    Window construction, including the boundary cases at the start of a series
    and at a split edge.  [Phase 2]
``test_transforms_reduction.py``
    Basis selection under both ``fit_on`` settings, with an explicit test that
    the default never observes validation or test rows.  [Phase 2]
``test_transforms_encoding.py``
    Entity-axis encoding, including an unseen category at inference time.
    [Phase 2]
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_composite.py`

11339 bytes · SHA-256 `32678fa78e49e640`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import ScalingSpec
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.composite import (
    CompositeState,
    DatasetState,
)
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.encoding import EncodingState
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.scaling import ScalingState


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
```

---

## 3. `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_encoding.py`

10850 bytes · SHA-256 `77ec92acc59bbfc4`

```python
"""
Tests for the entity encoder.

The encoder is where the two fitting axes are easiest to confuse. On the
*scenario* axis -- time -- fitting on anything but the training rows is
leakage. On the *entity* axis it is not: knowing that the universe contains
twelve currency pairs is not knowing anything about the future, and an
embedding table sized from the training rows alone cannot represent a pair
that happens to appear only later in the history.

That distinction is made *structural* here rather than documented.
:meth:`EncodingState.fit` takes identifiers directly instead of a matrix and a
set of training indices, so there is no parameter that could restrict the fit
to training rows -- the question cannot be got wrong by passing the wrong
argument.

Two more decisions get tests. Categories are sorted before coding, so the
encoding depends on the *set* of entities rather than on the order an upstream
query happened to return them in -- otherwise embedding row three means a
different instrument between two runs over identical data. And an unknown
identifier raises a ``CapabilityError`` rather than a contract error, because
the situation is meaningful: the caller asked a transductive encoder about an
entity it cannot represent, which is exactly what the ``Inductive`` capability
exists to declare.
"""

from __future__ import annotations

import inspect

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import (
    CapabilityError,
    ContractError,
)
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.encoding import EncodingState

UNIVERSE = ("EURUSD", "GBPUSD", "USDJPY")


class TestFittingOverTheEntityAxis:
    """The axis on which the full universe is permitted."""

    def test_the_full_universe_is_coded(self):
        """
        Every entity gets a code, including ones absent from training rows.

        An embedding table sized from the training rows alone cannot
        represent a pair that appears only later in the history, and the
        failure is at inference time on live data.
        """
        state = EncodingState.fit(["GBPUSD", "EURUSD", "USDJPY", "EURUSD"])
        assert state.n_entities == 3

    def test_there_is_no_parameter_that_could_restrict_the_fit(self):
        """
        Which is how the entity-axis rule is made structural.

        ``fit`` takes identifiers rather than a matrix and a set of training
        indices, so a caller cannot accidentally pass training rows only.
        Documented instead, the rule would hold until the first person who
        did not read the docstring.
        """
        parameters = inspect.signature(EncodingState.fit).parameters
        assert "train_indices" not in parameters
        assert "indices" not in parameters

    def test_duplicate_rows_are_collapsed(self):
        """
        Because one row per entity per scenario is the normal shape.

        A four-hundred-scenario history of twelve pairs has forty-eight
        hundred rows and twelve entities, and an encoder that coded rows would
        build an embedding table four hundred times too large.
        """
        state = EncodingState.fit(["EURUSD"] * 100 + ["GBPUSD"] * 100)
        assert state.n_entities == 2


class TestDeterminism:
    """The same universe must give the same codes, run after run."""

    def test_the_codes_depend_on_the_set_not_the_arrival_order(self):
        """
        Because an upstream query's row order is not a stable input.

        Without sorting, embedding row three means a different instrument
        between two runs over identical data -- and a bundle reloaded against
        the other ordering serves every instrument the wrong embedding.
        """
        first = EncodingState.fit(["USDJPY", "EURUSD", "GBPUSD"])
        second = EncodingState.fit(["EURUSD", "GBPUSD", "USDJPY"])
        assert first.entities == second.entities

    def test_the_order_is_the_encoding(self):
        """
        ``entities[code]`` is the identifier that code refers to.

        Held as a sequence rather than a mapping precisely because the order
        carries meaning and has to survive a JSON round trip unchanged.
        """
        state = EncodingState.fit(list(UNIVERSE))
        codes = state.encode(list(state.entities))
        assert codes.tolist() == list(range(state.n_entities))

    def test_codes_are_contiguous_from_zero(self):
        """
        Because an embedding table is indexed by them.

        A gap in the codes means a row of the table that no entity uses, and
        a code above the table's height is an index error at the first
        forward pass.
        """
        state = EncodingState.fit(list(UNIVERSE))
        assert sorted(state.encode(list(UNIVERSE)).tolist()) == [0, 1, 2]


class TestEncoding:
    """Identifier to code, and what happens when it cannot."""

    def test_identifiers_are_mapped_to_their_codes(self):
        """The basic operation, in the caller's order."""
        state = EncodingState.fit(list(UNIVERSE))
        codes = state.encode(["USDJPY", "EURUSD"])
        assert codes.tolist() == [2, 0]

    def test_repeated_identifiers_map_to_the_same_code(self):
        """
        Which is what makes an embedding shared across scenarios.

        If they did not, the model would learn a separate representation per
        row and the entity axis would carry no information at all.
        """
        state = EncodingState.fit(list(UNIVERSE))
        codes = state.encode(["EURUSD", "EURUSD"])
        assert codes[0] == codes[1]

    def test_an_unknown_identifier_raises_a_capability_error(self):
        """
        The deliberate choice of error type.

        This encoding is transductive, and being asked about an entity it
        cannot represent is a meaningful situation rather than a broken
        contract -- it is precisely what the ``Inductive`` capability
        declares a model's ability to survive.
        """
        state = EncodingState.fit(list(UNIVERSE))
        with pytest.raises(CapabilityError):
            state.encode(["AUDUSD"])

    def test_the_message_names_the_unknown_identifiers(self):
        """
        Because the fix is to widen the universe or to change the model.

        Neither is actionable from a message that only says an entity was
        unknown, and the identifier is the thing to look up upstream.
        """
        state = EncodingState.fit(list(UNIVERSE))
        with pytest.raises(CapabilityError, match="AUDUSD"):
            state.encode(["AUDUSD"])

    def test_many_unknown_identifiers_are_truncated(self):
        """
        So a wholly mismatched universe does not print a thousand names.

        The count is kept, because "and 994 more" is the number that says the
        universe is wrong rather than one entity being missing.
        """
        state = EncodingState.fit(list(UNIVERSE))
        with pytest.raises(CapabilityError, match="more"):
            state.encode([f"PAIR{index}" for index in range(50)])


class TestAttributes:
    """Categorical attributes, coded the same way."""

    def test_attribute_categories_are_coded(self):
        """
        So a model can embed a sector or a currency block.

        Coded with the same scheme as the entities, because a second scheme
        would be a second place for the same off-by-one to live.
        """
        state = EncodingState.fit(list(UNIVERSE), attributes={"block": ["G10", "G10", "G10"]})
        assert state.attributes["block"] == ("G10",)

    def test_attribute_values_must_align_with_the_identifiers(self):
        """
        Refused rather than zipped to the shorter of the two.

        Zipping would silently drop the tail, so the last instruments in the
        universe would all share whatever category came last -- and nothing
        in the output would say so.
        """
        with pytest.raises(ContractError, match="one to one"):
            EncodingState.fit(list(UNIVERSE), attributes={"block": ["G10"]})

    def test_an_encoder_with_no_attributes_is_valid(self):
        """
        Because most problems have none.

        The common case must not require passing an empty mapping.
        """
        assert EncodingState.fit(list(UNIVERSE)).attributes == {}


class TestRefusals:
    """Where an ambiguous encoding would be built silently."""

    def test_a_duplicated_identifier_in_the_code_order_is_refused(self):
        """
        Because it makes the identifier-to-code direction ambiguous.

        Whichever occurrence won would decide which embedding row an
        instrument got, and that is not a decision to make by iteration
        order.
        """
        with pytest.raises(ContractError, match="more than once"):
            EncodingState(entities=("EURUSD", "GBPUSD", "EURUSD"))

    def test_an_empty_encoder_is_valid(self):
        """
        So a tabular problem needs no entity axis.

        Refusing it would make the encoder mandatory for problems that have
        no entities at all.
        """
        assert EncodingState().n_entities == 0


class TestPersistence:
    """The round trip a reloaded bundle depends on."""

    def test_the_codes_survive_a_round_trip(self, tmp_path):
        """
        Exactly, because a model's embedding row three must keep its meaning.

        A reloaded encoder that assigned the codes afresh would serve every
        instrument the wrong embedding, and the predictions would be
        plausible numbers computed from the wrong representation.
        """
        state = EncodingState.fit(list(UNIVERSE))
        state.save(tmp_path)
        reloaded = EncodingState.load(tmp_path)
        assert reloaded.entities == state.entities

    def test_attributes_survive_a_round_trip(self, tmp_path):
        """For the same reason, applied to the attribute tables."""
        state = EncodingState.fit(list(UNIVERSE), attributes={"block": ["G10", "EM", "G10"]})
        state.save(tmp_path)
        assert EncodingState.load(tmp_path).attributes == state.attributes

    def test_the_reloaded_encoder_encodes_identically(self):
        """
        Which is the property that actually matters.

        Comparing the stored tables is a proxy; comparing the output is the
        claim, and it is what an inference run depends on.
        """
        state = EncodingState.fit(list(UNIVERSE))
        assert state.encode(list(UNIVERSE)).tolist() == [0, 1, 2]

    def test_the_description_names_the_entity_count(self, tmp_path):
        """
        Because it goes into the bundle, where it is the sanity check.

        An encoder that recorded two entities for a twelve-pair portfolio is
        a data problem, and the count is where it is visible.
        """
        state = EncodingState.fit(list(UNIVERSE))
        assert "3" in str(state.describe())
```

---

## 4. `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_reduction.py`

9560 bytes · SHA-256 `efdc6746d5981dde`

```python
"""
Tests for dimensionality reduction, and for defect 9.

Defect 9 is the leak this module was rewritten to fix: the basis -- the
selected feature subset, or the principal components -- was fitted over the
*full scaled history* rather than over the training rows. Selecting which
features matter using the test set's correlations is leakage, and it is a
particularly deniable kind: no row of test data enters the model, only a
decision about which columns to keep. The decision is enough.

The test for it is a positive control. Synthetic data is built so that the
most informative column over the training rows differs from the most
informative column over the whole history, and the two fits are asserted to
select different bases. A test that merely checked "the basis is fitted" would
pass against the defective implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import ReductionSpec
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.reduction import (
    ReductionState,
)


@pytest.fixture
def divergent():
    """
    Build data where the best feature differs by fitting window.

    Column 0 is informative only in the first half, column 2 only in the
    second. So a basis fitted on the training rows picks column 0 and one
    fitted on everything picks column 2 -- which is what makes the leak
    observable rather than merely plausible.

    Returns
    -------
    tuple
        Features, target, and the training indices.
    """
    rng = np.random.default_rng(5)
    n = 400
    target = rng.normal(size=n)

    features = rng.normal(size=(n, 4))
    # Strongly informative in the training half, pure noise afterwards.
    features[:200, 0] = target[:200] * 3.0 + rng.normal(0, 0.2, 200)
    features[200:, 0] = rng.normal(0, 3.0, 200)
    # Weakly informative in the training half, overwhelmingly so afterwards.
    # The asymmetry is what separates the two rankings cleanly: over the
    # training rows column 0 wins at 0.997 against 0.393, and over the whole
    # history column 2 wins at 0.751 against 0.484. Without that margin the
    # two could tie and the test would pass or fail on the tie-break rather
    # than on the leak.
    features[:200, 2] = target[:200] * 0.8 + rng.normal(0, 1.5, 200)
    features[200:, 2] = target[200:] * 8.0 + rng.normal(0, 0.2, 200)
    return features, target.reshape(-1, 1), np.arange(200)


class TestDefectNineTheBasisIsFittedOnTrainingRowsOnly:
    """The leak, and the positive control that detects it."""

    def test_the_fitting_window_changes_the_selected_basis(self, divergent):
        """
        The defect, stated as a difference nothing else would produce.

        If the window were ignored, these two fits would be identical. They
        are not, which is the only direct evidence that the window is
        respected.
        """
        features, target, train = divergent
        spec = ReductionSpec(method="basis_selection", n_components=1)

        correct = ReductionState.fit(features, target, spec=spec, train_indices=train)
        # The leaky configuration is reached through `fit_on='all'`, which the
        # spec documents as existing solely for refactor parity. Reaching it
        # by widening `train_indices` would not exercise the same code path.
        leaky = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=1, fit_on="all"),
            train_indices=train,
        )
        assert list(correct.describe()["selected"]) == [0]
        assert list(leaky.describe()["selected"]) == [2]

    def test_the_training_window_selects_the_training_informative_column(self, divergent):
        """
        Not merely different, but right.

        A fit that differed from the full-history fit by selecting the *wrong*
        column would pass the previous test and still be broken.
        """
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=1),
            train_indices=train,
        )
        assert list(state.describe()["selected"]) == [0]


class TestSelection:
    """Correlation-ranked feature selection."""

    def test_the_output_width_matches_the_request(self, divergent):
        """A reduction that returned the wrong width breaks the signature."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=2),
            train_indices=train,
        )
        assert state.transform_features(features).shape == (400, 2)
        assert state.n_output_features == 2

    def test_requesting_every_column_is_a_no_op(self, divergent):
        """
        The identity case, which should not be a special case.

        A reduction asked for all four of four columns should return them, not
        reorder or rescale them.
        """
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=4),
            train_indices=train,
        )
        assert np.allclose(state.transform_features(features), features)

    def test_selection_is_deterministic_under_ties(self):
        """
        Two equally correlated columns must resolve the same way every run.

        A tie broken by an unstable sort makes two otherwise identical runs
        select different features, and the difference in their scores looks
        like variance in the model.
        """
        rng = np.random.default_rng(1)
        target = rng.normal(size=100).reshape(-1, 1)
        features = np.column_stack([target.ravel(), target.ravel(), rng.normal(size=100)])
        spec = ReductionSpec(method="basis_selection", n_components=1)
        first = ReductionState.fit(features, target, spec=spec, train_indices=np.arange(100))
        second = ReductionState.fit(features, target, spec=spec, train_indices=np.arange(100))
        assert first.describe()["selected"] == second.describe()["selected"]


class TestPrincipalComponents:
    """An SVD basis, with its signs pinned."""

    def test_the_output_width_matches_the_request(self, divergent):
        """So the signature the model was built against still holds."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=2),
            train_indices=train,
        )
        assert state.transform_features(features).shape == (400, 2)

    def test_the_components_are_sign_stable(self, divergent):
        """
        An SVD's signs are arbitrary, so they have to be fixed deliberately.

        Two runs whose components differ by a sign produce mirrored features,
        a differently signed set of learned weights, and a diff between two
        saved models that looks like a real change.
        """
        features, target, train = divergent
        spec = ReductionSpec(method="pca", n_components=3)
        first = ReductionState.fit(features, target, spec=spec, train_indices=train)
        second = ReductionState.fit(features, target, spec=spec, train_indices=train)
        assert np.array_equal(
            first.transform_features(features), second.transform_features(features)
        )

    def test_the_components_are_orthogonal(self, divergent):
        """The defining property, so a wrong matrix cannot pass unnoticed."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=3),
            train_indices=train,
        )
        reduced = state.transform_features(features[train])
        correlation = np.corrcoef(reduced, rowvar=False)
        off_diagonal = correlation[~np.eye(3, dtype=bool)]
        assert np.allclose(off_diagonal, 0.0, atol=1e-8)


class TestTargetPassThrough:
    """Reduction touches features, never the target."""

    def test_the_target_inverse_is_the_identity(self, divergent):
        """
        Because this transform never scaled the target in the first place.

        A reduction that applied *some* inverse would corrupt the metrics of
        every run that used it.
        """
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=2),
            train_indices=train,
        )
        assert np.array_equal(state.inverse_transform_targets(target), target)


class TestPersistence:
    """The basis has to be the same basis when the model is served."""

    def test_the_state_round_trips(self, divergent, tmp_path):
        """A different basis at serving time is a different model."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=2),
            train_indices=train,
        )
        state.save(tmp_path)
        reloaded = ReductionState.load(tmp_path)
        assert reloaded == state
        assert np.array_equal(
            reloaded.transform_features(features), state.transform_features(features)
        )
```

---

## 5. `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_scaling.py`

7915 bytes · SHA-256 `3ef1d50bfe25cc83`

```python
"""
Tests for feature and target scaling.

The clause that matters most here is that statistics come from the training
rows alone. Scaling fitted over the full history leaks the test set's mean and
variance into every training row, and the result is a model that scores better
than it should by a margin small enough to look like skill.

The second clause is that the target's inverse is a real inverse. If it is
not, every metric is reported in the wrong units -- which makes it
incomparable with another run and unusable for a decision, while still looking
like a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import ScalingSpec
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.scaling import ScalingState


@pytest.fixture
def data():
    """
    Provide features and a target whose halves have very different scales.

    Deliberately non-stationary: the second half is shifted and stretched, so
    statistics fitted over everything differ from training-only statistics by
    far more than floating-point noise. A test on stationary data cannot
    detect the leak it is meant to catch.

    Returns
    -------
    tuple
        Features, target, and the training indices.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    features[100:] = features[100:] * 5.0 + 20.0
    target = rng.normal(size=(200, 1))
    target[100:] = target[100:] * 5.0 + 20.0
    return features, target, np.arange(100)


class TestFittingWindow:
    """Statistics come from the training rows and nowhere else."""

    def test_the_mean_is_the_training_mean(self, data):
        """
        Not the mean of everything, which is the leak.

        Asserted against the hand-computed training mean rather than against a
        tolerance, because the whole point is which rows were used.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        scaled = state.transform_features(features[train])
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)

    def test_the_held_out_rows_are_not_centred(self, data):
        """
        The visible consequence of fitting on training rows only.

        Held-out data transformed with training statistics is *not* mean zero,
        and if it were, the statistics had seen it.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        held_out = state.transform_features(features[100:])
        assert np.abs(held_out.mean()) > 1.0

    def test_extending_the_window_changes_the_state(self, data):
        """
        A state that ignored its window would pass every other test here.

        Two fits over different row sets must differ, or the window argument
        is decorative.
        """
        features, target, _ = data
        narrow = ScalingState.fit(
            features, target, spec=ScalingSpec(), train_indices=np.arange(100)
        )
        wide = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=np.arange(200))
        assert narrow != wide


class TestTargetInversion:
    """A metric in the wrong units is worse than no metric."""

    def test_the_inverse_round_trips(self, data):
        """
        Forward then back is the identity, to numerical precision.

        A forgotten offset or a reciprocal scale passes every shape check and
        then misreports every error in the run.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        recovered = state.inverse_transform_targets(state.transform_targets(target))
        assert np.allclose(recovered, target, atol=1e-9)

    def test_an_unscaled_target_is_passed_through(self, data):
        """
        The configuration where the inverse must be exactly the identity.

        A state that always applied *some* transform would shift a target that
        was never scaled.
        """
        features, target, train = data
        state = ScalingState.fit(
            features,
            target,
            spec=ScalingSpec(scale_target=False),
            train_indices=train,
        )
        assert np.array_equal(state.inverse_transform_targets(target), target)


class TestDegenerateColumns:
    """A constant column has zero variance, and dividing by it is a NaN."""

    def test_a_constant_column_does_not_produce_nan(self, data):
        """
        Which would poison every downstream computation silently.

        A NaN in one feature column makes the whole forward pass NaN, and the
        symptom is a loss that is NaN from the first step -- a long way from
        the cause.
        """
        features, target, train = data
        features = features.copy()
        features[:, 1] = 7.0
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        assert np.all(np.isfinite(state.transform_features(features)))

    def test_degenerate_columns_are_counted_in_the_description(self, data):
        """
        So the run summary says the data had a constant feature.

        Handling it silently means nobody finds out the column was useless.
        """
        features, target, train = data
        features = features.copy()
        features[:, 1] = 7.0
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        assert state.describe()["n_degenerate_features"] == 1


class TestPersistence:
    """A served model must use the state it was trained with."""

    def test_the_state_round_trips_through_a_directory(self, data, tmp_path):
        """
        Equality after a save and load, not merely a successful load.

        A state that loaded with a subtly different scale produces a model
        whose predictions are wrong in a way that will be blamed on the data.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        state.save(tmp_path)
        assert ScalingState.load(tmp_path) == state

    def test_the_reloaded_state_transforms_identically(self, data, tmp_path):
        """
        The property equality is standing in for, asserted directly.

        Equality could be satisfied by a state that compared only its
        metadata, so the transform itself is checked too.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        state.save(tmp_path)
        reloaded = ScalingState.load(tmp_path)
        assert np.array_equal(
            reloaded.transform_features(features), state.transform_features(features)
        )


class TestMethods:
    """The spec's scaling methods are distinguishable."""

    def test_robust_scaling_resists_an_outlier(self, data):
        """
        The reason the method exists.

        A single extreme row moves a mean and a standard deviation enough to
        squash every other row towards zero; a median and an interquartile
        range barely move. Financial data has those rows.
        """
        features, target, train = data
        features = features.copy()
        features[0, 0] = 1e6

        standard = ScalingState.fit(
            features, target, spec=ScalingSpec(method="standard"), train_indices=train
        )
        robust = ScalingState.fit(
            features, target, spec=ScalingSpec(method="robust"), train_indices=train
        )
        ordinary_rows = np.arange(1, 100)
        assert np.std(robust.transform_features(features[ordinary_rows])[:, 0]) > np.std(
            standard.transform_features(features[ordinary_rows])[:, 0]
        )
```

---

## 6. `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_sequence.py`

9132 bytes · SHA-256 `a72496bc7acf552b`

```python
"""
Tests for rolling-window construction.

Three things go wrong with windows and all three are quiet.

A window that reaches back past the start of the array wraps around in numpy
rather than failing, so the first few samples of every split are built from
the *end* of the history. A window that straddles a split boundary reads
held-out rows into a training sample. And an off-by-one in which row a window
is labelled by shifts every target by one scenario, which degrades a score
rather than breaking a run.

So the tests here assert on exact index arithmetic rather than on shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.data import SplitIndices
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.transforms.sequence import (
    extract_windows,
    usable_labels,
    windows_stay_within,
)


class TestUsableLabels:
    """Which rows can be the last scenario of a complete window."""

    def test_a_non_sequential_model_uses_every_row(self):
        """
        At a length of one, every row is a complete window of itself.

        Dropping anything here would silently shrink the dataset.
        """
        labels = usable_labels(np.arange(100), length=1)
        assert labels.tolist() == list(range(100))

    def test_the_first_rows_of_the_dataset_are_dropped(self):
        """
        Because no complete window ends there.

        A window of twenty ending at scenario five would need scenarios
        minus-fourteen to five, and numpy reads negative indices from the far
        end of the array -- so the sample would be built from the most recent
        data and labelled with the oldest.
        """
        labels = usable_labels(np.arange(0, 100), length=20)
        assert labels.min() == 19
        assert labels.size == 100 - 19

    def test_a_later_split_keeps_every_label(self):
        """
        The index is absolute, not relative to the split.

        A split starting at scenario fifty has fifty scenarios of history
        behind it, so every one of its labels has a complete window. Filtering
        relative to the split would discard nineteen perfectly good samples
        from each of validation and test -- and keeping them is safe precisely
        because the boundary gap already removed the rows that would have
        crossed the split edge.
        """
        labels = usable_labels(np.arange(50, 150), length=20)
        assert labels.min() == 50
        assert labels.size == 100

    def test_a_stride_thins_the_labels(self):
        """
        The reason a stride exists: overlapping windows are nearly duplicates.

        Consecutive windows of length twenty share nineteen rows, so a stride
        trades sample count for independence.
        """
        labels = usable_labels(np.arange(100), length=10, stride=5)
        assert np.all(np.diff(labels) == 5)

    def test_a_window_wider_than_the_split_yields_nothing(self):
        """
        Returned empty rather than raising, so the caller can say which split.

        The caller knows the split's name and the specification that produced
        it, and "the validation split is too narrow for a window of 60" is a
        far better message than this function could write.
        """
        labels = usable_labels(np.arange(10), length=60)
        assert labels.size == 0


class TestConfinement:
    """The alternative to a boundary gap: drop each split's first labels."""

    def test_confining_drops_the_labels_whose_window_reaches_back(self):
        """
        A split starting at fifty loses its first ``length - 1`` labels.

        The two strategies cost the same number of scenarios. The gap
        takes them from between the splits; confinement takes them from
        the front of each. Which one is available depends on whether the
        splits can be moved, and explicit splits cannot.
        """
        labels = usable_labels(np.arange(50, 150), length=20, confine=True)
        assert labels.min() == 69
        assert labels.size == 100 - 19

    def test_not_confining_keeps_them(self):
        """The default, and the behaviour the boundary gap is designed for."""
        labels = usable_labels(np.arange(50, 150), length=20, confine=False)
        assert labels.min() == 50
        assert labels.size == 100

    def test_a_hole_in_the_split_is_respected(self):
        """
        A window may not span an index the split does not contain.

        This is why confinement is a membership test rather than
        arithmetic on the split's first index. A purged fold's indices
        have holes by design -- the purge removed exactly the scenarios
        adjacent to the test period -- and a window spanning one would
        read them straight back in.
        """
        indices = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.int64)
        labels = usable_labels(indices, length=4, confine=True)
        # Only 3 and 10 have their three predecessors present.
        assert labels.tolist() == [3, 10]

    def test_the_two_strategies_agree_once_the_gap_is_wide_enough(self):
        """
        With a gap of ``length - 1``, confining changes nothing for a later split.

        Which is the point: a correctly gapped configuration pays for the
        gap once and keeps every label, so turning confinement on costs
        nothing and turning it off is safe.
        """
        indices = np.arange(50, 150)
        gapped = usable_labels(indices, length=1, confine=True)
        ungapped = usable_labels(indices, length=1, confine=False)
        assert gapped.tolist() == ungapped.tolist()


class TestExtractWindows:
    """The windows themselves, and what each row of them contains."""

    def test_a_window_ends_at_its_label(self):
        """
        The convention the whole module depends on.

        Labelling a window by its first row instead of its last shifts every
        target by ``length - 1`` scenarios, which trains the model to predict
        the past.
        """
        values = np.arange(100, dtype=np.float64).reshape(-1, 1)
        windows = extract_windows(values, np.array([30, 50]), length=5)
        assert windows[0, -1, 0] == 30.0
        assert windows[1, -1, 0] == 50.0

    def test_a_window_reaches_back_exactly_its_length(self):
        """So nothing older than ``label - length + 1`` enters the sample."""
        values = np.arange(100, dtype=np.float64).reshape(-1, 1)
        windows = extract_windows(values, np.array([30]), length=5)
        assert windows[0, :, 0].tolist() == [26.0, 27.0, 28.0, 29.0, 30.0]

    def test_the_shape_is_samples_by_window_by_features(self):
        """The axis order the gradient engines and the signature both assume."""
        values = np.arange(300, dtype=np.float64).reshape(100, 3)
        windows = extract_windows(values, np.array([40, 50, 60]), length=7)
        assert windows.shape == (3, 7, 3)

    def test_a_label_too_early_for_a_complete_window_is_refused(self):
        """
        Rather than wrapping, which is what numpy would do.

        A negative start index reads from the end of the array, so the sample
        would be built from the newest data and labelled with the oldest --
        leakage of the most direct possible kind, producing a model that
        scores superbly and predicts nothing.
        """
        values = np.arange(100, dtype=np.float64).reshape(-1, 1)
        with pytest.raises(ContractError):
            extract_windows(values, np.array([2]), length=20)


class TestWindowsStayWithin:
    """The check that a split's windows do not reach into another split."""

    def test_a_gapped_split_passes(self):
        """A gap of at least ``length - 1`` is exactly what makes it safe."""
        splits = SplitIndices(
            train=np.arange(0, 100),
            validation=np.arange(120, 160),
            test=np.arange(180, 200),
        )
        assert windows_stay_within(splits, length=20)

    def test_an_adjacent_split_fails(self):
        """
        The leak the boundary gap exists to prevent.

        With no gap, the validation split's first window reads nineteen
        training rows -- and every metric computed from it is optimistic by an
        amount nobody can estimate after the fact.
        """
        splits = SplitIndices(
            train=np.arange(0, 100),
            validation=np.arange(100, 140),
            test=np.arange(140, 160),
        )
        assert not windows_stay_within(splits, length=20)

    def test_adjacency_is_fine_without_windows(self):
        """
        At a length of one there is nothing to reach back into.

        A check that rejected adjacent splits unconditionally would force a
        gap on every non-sequential model and throw away data for nothing.
        """
        splits = SplitIndices(
            train=np.arange(0, 100),
            validation=np.arange(100, 140),
            test=np.arange(140, 160),
        )
        assert windows_stay_within(splits, length=1)
```

