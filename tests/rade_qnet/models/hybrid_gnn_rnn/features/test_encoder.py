"""
Tests for the flagship's attribute encoder.

The encoder is fitted along the entity axis, which is the one place in the
model where fitting over data outside the training split is correct rather
than leakage. Several tests below exist to pin that down, because it is
exactly the kind of thing a later reader "fixes".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.features.encoder import (
    EntityEncoderState,
    decay_lambdas,
)
from src.rade_qnet.models.hybrid_gnn_rnn.spec import AttributeEncoderSpec


@pytest.fixture
def state(attributes: Mapping[str, Sequence[Any]]) -> EntityEncoderState:
    """Fit the encoder on the shared attribute fixture."""
    return EntityEncoderState.fit(attributes, spec=AttributeEncoderSpec())


class TestDecayLambdas:
    """The maturity decay rates."""

    def test_the_rates_span_fast_to_slow(self) -> None:
        """
        A single rate can only express one horizon.

        The point of the decay block is to let the network distinguish a
        one-month instrument from a five-year one; a set of rates spanning
        fast to slow gives it a basis to do that in, where one rate gives it
        a single monotone transform of maturity it already has.
        """
        lambdas = decay_lambdas(3)
        assert lambdas == tuple(sorted(lambdas, reverse=True))
        assert lambdas[0] > lambdas[-1]

    def test_zero_terms_is_allowed(self) -> None:
        """A caller that does not want the decay block should be able to say so."""
        assert decay_lambdas(0) == ()

    def test_one_term_does_not_divide_by_zero(self) -> None:
        """
        A single rate is the degenerate case of a linear span.

        Worth its own test because the obvious implementation divides by
        ``n_terms - 1`` and raises here.
        """
        assert len(decay_lambdas(1)) == 1


class TestFitting:
    """What the fit stage records."""

    def test_the_statistics_are_held_in_float64(self, state: EntityEncoderState) -> None:
        """
        Accumulating a mean in float32 loses digits the transform then bakes in.

        The working precision applies to the transform; the statistics are
        always accumulated in float64 regardless, because a mean over a
        thousand instruments in float32 is measurably wrong and nothing
        downstream can recover it.
        """
        assert state.numeric_centre.dtype == np.float64
        assert state.numeric_scale.dtype == np.float64

    def test_a_constant_attribute_does_not_divide_by_zero(
        self, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        A zero-variance attribute has a scale of zero.

        Not hypothetical: an all-at-the-money book has constant moneyness.
        Dividing by it would make every feature NaN and the failure would
        surface as a loss of NaN many stages later.
        """
        constant = {**attributes, "delta": [0.5] * 6}
        fitted = EntityEncoderState.fit(constant, spec=AttributeEncoderSpec())
        encoded = fitted.transform(constant)
        assert np.isfinite(encoded.features).all()

    def test_the_levels_are_sorted(self, state: EntityEncoderState) -> None:
        """
        Column order must not depend on which instrument happened to come first.

        If the levels were recorded in encounter order, re-fitting on a
        reordered universe would produce the same columns in a different
        order, and a saved model would silently mis-read every one-hot
        block.
        """
        for levels in state.categorical_levels.values():
            assert list(levels) == sorted(levels)
        for levels in state.multi_label_levels.values():
            assert list(levels) == sorted(levels)

    def test_fitting_cannot_be_restricted_to_a_subset(self) -> None:
        """
        There is no parameter that would let a caller fit on training rows only.

        This is the structural half of the leakage argument. The scenario
        axis must be fitted on training rows alone; the entity axis must not
        be, because which instruments exist is known up front and a
        universe-sized encoding is what the graph needs. Making that
        impossible to get wrong beats documenting it.
        """
        parameters = set(EntityEncoderState.fit.__func__.__code__.co_varnames)
        assert not parameters & {"train_idx", "split", "mask", "subset", "fit_on"}

    def test_a_missing_attribute_is_refused(self, attributes: Mapping[str, Sequence[Any]]) -> None:
        """
        A silently absent attribute would produce a narrower matrix.

        That matrix would train, and the only symptom would be worse
        predictions, so it must be an error rather than a warning.
        """
        incomplete = {k: v for k, v in attributes.items() if k != "vega"}
        with pytest.raises(ContractError, match="vega"):
            EntityEncoderState.fit(incomplete, spec=AttributeEncoderSpec())


class TestTransform:
    """What the transform stage produces."""

    def test_the_columns_are_named_and_counted_consistently(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """A name per column, and the advertised width."""
        encoded = state.transform(attributes)
        assert encoded.features.shape == (6, len(encoded.names))
        assert encoded.features.shape[1] == state.n_features

    def test_the_blocks_tile_the_matrix_without_gaps(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        The graph slices columns by block, so the spans must cover exactly.

        A gap would silently drop an attribute from the distance; an overlap
        would double-count one.
        """
        encoded = state.transform(attributes)
        spans = sorted(encoded.blocks.values())
        assert spans[0][0] == 0
        assert spans[-1][1] == encoded.features.shape[1]
        for (_, end), (start, _) in pairwise(spans):
            assert end == start

    def test_the_scaled_numeric_columns_are_standardised(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """Zero mean and unit scale, which is the whole claim of the stage."""
        encoded = state.transform(attributes)
        start, end = encoded.blocks["moneyness"]
        column = encoded.features[:, start:end]
        assert column.mean() == pytest.approx(0.0, abs=1e-6)
        assert column.std() == pytest.approx(1.0, abs=1e-6)

    def test_the_one_hot_rows_select_exactly_one_level(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """A categorical attribute takes one value, so its block sums to one."""
        encoded = state.transform(attributes)
        start, end = encoded.blocks["product_type"]
        assert np.array_equal(
            encoded.features[:, start:end].sum(axis=1), np.ones(6, dtype=np.float32)
        )

    def test_the_multi_label_rows_are_unit_norm(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        An instrument exposed to two risk factors must not out-weigh one exposed to one.

        Without the normalisation, label count alone would drive the
        distance and the graph would cluster by breadth of exposure rather
        than by similarity.
        """
        encoded = state.transform(attributes)
        start, end = encoded.blocks["underlying_risk_factors"]
        norms = np.linalg.norm(encoded.features[:, start:end], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_an_unseen_categorical_level_encodes_as_all_zero(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        Inference will meet a product type the fit never saw.

        Raising would make the model unusable on a new book; inventing a
        column would change the matrix width and break the saved weights.
        An all-zero row in that block is the only option that keeps the
        signature stable, and it is the honest encoding of "none of the
        known levels".
        """
        unseen = {**attributes, "product_type": ["swaption"] * 6}
        encoded = state.transform(unseen)
        start, end = encoded.blocks["product_type"]
        assert not encoded.features[:, start:end].any()

    def test_the_width_does_not_depend_on_the_rows_transformed(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        Transforming one instrument must give the same columns as transforming six.

        Inference scores a handful of instruments at a time, so a width that
        tracked the batch would make the saved weights unusable.
        """
        single = {key: list(values)[:1] for key, values in attributes.items()}
        assert state.transform(single).names == state.transform(attributes).names


class TestPrecision:
    """The ``numeric_precision`` compatibility flag."""

    def test_the_default_is_the_accurate_one(self) -> None:
        """
        A flag whose default reproduces a known-worse computation is a trap.

        Every compatibility flag in this model defaults to the correct
        behaviour, so that forgetting to set one fails safe.
        """
        assert AttributeEncoderSpec().numeric_precision == "float64"

    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_both_precisions_produce_a_float32_matrix(
        self, attributes: Mapping[str, Sequence[Any]], precision: str
    ) -> None:
        """
        The flag governs the arithmetic, not the output dtype.

        The network consumes float32 either way; what differs is whether the
        subtraction and division happen in float32 before being stored.
        """
        spec = AttributeEncoderSpec(numeric_precision=precision)  # type: ignore[arg-type]
        encoded = EntityEncoderState.fit(attributes, spec=spec).transform(attributes)
        assert encoded.features.dtype == np.float32

    def test_the_flag_actually_changes_the_arithmetic(
        self, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        A flag that changed nothing would make the parity tests meaningless.

        They would pass with it set and pass with it unset, and would
        therefore not be evidence of anything.
        """
        results = {
            precision: EntityEncoderState.fit(
                attributes,
                spec=AttributeEncoderSpec(numeric_precision=precision),  # type: ignore[arg-type]
            )
            .transform(attributes)
            .features
            for precision in ("float64", "float32")
        }
        assert not np.array_equal(results["float64"], results["float32"])
        # Different in the last bit only. A larger gap would mean the flag
        # is doing something beyond rounding, which is not what it claims.
        assert np.abs(results["float64"] - results["float32"]).max() < 1e-5


class TestRoundTrip:
    """Saving and loading."""

    def test_a_reloaded_state_transforms_identically(
        self, state: EntityEncoderState, attributes: Mapping[str, Sequence[Any]], tmp_path
    ) -> None:
        """
        The state is only useful at inference if it survives the round trip.

        Column *order* is checked before the values, and separately, because
        that is the half that fails quietly. A reordered block still gives a
        matrix of the right shape and plausible numbers, so the saved
        weights are applied to the wrong columns and the only symptom is bad
        predictions.
        """
        state.save(tmp_path)
        restored = EntityEncoderState.load(tmp_path)

        before, after = state.transform(attributes), restored.transform(attributes)
        assert after.names == before.names
        assert dict(after.blocks) == dict(before.blocks)
        assert np.array_equal(after.features, before.features)

    def test_nothing_is_pickled(self, state: EntityEncoderState, tmp_path) -> None:
        """
        A pickled estimator ties the saved model to a library version.

        The original stored fitted scikit-learn objects, so loading a model
        a year later needed the same scikit-learn. Hand-rolling the
        encoders buys plain arrays and JSON, and this test is what keeps
        that property from quietly regressing.
        """
        state.save(tmp_path)
        suffixes = {path.suffix for path in tmp_path.rglob("*") if path.is_file()}
        assert suffixes <= {".npy", ".npz", ".json"}
