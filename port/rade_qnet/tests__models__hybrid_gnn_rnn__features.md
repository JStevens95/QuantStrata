# `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 60 | `7cd332b12fc0f4a5` |
| 2 | `conftest.py` | 74 | 2170 | `c03a6d739771af29` |
| 3 | `test_basis.py` | 226 | 9395 | `8591510cb75cc9e8` |
| 4 | `test_encoder.py` | 308 | 12972 | `b9189e00a51b762a` |
| 5 | `test_graph.py` | 353 | 14417 | `23f7649ad5faa458` |

---

## 1. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/__init__.py`

60 bytes · SHA-256 `7cd332b12fc0f4a5`

```python
"""Mirrors ``rade_qnet.models.hybrid_gnn_rnn.features``."""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/conftest.py`

2170 bytes · SHA-256 `c03a6d739771af29`

```python
"""
Attribute fixtures shared by the encoder and graph tests.

Deliberately hand-written rather than generated: the encoder's output
depends on which levels a categorical attribute takes and on how many
labels a multi-label attribute carries, so a test that asserts on column
layout needs those to be visible in the file rather than hidden behind a
random draw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest


@pytest.fixture
def attributes() -> Mapping[str, Sequence[Any]]:
    """
    Six instruments: four elementary, two target.

    Chosen so that every branch of the encoder is exercised at least once.
    ``product_subtype`` takes two levels and ``product_type`` takes two, so
    a one-hot block of width one -- which is a degenerate case the width
    normalisation would divide by -- is not the only thing under test. The
    risk-factor sets have different cardinalities, so the per-row norm has
    something to do.

    Returns
    -------
    Mapping
        Attribute name to one value per instrument.
    """
    return {
        "trade_id": ["E1", "E2", "E3", "E4", "T1", "T2"],
        "moneyness": [0.95, 1.00, 1.05, 1.10, 0.98, 1.02],
        "yrs_to_maturity": [0.25, 0.50, 1.00, 2.00, 0.75, 1.50],
        "delta": [0.30, 0.45, 0.55, 0.70, 0.40, 0.60],
        "vega": [10.0, 12.5, 15.0, 20.0, 11.0, 18.0],
        "product_type": [
            "vanilla_option",
            "vanilla_option",
            "forward",
            "forward",
            "vanilla_option",
            "forward",
        ],
        "product_subtype": [
            "european",
            "european",
            "outright",
            "outright",
            "european",
            "outright",
        ],
        "trade_type": [
            "elementary",
            "elementary",
            "elementary",
            "elementary",
            "target",
            "target",
        ],
        "underlying_risk_factors": [
            ["FX"],
            ["FX", "RATES"],
            ["RATES"],
            ["FX"],
            ["FX", "RATES"],
            ["RATES"],
        ],
    }
```

---

## 3. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/test_basis.py`

9395 bytes · SHA-256 `8591510cb75cc9e8`

```python
"""
Tests for basis selection.

The property under test is not "it reduces" -- anything reduces. It is that
the instruments kept *span* the ones dropped, and that the budget is spread
across the book rather than consumed by whichever group happens to be
loudest.
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.basis import (
    effective_rank,
    select_basis,
)

N_SCENARIOS = 200


def redundant_book(
    *, n_groups: int = 2, per_group: int = 5, factors: int = 2, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """
    Build a book with fewer underlying drivers than instruments.

    Each group's P&L is a random mix of ``factors`` independent drivers, so
    the group's true dimension is ``factors`` however many instruments it
    holds. That makes the correct answer known in advance, which is what
    lets the tests below assert on it.

    Parameters
    ----------
    n_groups
        How many underlyings.
    per_group
        Instruments per underlying.
    factors
        Independent drivers per group.
    seed
        Generator seed.

    Returns
    -------
    tuple
        The P&L matrix and its instrument identifiers.
    """
    generator = np.random.default_rng(seed)
    blocks, identifiers = [], []
    for group in range(n_groups):
        drivers = generator.normal(size=(N_SCENARIOS, factors))
        loadings = generator.normal(size=(factors, per_group))
        blocks.append(drivers @ loadings)
        identifiers += [f"CCY{group}|forward|{number:03d}" for number in range(per_group)]
    return np.hstack(blocks), identifiers


class TestEffectiveRank:
    """How many components the variance threshold asks for."""

    def test_it_finds_the_true_dimension(self) -> None:
        """
        Three drivers behind ten instruments is a rank of three.

        The whole premise of the reduction is that this number is much
        smaller than the instrument count; a method that could not recover
        it on noiseless data would not be worth running on real data.
        """
        generator = np.random.default_rng(0)
        values = generator.normal(size=(N_SCENARIOS, 3)) @ generator.normal(size=(3, 10))
        assert effective_rank(values, variance_threshold=0.999) == 3

    def test_a_higher_threshold_keeps_at_least_as_many(self) -> None:
        """Monotonicity. A threshold that kept fewer would be unusable as a knob."""
        values, _ = redundant_book(n_groups=1, per_group=8, factors=4)
        ranks = [effective_rank(values, variance_threshold=t) for t in (0.5, 0.9, 0.99)]
        assert ranks == sorted(ranks)

    def test_a_single_instrument_is_its_own_basis(self) -> None:
        """
        A one-instrument group is its own basis.

        The degenerate case, which the SVD would otherwise be asked to
        handle on a one-column matrix.
        """
        assert effective_rank(np.arange(10.0).reshape(-1, 1)) == 1

    def test_a_constant_group_still_keeps_one_instrument(self) -> None:
        """
        A group that never moved has no variance to explain.

        Returning zero would leave that corner of the book with no
        representative at all, so the model could not predict a position in
        it. One is the smallest honest answer.
        """
        assert effective_rank(np.ones((20, 4))) == 1


class TestSelection:
    """Which instruments come back."""

    def test_the_kept_instruments_span_the_dropped_ones(self) -> None:
        """
        The claim that justifies the whole stage.

        If the selected columns really span the book, then regressing a
        dropped instrument onto them leaves almost no residual. A selection
        that merely kept the most volatile instruments would fail this,
        because those tend to be several expressions of one risk.
        """
        pnl, identifiers = redundant_book(n_groups=1, per_group=6, factors=2)
        selected = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.999)
        kept = [identifiers.index(name) for name in selected]
        dropped = [i for i in range(len(identifiers)) if i not in kept]

        basis = pnl[:, kept]
        for column in dropped:
            residual = (
                pnl[:, column] - basis @ np.linalg.lstsq(basis, pnl[:, column], rcond=None)[0]
            )
            assert np.abs(residual).max() < 1e-8 * np.abs(pnl[:, column]).max() + 1e-8

    def test_every_group_keeps_a_representative(self) -> None:
        """
        No underlying may be left without a representative.

        Per-group selection is what stops a large, volatile underlying from
        consuming the whole budget. With a global selection, the loud group below would take every slot
        and the model would have nothing at all to predict a quiet-group
        position from.
        """
        quiet, quiet_ids = redundant_book(n_groups=1, per_group=4, factors=2, seed=1)
        loud, loud_ids = redundant_book(n_groups=1, per_group=4, factors=2, seed=2)
        pnl = np.hstack([quiet * 1e-4, loud * 1e4])
        identifiers = [i.replace("CCY0", "QUIET") for i in quiet_ids] + [
            i.replace("CCY0", "LOUD") for i in loud_ids
        ]

        selected = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.99)
        assert any(name.startswith("QUIET") for name in selected)
        assert any(name.startswith("LOUD") for name in selected)

    def test_the_order_does_not_depend_on_the_input_order(self) -> None:
        """
        Shuffling the input columns must not reorder the result.

        Groups are visited in sorted order. The returned order is the column order of every downstream array,
        so an order that tracked the input would make two runs on the same
        book incomparable.
        """
        pnl, identifiers = redundant_book()
        first = select_basis(pnl, instrument_ids=identifiers)

        permutation = np.random.default_rng(0).permutation(len(identifiers))
        shuffled = select_basis(
            pnl[:, permutation], instrument_ids=[identifiers[i] for i in permutation]
        )
        assert shuffled == first

    def test_the_result_is_a_sequence_not_a_set(self) -> None:
        """
        Pivot order is importance order, and it is load-bearing.

        A selection returned sorted would still be the same instruments but
        a different feature matrix, so parity level 1 compares it in order.
        """
        pnl, identifiers = redundant_book(n_groups=1, per_group=6, factors=4)
        selected = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.99)
        assert list(selected) != sorted(selected)

    def test_tail_weighting_changes_which_instrument_is_chosen(self) -> None:
        """
        Up-weighting the violent days picks a different instrument.

        The book below holds one instrument that is noisy day to day and
        one that is quiet until it is not. Unweighted, the noisy one has
        the larger variance and is chosen. Weighted towards the tail, the
        other is -- and the tail is what a risk model exists for. A knob
        that could not flip this would not be doing anything.
        """
        generator = np.random.default_rng(3)
        tail_days = generator.choice(N_SCENARIOS, 5, replace=False)

        noisy_every_day = generator.normal(scale=1.0, size=N_SCENARIOS)
        quiet_then_violent = generator.normal(scale=0.05, size=N_SCENARIOS)
        # Four sigma on five days out of two hundred. Large enough that the
        # tail detector sees those days, small enough that the instrument's
        # total variance stays below the noisy one's -- which is what makes
        # the unweighted and weighted answers differ.
        quiet_then_violent[tail_days] = 4.0

        pnl = np.column_stack([noisy_every_day, quiet_then_violent])
        identifiers = ["CCY0|forward|000", "CCY0|forward|001"]

        # A threshold of 0.5 keeps one instrument, so the choice is visible.
        unweighted = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.5)
        weighted = select_basis(
            pnl, instrument_ids=identifiers, variance_threshold=0.5, weight_tail=50.0
        )
        assert unweighted == ("CCY0|forward|000",)
        assert weighted == ("CCY0|forward|001",)

    def test_a_mismatched_identifier_count_is_refused(self) -> None:
        """
        The identifiers name the columns, so a mismatch misnames the basis.

        It would still be the right length, so nothing downstream would
        notice.
        """
        pnl, identifiers = redundant_book()
        with pytest.raises(ContractError, match="identifier"):
            select_basis(pnl, instrument_ids=identifiers[:-1])

    def test_an_unparseable_identifier_is_refused(self) -> None:
        """
        Grouping is parsed out of the identifier, so it must parse.

        An identifier without a product type would land every instrument in
        one group, silently undoing the per-group guarantee above.
        """
        pnl, identifiers = redundant_book()
        with pytest.raises(ContractError, match="does not parse"):
            select_basis(pnl, instrument_ids=["bare_name", *identifiers[1:]])
```

---

## 4. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/test_encoder.py`

12972 bytes · SHA-256 `b9189e00a51b762a`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.encoder import (
    EntityEncoderState,
    decay_lambdas,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import AttributeEncoderSpec


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
```

---

## 5. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/test_graph.py`

14417 bytes · SHA-256 `23f7649ad5faa458`

```python
"""
Tests for the flagship's instrument graph.

Two things are being defended. The first is the mathematics: rows that sum
to one, self-loops present, weights that fall with distance. The second is
the port's one deliberate difference from the baseline -- the distance is
computed more accurately than scikit-learn computed it -- and the claim
that the difference cannot change which edges exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.encoder import (
    EntityEncoderState,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.graph import (
    SparseGraphState,
    build_graph,
    weighted_features,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import (
    AttributeEncoderSpec,
    GraphSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.testkit.parity import ADJACENCY_VALUE_ATOL

N_INSTRUMENTS = 6

#: Four elementary then two target, matching the shared attribute fixture.
IS_TARGET = np.array([False, False, False, False, True, True])


@pytest.fixture
def encoded(attributes: Mapping[str, Sequence[Any]]):
    """Encode the shared attribute fixture."""
    return EntityEncoderState.fit(attributes, spec=AttributeEncoderSpec()).transform(attributes)


@pytest.fixture
def graph(encoded) -> SparseGraphState:
    """Build a three-neighbour graph over the fixture."""
    return build_graph(encoded, spec=GraphSpec(n_neighbours=3), is_target=IS_TARGET)


def densify(graph: SparseGraphState) -> np.ndarray:
    """
    Expand the edge list into a dense matrix.

    Tests read far better against a dense matrix, and at six nodes the cost
    is nothing. Production never does this.

    Parameters
    ----------
    graph
        The graph to expand.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_nodes)``.
    """
    dense = np.zeros(graph.dense_shape, dtype=np.float64)
    dense[graph.indices[:, 0], graph.indices[:, 1]] = graph.values
    return dense


class TestStructure:
    """What the edge list looks like."""

    def test_every_node_has_a_self_loop(self, graph: SparseGraphState) -> None:
        """
        Without one, message passing discards a node's own features.

        One round of aggregation would replace each instrument with the mean
        of its neighbours, so after two layers nothing of the instrument
        itself remains.
        """
        dense = densify(graph)
        assert (np.diag(dense) > 0).all()

    def test_each_node_has_the_requested_neighbours_plus_itself(
        self, graph: SparseGraphState
    ) -> None:
        """Three neighbours and one self-loop, for every node."""
        assert np.array_equal(graph.degrees(), np.full(N_INSTRUMENTS, 4))

    def test_the_edges_are_sorted_row_major(self, graph: SparseGraphState) -> None:
        """
        The order is part of the state, so it has to be deterministic.

        Two graphs with the same edges listed differently are the same
        graph but not the same array, and the comparison against the
        baseline is an array comparison.
        """
        keys = graph.indices[:, 0] * graph.n_nodes + graph.indices[:, 1]
        assert np.array_equal(keys, np.sort(keys))
        assert len(np.unique(keys)) == len(keys)

    def test_a_node_is_never_its_own_neighbour_twice(self, graph: SparseGraphState) -> None:
        """
        The self-loop is added explicitly, so the search must not find it too.

        If it did, the diagonal would be counted twice and the node would
        weight itself at roughly double, with no error anywhere.
        """
        rows, columns = graph.indices[:, 0], graph.indices[:, 1]
        assert int((rows == columns).sum()) == N_INSTRUMENTS

    def test_more_neighbours_than_instruments_is_clamped(self, encoded) -> None:
        """
        A small cluster is legitimate and must not raise.

        Asking for twenty neighbours among six instruments should give five
        plus a self-loop, not an error and not a wrapped index.
        """
        graph = build_graph(encoded, spec=GraphSpec(n_neighbours=20), is_target=IS_TARGET)
        assert np.array_equal(graph.degrees(), np.full(N_INSTRUMENTS, N_INSTRUMENTS))


class TestWeights:
    """What the edge weights mean."""

    def test_every_row_sums_to_one(self, graph: SparseGraphState) -> None:
        """
        Row normalisation is what makes aggregation a mean rather than a sum.

        Without it a node's activation scales with its degree, and the
        network spends capacity learning to undo that.
        """
        assert np.allclose(densify(graph).sum(axis=1), 1.0, atol=1e-6)

    def test_the_weights_are_positive(self, graph: SparseGraphState) -> None:
        """
        A negative weight would mean a neighbour subtracts from a node.

        The kernel cannot produce one, so this is a guard on the
        normalisation arithmetic rather than on the kernel.
        """
        assert (graph.values > 0).all()

    def test_a_closer_neighbour_is_weighted_more_heavily(self, encoded) -> None:
        """
        The whole purpose of the kernel is that similarity decays with distance.

        A graph that weighted all neighbours equally would make the
        attribute encoding irrelevant to message passing.
        """
        features = weighted_features(encoded, spec=GraphSpec())
        graph = build_graph(encoded, spec=GraphSpec(n_neighbours=3), is_target=IS_TARGET)
        dense = densify(graph)

        node = 0
        neighbours = [c for c in range(N_INSTRUMENTS) if c != node and dense[node, c] > 0]
        distances = [np.linalg.norm(features[node] - features[c]) for c in neighbours]
        weights = [dense[node, c] for c in neighbours]

        by_distance = [w for _, w in sorted(zip(distances, weights, strict=True))]
        assert by_distance == sorted(by_distance, reverse=True)

    def test_duplicate_instruments_do_not_produce_nan(
        self, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        Two identical instruments sit at distance zero from each other.

        Their kernel bandwidth is the median of their neighbour distances,
        which is then zero, and dividing by it would make every one of their
        edges NaN. Not hypothetical -- a book holding the same trade twice
        is ordinary.
        """
        duplicated = {key: [values[0]] * 3 + list(values)[3:] for key, values in attributes.items()}
        state = EntityEncoderState.fit(duplicated, spec=AttributeEncoderSpec())
        graph = build_graph(
            state.transform(duplicated), spec=GraphSpec(n_neighbours=2), is_target=IS_TARGET
        )
        assert np.isfinite(graph.values).all()
        assert np.allclose(densify(graph).sum(axis=1), 1.0, atol=1e-6)


class TestWeightedFeatures:
    """How the attribute groups are combined before measuring distance."""

    def test_the_weight_applies_to_squared_distance(self, encoded) -> None:
        """
        Square-rooting alpha is what makes the weights mean what they say.

        Quadrupling an attribute's alpha should double the contribution of
        its differences to the distance, not quadruple it.
        """
        base = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=1.0))
        scaled = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=4.0))
        assert scaled[:, 0] == pytest.approx(2.0 * base[:, 0], rel=1e-6)

    def test_a_zero_weight_drops_the_attribute(self, encoded) -> None:
        """Weighting a group to zero should remove its columns, not zero them."""
        with_moneyness = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=1.0))
        without = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=0.0))
        assert without.shape[1] == with_moneyness.shape[1] - 1

    def test_a_wide_one_hot_does_not_dominate_by_width(self, encoded) -> None:
        """
        A categorical attribute's influence should follow its weight, not its levels.

        Two instruments differing in product type differ in two one-hot
        columns by one each, which without the width normalisation
        contributes twice what a one-column scalar does -- purely because
        the attribute happens to have two levels.
        """
        features = weighted_features(encoded, spec=GraphSpec())
        start, end = encoded.blocks["product_type"]
        width = end - start
        block = features[:, 4 : 4 + width]
        # Two rows with different product types differ by exactly sqrt(2/width)
        # per differing column, so their squared separation is 2/width * ... ,
        # i.e. independent of width.
        separation = np.linalg.norm(block[0] - block[2]) ** 2
        assert separation == pytest.approx(2.0 / width, rel=1e-5)

    def test_weighting_everything_to_zero_is_refused(self, encoded) -> None:
        """
        A graph over no features would have arbitrary structure.

        Returning one would be worse than failing: it would train, and the
        GNN would be aggregating over noise.
        """
        with pytest.raises(ContractError, match="weighted to zero"):
            weighted_features(
                encoded,
                spec=GraphSpec(
                    alpha_moneyness=0.0,
                    alpha_maturity=0.0,
                    alpha_delta=0.0,
                    alpha_vega=0.0,
                    alpha_product_type=0.0,
                    alpha_product_subtype=0.0,
                    alpha_underlying=0.0,
                    alpha_underlying_risk_factors=0.0,
                ),
            )


class TestMetrics:
    """The distance metrics."""

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_each_metric_produces_a_valid_graph(self, encoded, metric: str) -> None:
        """Whichever metric is chosen, the output is still a row-stochastic graph."""
        graph = build_graph(
            encoded, spec=GraphSpec(n_neighbours=3, distance_metric=metric), is_target=IS_TARGET
        )
        assert np.allclose(densify(graph).sum(axis=1), 1.0, atol=1e-6)

    def test_an_unknown_metric_is_refused(self, encoded) -> None:
        """Silently falling back to euclidean would hide a typo in a config."""
        spec = GraphSpec(n_neighbours=3).model_copy(update={"distance_metric": "mahalanobis"})
        with pytest.raises(ContractError, match="mahalanobis"):
            build_graph(encoded, spec=spec, is_target=IS_TARGET)


class TestTheGraphIsRobust:
    """
    The port's one deliberate numerical difference from the baseline.

    scikit-learn computed neighbour distances in float32 from an expansion
    that cancels; this port computes the explicit difference in float64.
    The weights therefore differ in their last float32 bit, which
    ``ADJACENCY_VALUE_ATOL`` allows. These tests are the evidence that the
    allowance is safe -- that the noise is far too small to change which
    edges exist.
    """

    def test_the_neighbour_margin_dwarfs_the_numerical_noise(self, encoded) -> None:
        """
        Measure the gap between the furthest kept neighbour and the nearest dropped one.

        If that margin were comparable to the arithmetic noise, the two
        implementations could disagree about the graph's *structure* and the
        relaxed weight tolerance would be hiding a real difference. It is
        four orders of magnitude larger.
        """
        features = weighted_features(encoded, spec=GraphSpec()).astype(np.float64)
        difference = features[:, None, :] - features[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", difference, difference))
        np.fill_diagonal(distances, np.inf)

        n_neighbours = 3
        ordered = np.sort(distances, axis=1)
        margin = (ordered[:, n_neighbours] - ordered[:, n_neighbours - 1]).min()
        assert margin > 100 * ADJACENCY_VALUE_ATOL

    def test_perturbing_the_features_by_that_noise_leaves_the_edges_unchanged(
        self, encoded
    ) -> None:
        """
        The direct statement of the claim, rather than a proxy for it.

        Jitter the features by more than the two implementations disagree
        by, and the edge list must come out identical.
        """
        spec = GraphSpec(n_neighbours=3)
        baseline = build_graph(encoded, spec=spec, is_target=IS_TARGET)

        generator = np.random.default_rng(0)
        noise = generator.normal(scale=1e-6, size=encoded.features.shape).astype(np.float32)
        jittered = build_graph(
            type(encoded)(
                features=encoded.features + noise, names=encoded.names, blocks=encoded.blocks
            ),
            spec=spec,
            is_target=IS_TARGET,
        )
        assert np.array_equal(jittered.indices, baseline.indices)

    def test_the_tolerance_is_one_float32_bit_not_a_blanket(self) -> None:
        """
        A tolerance wide enough to hide a real bug is not a tolerance.

        One float32 ULP near unit magnitude is about 1.2e-07; anything much
        larger would start admitting genuine differences in the kernel or
        the normalisation.
        """
        assert np.finfo(np.float32).eps >= ADJACENCY_VALUE_ATOL


class TestRoundTrip:
    """Saving and loading."""

    def test_a_reloaded_graph_is_identical(self, graph: SparseGraphState, tmp_path) -> None:
        """Including the edge order, which is part of the state."""
        graph.save(tmp_path)
        restored = SparseGraphState.load(tmp_path)
        assert np.array_equal(restored.indices, graph.indices)
        assert np.array_equal(restored.values, graph.values)
        assert restored.n_nodes == graph.n_nodes
        assert np.array_equal(restored.is_target, graph.is_target)

    def test_nothing_is_pickled(self, graph: SparseGraphState, tmp_path) -> None:
        """
        The original stored a SciPy sparse matrix by pickle.

        That ties a saved model to a SciPy version. Plain arrays and JSON do
        not.
        """
        graph.save(tmp_path)
        suffixes = {path.suffix for path in tmp_path.rglob("*") if path.is_file()}
        assert suffixes <= {".npy", ".npz", ".json"}
```

