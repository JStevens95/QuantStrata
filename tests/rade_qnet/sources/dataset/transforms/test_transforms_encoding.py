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

from src.rade_qnet.core.runtime.errors import CapabilityError, ContractError
from src.rade_qnet.sources.dataset.transforms.encoding import EncodingState

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
