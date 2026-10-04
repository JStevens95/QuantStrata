"""
Tests for the input contract: what a model consumes, declared and checked.

The cases below are organised around the two defects this type was
introduced to kill, rather than around its methods. Both were silent --
they produced a trained model and a plausible loss curve -- so the tests
that matter most are the ones asserting that something now *fails*.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.requirement import InputRequirement, RequiredInput
from src.rade_qnet.core.contract.signature import InputSignature, TensorSpec
from src.rade_qnet.core.runtime.errors import ContractError

WINDOW = TensorSpec(shape=(None, 5, 4), dtype="float32")
FLAT = TensorSpec(shape=(None, 4), dtype="float32")
TARGET = TensorSpec(shape=(None, 1), dtype="float32")


def signature(**dynamic: TensorSpec) -> InputSignature:
    """
    Build a signature with the given dynamic inputs.

    Parameters
    ----------
    **dynamic
        Name to spec.

    Returns
    -------
    InputSignature
        With a scalar target.
    """
    return InputSignature(dynamic=dynamic, target=TARGET)


class TestTheAmbiguousInputDefect:
    """
    A model that reads "one input" given two of them.

    Before this contract existed, the recurrent baseline scanned the batch
    and took the first tensor it found. Two feature blocks meant it trained
    on whichever the dictionary yielded first, so reordering the data build
    changed the model and nothing said so.
    """

    def test_one_unnamed_input_is_satisfied_by_any_single_input(self) -> None:
        """
        The name is the source's choice, so the model must not assume it.

        A model that hard-coded ``features`` would work against every
        fixture in this repository and fail against the first user whose
        block is called something else.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        requirement.check(signature(whatever_they_called_it=WINDOW), model="m")

    def test_two_inputs_are_refused_when_one_is_required(self) -> None:
        """
        The headline. This is the run that used to succeed and be wrong.

        The message names both inputs, because the user's next question is
        always "which one did it want?" and the answer is "it cannot tell,
        and neither can you".
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        with pytest.raises(ContractError, match="exactly one unnamed"):
            requirement.check(signature(prices=WINDOW, volumes=WINDOW), model="m")

    def test_the_failure_names_both_candidates(self) -> None:
        """An error a user cannot act on is barely better than silence."""
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        with pytest.raises(ContractError) as caught:
            requirement.check(signature(prices=WINDOW, volumes=WINDOW), model="m")
        assert "prices" in str(caught.value)
        assert "volumes" in str(caught.value)


class TestTheWrongRankDefect:
    """
    A tensor of the wrong shape that the library accepts anyway.

    ``torch.atleast_3d`` appends its axis, turning ``(samples, features)``
    into ``(samples, features, 1)``. The recurrence accepts that happily
    and learns nonsense from it.
    """

    def test_an_accepted_rank_passes(self) -> None:
        """Rank 2 is legal for this model, as a window of length one."""
        requirement = InputRequirement(dynamic=(RequiredInput(rank=(3, 2)),))
        requirement.check(signature(features=FLAT), model="m")

    def test_an_unaccepted_rank_is_refused(self) -> None:
        """
        Rank 4 is not quietly reshaped into something plausible.

        Declaring *which* ranks are acceptable is the point: a model that
        handles two of them should say so, rather than leaving a reader to
        infer it from a reshape buried in a forward pass.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        with pytest.raises(ContractError, match="rank 2, but rank 3"):
            requirement.check(signature(features=FLAT), model="m")


class TestNamedRequirements:
    """For a model whose inputs are not interchangeable."""

    def test_a_missing_named_input_is_refused(self) -> None:
        """
        The flagship's case: swapping two statics trains a wrong model.

        ``adjacency_values`` and ``target_indices`` are not
        interchangeable, and a build that supplied one for the other would
        produce a decreasing loss and attribute every prediction to the
        wrong instrument -- invisible in every metric a run reports.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(name="pnl_history", rank=3),))
        with pytest.raises(ContractError, match="'pnl_history' is required"):
            requirement.check(signature(something_else=WINDOW), model="m")

    def test_a_named_requirement_does_not_consume_an_unnamed_ones_input(
        self,
    ) -> None:
        """
        Named inputs are matched first, so order cannot change the outcome.

        If the unnamed requirement were matched first it could claim the
        very input a named one asked for, and whether it did would depend
        on dictionary ordering -- which is the defect, reintroduced.
        """
        requirement = InputRequirement(
            dynamic=(
                RequiredInput(name="pnl_history", rank=3),
                RequiredInput(rank=2),
            )
        )
        requirement.check(signature(pnl_history=WINDOW, extras=FLAT), model="m")

    def test_an_unconsumed_input_is_refused_by_default(self) -> None:
        """
        A build producing something the model never reads is a warning sign.

        Either the work is wasted, or -- the dangerous case -- the user
        believes the model is using information it cannot see. The second
        is invisible in every metric, so it is refused rather than logged.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(name="pnl_history", rank=3),))
        with pytest.raises(ContractError, match="does not consume"):
            requirement.check(signature(pnl_history=WINDOW, spare=FLAT), model="m")


class TestOptionalRefinements:
    """Dtype and shape, for the cases where they are genuine requirements."""

    def test_a_wrong_dtype_is_refused_when_declared(self) -> None:
        """An index tensor of floats is a bug, not a conversion."""
        requirement = InputRequirement(dynamic=(RequiredInput(rank=2, dtype="int64"),))
        with pytest.raises(ContractError, match="dtype float32"):
            requirement.check(signature(indices=FLAT), model="m")

    def test_a_declared_shape_constrains_the_axes_it_names(self) -> None:
        """
        Edge endpoints come in pairs; that is a requirement, not a reading.

        The wildcard axis is left free because the edge count is a property
        of the book. Pinning it would duplicate a number the signature
        already carries, and two copies of a number can disagree.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=2, shape=(None, 2)),))
        requirement.check(signature(edges=TensorSpec(shape=(99, 2), dtype="int64")), model="m")
        with pytest.raises(ContractError, match="is required"):
            requirement.check(signature(edges=TensorSpec(shape=(99, 3), dtype="int64")), model="m")

    def test_unconstrained_accepts_anything(self) -> None:
        """
        The honest declaration for a model that flattens what it is given.

        Spelled rather than left unset, so a reader knows the question was
        asked and answered.
        """
        InputRequirement.unconstrained().check(signature(a=WINDOW, b=FLAT), model="ridge")


class TestTheDeclarationItselfIsChecked:
    """A requirement that cannot be satisfied should fail its author."""

    def test_two_unnamed_entries_are_refused(self) -> None:
        """
        They are indistinguishable, so matching would depend on ordering.

        That is precisely the defect this type removes, so permitting it
        here would let a model reintroduce it while appearing to declare a
        contract.
        """
        with pytest.raises(ValidationError, match="more than one unnamed"):
            InputRequirement(dynamic=(RequiredInput(rank=3), RequiredInput(rank=3)))

    def test_a_duplicated_name_is_refused(self) -> None:
        """Two requirements on one input cannot both be the contract."""
        with pytest.raises(ValidationError, match="name an input twice"):
            InputRequirement(
                dynamic=(
                    RequiredInput(name="x", rank=3),
                    RequiredInput(name="x", rank=2),
                )
            )

    def test_a_shape_contradicting_the_rank_is_refused(self) -> None:
        """
        Caught at authoring time, so the model's author hears about it.

        Left to check time, this would surface as a requirement no data
        build can ever satisfy, reported to a user who did nothing wrong.
        """
        with pytest.raises(ValidationError, match="not among the accepted ranks"):
            RequiredInput(rank=3, shape=(None, 2))

    def test_a_rank_below_one_is_refused(self) -> None:
        """Every batched input has at least a batch axis."""
        with pytest.raises(ValidationError, match="minimum rank is 1"):
            RequiredInput(rank=0)
