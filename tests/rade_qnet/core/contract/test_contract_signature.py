"""
Tests for the input signature.

:class:`InputSignature` is the most load-bearing contract in the framework
because three unrelated mechanisms depend on its exact shape: static-input
device placement, lazy parameter materialisation, and rebuilding a model from a
saved bundle. The static/dynamic split in particular is what lets an engine
treat the two differently without inspecting any data -- which is what retires
the per-sample static tensor comparison in the implementation being replaced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.signature import (
    InputSignature,
    PolicySignature,
    SpaceSpec,
    TensorSpec,
)
from src.rade_qnet.core.lifecycle.errors import ContractError, SpecError


@pytest.fixture
def signature():
    """
    Provide a signature with one dynamic input, one static input and a target.

    Returns
    -------
    InputSignature
        A signature exercising both input groups.
    """
    return InputSignature(
        dynamic={"features": TensorSpec(shape=(None, 8), dtype="float32")},
        static={"adjacency": TensorSpec(shape=(12, 12), dtype="float32")},
        target=TensorSpec(shape=(None, 1), dtype="float32"),
    )


class TestTensorSpec:
    """A tensor spec describes a shape without naming a library's dtype."""

    def test_the_dtype_is_a_plain_string(self):
        """
        Library-agnostic by construction.

        ``core`` may not import a training library, and a signature written by
        the Torch engine has to be readable by the XGBoost one.
        """
        assert TensorSpec(shape=(4,), dtype="float32").dtype == "float32"

    def test_a_wildcard_dimension_is_accepted(self):
        """``None`` marks the batch dimension, whose size is not yet known."""
        assert TensorSpec(shape=(None, 8), dtype="float32").rank == 2

    def test_a_zero_dimension_is_rejected(self):
        """
        A zero dimension produces an empty tensor.

        Which trains without error and learns nothing -- far harder to
        diagnose later than a rejection here.
        """
        with pytest.raises(ValidationError, match="positive"):
            TensorSpec(shape=(0, 4), dtype="float32")

    def test_a_negative_dimension_is_rejected(self):
        """Not a valid shape under any interpretation."""
        with pytest.raises(ValidationError, match="positive"):
            TensorSpec(shape=(-1, 4), dtype="float32")

    def test_a_scalar_shape_is_accepted(self):
        """A zero-rank tensor is a legitimate loss or scalar feature."""
        assert TensorSpec(shape=(), dtype="float32").rank == 0


class TestCompatibility:
    """Compatibility is weaker than equality, deliberately."""

    def test_a_wildcard_matches_a_concrete_size(self):
        """
        What lets a declared signature be checked against a real batch.

        The signature says "any batch size"; the batch says 16.
        """
        declared = TensorSpec(shape=(None, 8), dtype="float32")
        actual = TensorSpec(shape=(16, 8), dtype="float32")
        assert declared.is_compatible_with(actual)

    def test_a_mismatched_dtype_is_incompatible(self):
        """float32 and float64 data are not interchangeable."""
        assert not TensorSpec(shape=(16, 8), dtype="float32").is_compatible_with(
            TensorSpec(shape=(16, 8), dtype="float64")
        )

    def test_a_mismatched_concrete_dimension_is_incompatible(self):
        """Eight features are not sixteen features."""
        assert not TensorSpec(shape=(16, 8), dtype="float32").is_compatible_with(
            TensorSpec(shape=(16, 16), dtype="float32")
        )

    def test_a_mismatched_rank_is_incompatible(self):
        """
        Checked before the dimensions are compared.

        Otherwise zipping two shapes of different length would either raise or
        silently compare a prefix.
        """
        assert not TensorSpec(shape=(16, 8), dtype="float32").is_compatible_with(
            TensorSpec(shape=(16, 8, 1), dtype="float32")
        )

    def test_compatibility_is_symmetric_for_wildcards(self):
        """A wildcard on either side matches."""
        wild = TensorSpec(shape=(None, 8), dtype="float32")
        concrete = TensorSpec(shape=(16, 8), dtype="float32")
        assert wild.is_compatible_with(concrete) == concrete.is_compatible_with(wild)


class TestConcreteShape:
    """Resolving wildcards is how a dummy batch is synthesised."""

    def test_wildcards_resolve_to_the_given_batch_size(self):
        """
        The mechanism behind lazy parameter materialisation.

        A model whose shapes are known only after the data build is
        materialised by one forward pass over a synthetic batch, built from
        this.
        """
        assert TensorSpec(shape=(None, 8), dtype="float32").concrete_shape(16) == (16, 8)

    def test_concrete_dimensions_are_unchanged(self):
        """A static input's shape is already fully known."""
        assert TensorSpec(shape=(12, 12), dtype="float32").concrete_shape(16) == (12, 12)

    def test_a_non_positive_batch_size_is_rejected(self):
        """A dummy batch of zero samples materialises nothing."""
        with pytest.raises(SpecError):
            TensorSpec(shape=(None, 8), dtype="float32").concrete_shape(0)


class TestInputSignature:
    """The static/dynamic split is enforced, not merely documented."""

    def test_input_names_cover_both_groups(self, signature):
        """A caller asking what the model consumes gets everything."""
        assert signature.input_names == ("adjacency", "features")

    def test_an_empty_dynamic_set_is_rejected(self):
        """
        A model with only static inputs cannot vary with the data.

        It would produce the same output for every sample, which is not a
        model.
        """
        with pytest.raises(ValidationError, match="dynamic"):
            InputSignature(dynamic={}, target=TensorSpec(shape=(None,), dtype="float32"))

    def test_a_name_in_both_groups_is_rejected(self, signature):
        """
        A duplicated name is unresolvable.

        The engine would have to guess whether to collate it per sample or
        upload it once, and either choice is silently wrong half the time.
        """
        spec = TensorSpec(shape=(None, 8), dtype="float32")
        with pytest.raises(ValidationError, match="both"):
            InputSignature(dynamic={"x": spec}, static={"x": spec}, target=spec)

    def test_static_inputs_default_to_empty(self):
        """Most models have none, and should not have to say so."""
        spec = TensorSpec(shape=(None, 8), dtype="float32")
        assert InputSignature(dynamic={"x": spec}, target=spec).static == {}

    def test_a_spec_can_be_looked_up_from_either_group(self, signature):
        """A caller need not know which group a name is in."""
        assert signature.spec_for("features").dtype == "float32"
        assert signature.spec_for("adjacency").shape == (12, 12)

    def test_an_unknown_name_lists_the_declared_inputs(self, signature):
        """
        The usual cause is a renamed key in a data module.

        Listing what is declared makes that immediately visible.
        """
        with pytest.raises(ContractError, match="adjacency"):
            signature.spec_for("feature")


class TestBatchKeyValidation:
    """Keys are checked; shapes are the engine's business."""

    def test_a_complete_batch_passes(self, signature):
        """The normal case."""
        signature.validate_batch_keys({"features": 1, "target": 2}, where="train")

    def test_a_missing_dynamic_input_is_rejected(self, signature):
        """
        Caught here rather than as a dimension mismatch in a forward pass.

        The message names both sides, which is what makes a renamed key
        obvious instead of mysterious.
        """
        with pytest.raises(ContractError, match="features"):
            signature.validate_batch_keys({"target": 2}, where="train")

    def test_an_undeclared_key_is_rejected(self, signature):
        """
        An extra key means the data module and the model disagree.

        Silently ignoring it would mean a feature the user believes is in use
        is not.
        """
        with pytest.raises(ContractError, match="junk"):
            signature.validate_batch_keys({"features": 1, "junk": 3}, where="train")

    def test_a_static_input_in_the_batch_is_permitted(self, signature):
        """
        Permitted but not required.

        A source may deliver static inputs separately, which is the path that
        avoids collating them per sample.
        """
        signature.validate_batch_keys({"features": 1, "adjacency": 2, "target": 3}, where="train")

    def test_a_batch_without_a_target_is_permitted(self, signature):
        """An inference batch legitimately has no target."""
        signature.validate_batch_keys({"features": 1}, where="infer")

    def test_the_caller_is_named_in_the_error(self, signature):
        """
        So the failure says which stage produced the bad batch.

        A message naming only the key leaves the reader guessing between the
        training loader, the validation loader and the inference path.
        """
        with pytest.raises(ContractError, match="validation loader"):
            signature.validate_batch_keys({}, where="validation loader")

    def test_the_target_key_can_be_renamed(self, signature):
        """A model whose target is named differently is still checkable."""
        signature.validate_batch_keys(
            {"features": 1, "label": 2}, where="train", target_key="label"
        )


class TestSpaceSpec:
    """Observation and action spaces, for the interactive case."""

    def test_a_box_space_is_accepted(self):
        """The continuous case."""
        space = SpaceSpec(kind="box", shape=(4,), dtype="float32", low=-1.0, high=1.0)
        assert space.shape == (4,)

    def test_a_discrete_space_requires_its_size(self):
        """
        A discrete space with no action count is not a space.

        Any default would be arbitrary, and an arbitrary action count produces
        a policy head of the wrong width.
        """
        with pytest.raises(ValidationError, match="'n'"):
            SpaceSpec(kind="discrete")

    def test_an_action_count_on_a_box_space_is_rejected(self):
        """
        ``n`` has no meaning for a continuous space.

        Accepting it would let two contradictory descriptions of one space
        coexist.
        """
        with pytest.raises(ValidationError, match="'n'"):
            SpaceSpec(kind="box", shape=(4,), n=3)

    def test_a_zero_action_count_is_rejected(self):
        """A policy must have something to choose between."""
        with pytest.raises(ValidationError):
            SpaceSpec(kind="discrete", n=0)


class TestDescribeAndRoundTrip:
    """Signatures are rendered for humans and stored in bundles."""

    def test_a_tensor_spec_describes_compactly(self):
        """Used in error messages and reports."""
        assert TensorSpec(shape=(None, 20, 64), dtype="float32").describe() == (
            "float32[?, 20, 64]"
        )

    def test_a_signature_describes_every_input(self, signature):
        """A reader can see the whole interface at a glance."""
        rendered = signature.describe()
        assert "features" in rendered
        assert "adjacency" in rendered
        assert "target" in rendered

    def test_a_signature_round_trips_through_json(self, signature):
        """
        The property that makes a model rebuildable from a bundle.

        The signature is stored as JSON; if it did not reload exactly, a
        six-month-old bundle could not reconstruct its model.
        """
        assert InputSignature.model_validate_json(signature.model_dump_json()) == signature

    def test_a_policy_signature_round_trips(self):
        """The interactive counterpart, stored the same way."""
        policy = PolicySignature(
            observation=SpaceSpec(kind="box", shape=(4,), low=-1.0, high=1.0),
            action=SpaceSpec(kind="discrete", n=3, dtype="int64"),
        )
        assert PolicySignature.model_validate_json(policy.model_dump_json()) == policy

    def test_a_signature_is_frozen(self, signature):
        """A stage must not be able to redeclare the interface mid-run."""
        with pytest.raises(ValidationError):
            signature.target = TensorSpec(shape=(None, 2), dtype="float32")
