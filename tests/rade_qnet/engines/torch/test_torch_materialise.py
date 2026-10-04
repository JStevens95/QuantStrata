"""
Tests for lazy-parameter materialisation -- defect 6.

A model with lazily shaped parameters has *no parameters at all* until it has
seen one batch. ``torch.nn.LazyLinear`` is the common case: its weight is an
``UninitializedParameter`` with no shape until the first forward pass infers
the input width.

Handing such a model to an optimiser produces an optimiser tracking an empty
parameter group. It constructs without complaint, ``step()`` succeeds, and
nothing is ever updated. The loss curve is perfectly flat, which reads as a
learning-rate problem and sends an investigation in the wrong direction for as
long as it takes someone to count the parameters.

Wrapping it for distributed training is worse: the wrapper either crashes deep
inside the distributed library or, on some versions, synchronises an empty
parameter set -- so every rank trains independently and the run silently is
not distributed at all.

The fix is that materialisation happens first, driven by the input signature
rather than by a batch. That is why :class:`InputSignature` exists: the dummy
forward needs exact shapes and dtypes with no data present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.rade_qnet.core.contract.signature import InputSignature, TensorSpec
from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.engines.torch.materialise import (
    count_parameters,
    dummy_batch,
    has_lazy_parameters,
    materialise,
    torch_dtype,
)
from src.rade_qnet.testkit.fixtures import make_signature


class LazyNet(nn.Module):
    """A model whose first layer infers its own input width."""

    def __init__(self) -> None:
        """Build the lazy stack."""
        super().__init__()
        self.layer = nn.LazyLinear(8)
        self.head = nn.Linear(8, 1)

    def forward(self, features):
        """Run the forward pass."""
        return self.head(torch.relu(self.layer(features)))


class EagerNet(nn.Module):
    """A model whose shapes are fixed at construction."""

    def __init__(self, n_features: int = 4) -> None:
        """Build the eager stack."""
        super().__init__()
        self.layer = nn.Linear(n_features, 1)

    def forward(self, features):
        """Run the forward pass."""
        return self.layer(features)


class StaticNet(nn.Module):
    """A model consuming a static input alongside its dynamic one."""

    def __init__(self) -> None:
        """Build the stack."""
        super().__init__()
        self.layer = nn.LazyLinear(1)

    def forward(self, features, adjacency):
        """Mix the features through the adjacency before projecting."""
        return self.layer(features @ adjacency)


class TestDetection:
    """Telling a lazy model from an eager one, before anything consumes it."""

    def test_a_lazy_model_is_detected(self):
        """So the pipeline knows materialisation has work to do."""
        assert has_lazy_parameters(LazyNet())

    def test_an_eager_model_is_not(self):
        """
        So a model that never needed materialising pays nothing for it.

        Most models are eager, and a dummy forward pass on every one of them
        would be a cost for no benefit.
        """
        assert not has_lazy_parameters(EagerNet())

    def test_a_lazy_model_reports_no_parameters(self):
        """
        The defect, stated as the number it turns on.

        This is what an optimiser built before materialisation would see, and
        it is why such an optimiser updates nothing.
        """
        assert count_parameters(LazyNet()) == 0

    def test_a_materialised_model_reports_its_real_parameters(self):
        """
        And this is what it sees afterwards.

        4 by 8 weights plus 8 biases, then 8 by 1 plus 1: 73 in total,
        asserted exactly rather than as "more than zero", because a wrong
        inferred width would also be more than zero.
        """
        model = materialise(LazyNet(), make_signature(n_features=4, dtype="float32"))
        assert count_parameters(model) == (4 * 8 + 8) + (8 * 1 + 1)


class TestTheDummyBatch:
    """Synthesised from the signature, with no data present."""

    def test_every_declared_input_appears(self):
        """
        Including static inputs, because the forward pass takes them too.

        A dummy batch missing one produces a ``TypeError`` from inside
        ``forward``, which says nothing about the signature.
        """
        signature = InputSignature(
            dynamic={"features": TensorSpec(shape=(None, 4), dtype="float32")},
            static={"adjacency": TensorSpec(shape=(4, 4), dtype="float32")},
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        batch = dummy_batch(signature)
        assert set(batch) >= {"features", "adjacency"}

    def test_wildcard_dimensions_become_concrete(self):
        """
        A ``None`` in a shape is the batch axis, and a tensor cannot have one.

        Two rows rather than one, so a shape bug that collapses the batch axis
        is visible.
        """
        batch = dummy_batch(make_signature(n_features=4, dtype="float32"))
        assert batch["features"].shape[0] > 1
        assert batch["features"].shape[1] == 4

    def test_the_dtype_matches_the_declaration(self):
        """
        Because a float64 input meeting float32 weights is a hard error.

        Which is the right behaviour, but it has to be the *signature* that
        decides, not whatever numpy happened to default to.
        """
        batch = dummy_batch(make_signature(n_features=4, dtype="float32"))
        assert batch["features"].dtype == torch.float32

    def test_integer_inputs_are_filled_with_zeros(self):
        """
        Because an integer input is almost always an index.

        Ones would be a valid index into an embedding table of size two or
        more and would fail on a table of size one; zero is valid for any
        non-empty table.
        """
        signature = InputSignature(
            dynamic={"tokens": TensorSpec(shape=(None, 3), dtype="int64")},
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        assert torch.all(dummy_batch(signature)["tokens"] == 0)

    def test_float_inputs_are_filled_with_ones(self):
        """
        Zeros would hide a shape error in any multiplicative layer.

        A zero matrix multiplied by anything is zero whatever the shapes
        happen to be, so a dummy pass over zeros can succeed where real data
        would not.
        """
        batch = dummy_batch(make_signature(n_features=4, dtype="float32"))
        assert torch.all(batch["features"] == 1.0)


class TestMaterialise:
    """The pass itself, and what it reports when it cannot run."""

    def test_an_eager_model_is_returned_unchanged(self):
        """
        No dummy pass is run, which is the cheap path and the common one.

        Returning the same object rather than a copy means nothing downstream
        has to care whether materialisation was needed.
        """
        model = EagerNet()
        assert materialise(model, make_signature(n_features=4, dtype="float32")) is model

    def test_a_model_with_static_inputs_is_materialised(self):
        """
        The case the signature's static group exists for.

        A forward pass needing an adjacency matrix cannot be run from the
        dynamic inputs alone, so a signature that did not declare static
        inputs would make this model unmaterialisable.
        """
        signature = InputSignature(
            dynamic={"features": TensorSpec(shape=(None, 4), dtype="float32")},
            static={"adjacency": TensorSpec(shape=(4, 4), dtype="float32")},
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        model = materialise(StaticNet(), signature)
        assert count_parameters(model) > 0

    def test_materialisation_is_idempotent(self):
        """
        Because a pipeline stage may be re-run, and a tuning sweep will be.

        A second dummy pass that re-initialised the weights would discard
        whatever the first one established.
        """
        signature = make_signature(n_features=4, dtype="float32")
        once = materialise(LazyNet(), signature)
        before = count_parameters(once)
        assert count_parameters(materialise(once, signature)) == before

    def test_a_signature_that_does_not_fit_the_model_is_reported(self):
        """
        With the signature printed, because that is the thing to fix.

        A bare ``RuntimeError`` from inside a matrix multiply does not say
        which declared shape was wrong, and the shapes are not visible in the
        traceback.
        """
        signature = InputSignature(
            dynamic={
                "features": TensorSpec(shape=(None, 4), dtype="float32"),
                "unexpected": TensorSpec(shape=(None, 2), dtype="float32"),
            },
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        with pytest.raises(EngineError, match="input signature"):
            materialise(LazyNet(), signature)

    def test_materialisation_does_not_leave_gradients_behind(self):
        """
        The dummy pass is an initialisation, not a training step.

        A gradient left on a parameter is added to the first real batch's
        gradient, so the first update is computed partly from synthetic ones.
        """
        model = materialise(LazyNet(), make_signature(n_features=4, dtype="float32"))
        assert all(
            parameter.grad is None or torch.all(parameter.grad == 0)
            for parameter in model.parameters()
        )


class TestDtypeMapping:
    """Signature dtype strings to torch dtypes."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("int64", torch.int64),
        ],
    )
    def test_the_common_dtypes_map(self, name, expected):
        """So a signature can be written in numpy's vocabulary."""
        assert torch_dtype(name) == expected

    def test_an_unknown_dtype_is_reported_with_the_alternatives(self):
        """
        Rather than defaulting to float32.

        A silent default would make an integer index column into a float one,
        and an embedding lookup against a float index fails much later with a
        message about indices.
        """
        with pytest.raises(EngineError):
            torch_dtype("complex256")

    def test_numpy_dtype_names_are_accepted(self):
        """
        Because that is what a signature derived from arrays will carry.

        ``str(array.dtype)`` is the natural way to build a signature from
        data, and it must not need translating at every call site.
        """
        assert torch_dtype(str(np.zeros(1, dtype=np.float32).dtype)) == torch.float32
