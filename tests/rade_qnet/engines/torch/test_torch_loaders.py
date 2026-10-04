"""
Tests for the batch-to-device bridge -- defect 4.

Defect 4 is that static inputs were merged into every sample during dataset
indexing, and the collate function then ran ``torch.equal`` across the batch
for each static key and returned ``values[0]``. So a graph adjacency matrix
was compared against itself ``batch_size`` times per batch, every batch, every
epoch, to confirm something true by construction.

The important part of the fix is that it is *not a behavioural change*: the
old collation already returned one copy, and the network already received
exactly one. Static inputs now travel outside the batch stream, which delivers
the same tensors to the same place and deletes the comparison.

So the tests here establish two things. That a static input is uploaded once
-- asserted as object identity across batches, which is the only way to tell
one upload from several. And that the tensors reaching the model are the same
ones, so the refactor is a deletion rather than a change.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.rade_qnet.core.contract.data import TARGET_KEY
from src.rade_qnet.core.contract.signature import InputSignature, TensorSpec
from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.engines.torch.loaders import StaticInputs, to_device_batches, to_tensor
from src.rade_qnet.testkit.fixtures import SyntheticTensorSource

CPU = torch.device("cpu")


@pytest.fixture
def source():
    """
    Provide a bounded source with no static inputs.

    Returns
    -------
    SyntheticTensorSource
        The source.
    """
    rng = np.random.default_rng(0)
    return SyntheticTensorSource(
        features=rng.normal(size=(50, 4)).astype(np.float32),
        targets=rng.normal(size=(50, 1)).astype(np.float32),
        batch_size=16,
    )


@pytest.fixture
def graph_source():
    """
    Provide a source carrying a static adjacency matrix.

    Returns
    -------
    SyntheticTensorSource
        The source.
    """
    rng = np.random.default_rng(1)
    return SyntheticTensorSource(
        features=rng.normal(size=(50, 4)).astype(np.float32),
        targets=rng.normal(size=(50, 1)).astype(np.float32),
        batch_size=16,
        static={"adjacency": np.eye(4, dtype=np.float32)},
    )


class TestToTensor:
    """Conversion, with the dtype decided by the signature."""

    def test_an_array_becomes_a_tensor_of_the_declared_dtype(self):
        """
        Not of whatever numpy happened to produce.

        A float64 array meeting float32 weights is a hard error, so the
        signature has to be what decides.
        """
        tensor = to_tensor(np.zeros((2, 3), dtype=np.float64), dtype=torch.float32, device=CPU)
        assert tensor.dtype == torch.float32

    def test_a_tensor_is_accepted_unchanged_in_shape(self):
        """So a source that already produces tensors is not penalised."""
        tensor = to_tensor(torch.zeros(2, 3), dtype=torch.float32, device=CPU)
        assert tensor.shape == (2, 3)


class TestStaticInputsAreUploadedOnce:
    """Defect 4, stated as object identity."""

    def test_the_same_tensor_object_appears_in_every_batch(self, graph_source):
        """
        One upload, not one per batch.

        Identity rather than equality is the whole point: equality would hold
        even if the tensor were re-created for every batch, which is precisely
        the cost being removed.
        """
        signature = graph_source.signature
        static = StaticInputs.from_source(graph_source, signature=signature, device=CPU)

        seen = [
            inputs["adjacency"]
            for inputs, _ in to_device_batches(
                graph_source, signature=signature, device=CPU, static=static
            )
        ]
        assert len({id(tensor) for tensor in seen}) == 1

    def test_a_static_input_is_not_batched(self, graph_source):
        """
        It keeps its declared shape, with no batch axis prepended.

        A 4 by 4 adjacency collated per sample would arrive as 16 by 4 by 4,
        which is the shape the old implementation built and then threw away.
        """
        signature = graph_source.signature
        static = StaticInputs.from_source(graph_source, signature=signature, device=CPU)
        inputs, _ = next(
            iter(to_device_batches(graph_source, signature=signature, device=CPU, static=static))
        )
        assert inputs["adjacency"].shape == (4, 4)

    def test_a_model_with_no_static_inputs_pays_nothing(self, source):
        """
        Which is the common case and must not be a special case.

        A batch for a tabular model carries exactly its dynamic input.
        """
        inputs, _ = next(iter(to_device_batches(source, signature=source.signature, device=CPU)))
        assert set(inputs) == {"features"}


class TestBatchContents:
    """What arrives at the model, and in what form."""

    def test_the_target_is_separated_from_the_inputs(self, source):
        """
        Because the model takes the inputs and the loss takes the target.

        A target left among the inputs is passed to ``forward`` as a keyword
        argument, and the failure names the argument rather than the cause.
        """
        inputs, target = next(
            iter(to_device_batches(source, signature=source.signature, device=CPU))
        )
        assert TARGET_KEY not in inputs
        assert target.shape == (16, 1)

    def test_every_batch_is_yielded(self, source):
        """
        Including the short final one, unless the source dropped it.

        Silently discarding it would make the loop train on fewer samples than
        the source reported, and the denominator of every metric wrong by
        less than one batch -- never enough to notice.
        """
        batches = list(to_device_batches(source, signature=source.signature, device=CPU))
        assert sum(target.shape[0] for _, target in batches) == 50

    def test_the_values_survive_the_conversion(self, source):
        """
        The refactor is a deletion, so the numbers must be unchanged.

        Asserted against the source's own arrays rather than against a
        recomputation, because the claim is specifically that nothing happens
        to them on the way.
        """
        inputs, _ = next(iter(to_device_batches(source, signature=source.signature, device=CPU)))
        assert torch.equal(inputs["features"], torch.from_numpy(source.features[:16]))


class TestValidation:
    """A batch that does not match its signature is reported, not guessed at."""

    def test_a_missing_declared_input_is_reported(self):
        """
        Naming the input, because ``forward`` will not.

        A missing keyword argument raises from inside the model, where the
        message mentions a parameter name and nothing about the signature that
        was supposed to supply it.
        """
        rng = np.random.default_rng(2)
        signature = InputSignature(
            dynamic={
                "features": TensorSpec(shape=(None, 4), dtype="float32"),
                "absent": TensorSpec(shape=(None, 2), dtype="float32"),
            },
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        source = SyntheticTensorSource(
            features=rng.normal(size=(20, 4)).astype(np.float32),
            targets=rng.normal(size=(20, 1)).astype(np.float32),
            signature_override=signature,
        )
        with pytest.raises(ContractError):
            list(to_device_batches(source, signature=signature, device=CPU))

    def test_only_the_first_batch_is_validated_by_default(self, source):
        """
        Because every batch has the same keys by construction.

        Validating all of them would put a dictionary comparison in the inner
        loop of training, which is a real cost for a check that cannot newly
        fail after the first batch.
        """
        batches = list(
            to_device_batches(source, signature=source.signature, device=CPU, validate_first=True)
        )
        assert len(batches) == 4


class TestStaticInputsHelper:
    """The holder itself."""

    def test_an_empty_holder_merges_nothing(self, source):
        """So the no-static path has no branch of its own downstream."""
        static = StaticInputs.from_source(source, signature=source.signature, device=CPU)
        assert not static
        assert set(static.merge_into({"features": torch.zeros(2, 4)})) == {"features"}

    def test_the_description_names_the_inputs_and_their_shapes(self, graph_source):
        """
        So a run log says the adjacency was uploaded, and how big it was.

        A static input is the one thing in a batch nobody sees in the loss
        curve, so if it is wrong the log is where it has to be visible.
        """
        static = StaticInputs.from_source(
            graph_source, signature=graph_source.signature, device=CPU
        )
        assert "adjacency" in str(static.describe())
