"""
Tests for the assembled network, including parity level 3.

The layer tests check each block in isolation. These check that they are
wired together in the right order with the right widths, that the model
holds no state beyond its parameters, and -- the one that matters most --
that the whole network reproduces the original bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.rade_qnet.core.contract.signature import InputSignature, TensorSpec
from src.rade_qnet.core.lifecycle.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.model import HybridGnnRnn
from src.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec
from src.rade_qnet.testkit.parity import compare_forward, load_golden

#: The golden fixture captured from the original implementation.
FIXTURE = "hybrid_gnn_rnn"

#: Shapes of the captured batch, so a signature can be built without the
#: data module. The parity test asserts these against the fixture rather
#: than trusting them.
N_NODES, N_ATTRIBUTES = 11, 13
N_ELEMENTARY, SEQUENCE, N_TARGETS = 8, 4, 3
N_EDGES, BATCH = 66, 16


def signature(
    *,
    n_nodes: int = N_NODES,
    n_attributes: int = N_ATTRIBUTES,
    n_elementary: int = N_ELEMENTARY,
    n_targets: int = N_TARGETS,
) -> InputSignature:
    """Build a signature matching the fixture's batch."""
    return InputSignature(
        dynamic={"pnl_history": TensorSpec(shape=(None, SEQUENCE, n_elementary), dtype="float32")},
        static={
            "trade_features": TensorSpec(shape=(n_nodes, n_attributes), dtype="float32"),
            "adjacency_indices": TensorSpec(shape=(N_EDGES, 2), dtype="int64"),
            "adjacency_values": TensorSpec(shape=(N_EDGES,), dtype="float32"),
            "adjacency_shape": TensorSpec(shape=(2,), dtype="int64"),
            "target_indices": TensorSpec(shape=(n_targets,), dtype="int64"),
        },
        target=TensorSpec(shape=(None, n_targets), dtype="float32"),
    )


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """Return the captured training batch, keyed for the forward pass."""
    golden = load_golden(FIXTURE)
    arrays = golden.arrays("level2_tensors/train_batch_000.npz")
    renamed = {"adjacency_dense_shape": "adjacency_shape"}
    # `elementary_indices` is dropped, not renamed. The original passed it
    # into every batch and never read it; the refactored signature does not
    # declare it, and the parity test below is what proves the omission
    # changes no number. Keeping it here would hide that.
    discarded = {"target", "elementary_indices"}
    return {
        renamed.get(name, name): torch.tensor(value)
        for name, value in arrays.items()
        if name not in discarded
    }


@pytest.fixture
def model() -> HybridGnnRnn:
    """Return the network at the original's production width."""
    return HybridGnnRnn(HybridModelSpec(units=16), signature())


class TestConstruction:
    """What the signature determines."""

    def test_every_width_comes_from_the_signature(self) -> None:
        """
        Nothing is assumed about the data's shape.

        A width hard-coded here would make the model unusable on a
        cluster of a different size, which is the single most common
        thing to vary across a job set.
        """
        network = HybridGnnRnn(
            HybridModelSpec(units=16),
            signature(n_nodes=20, n_attributes=9, n_elementary=5, n_targets=4),
        )
        assert network.gnn_block.gnn_layers[0].fusion_dense.in_features == 3 * 9
        assert network.rnn_block.rnn.input_size == 5
        assert network.projection_layer._baseline_kernels.shape[0] == 4

    def test_the_model_is_fully_parameterised_before_any_forward_pass(
        self, model: HybridGnnRnn
    ) -> None:
        """
        Defect 6, at the level of the whole network.

        An optimiser constructed over a model with lazy parameters tracks
        only the eager ones. The lazy ones then materialise on the first
        forward call and are never updated -- and the loss still falls,
        through the layers that were tracked, so nothing reports a fault.
        """
        assert all(
            not isinstance(parameter, torch.nn.UninitializedParameter)
            for parameter in model.parameters()
        )

    def test_a_missing_static_input_is_refused(self) -> None:
        """
        With a message naming what is missing and what was declared.

        The alternative is a ``KeyError`` deep in a forward pass, which
        names the key but not which side of the contract was wrong.
        """
        declared = signature()
        partial = InputSignature(
            dynamic=declared.dynamic,
            static={
                name: spec for name, spec in declared.static.items() if name != "trade_features"
            },
            target=declared.target,
        )
        with pytest.raises(ContractError, match="trade_features"):
            HybridGnnRnn(HybridModelSpec(units=16), partial)

    def test_a_variable_width_is_refused(self) -> None:
        """
        Only the batch axis may vary.

        A variable feature axis is not something to default around: it
        means the data module and the network disagree about what is
        fixed, and guessing would build a layer of some arbitrary size.
        """
        declared = signature()
        vague = InputSignature(
            dynamic={"pnl_history": TensorSpec(shape=(None, SEQUENCE, None), dtype="float32")},
            static=declared.static,
            target=declared.target,
        )
        with pytest.raises(ContractError, match="variable"):
            HybridGnnRnn(HybridModelSpec(units=16), vague)


@pytest.mark.usefixtures("requires_golden")
class TestForward:
    """What the assembled network computes."""

    def test_one_prediction_per_target_per_sample(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """The output is the shape the target declares."""
        model.eval()
        with torch.no_grad():
            assert model(**batch).shape == (BATCH, N_TARGETS)

    def test_every_parameter_receives_a_gradient(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        All 62 tensors are connected to the output.

        Across five blocks this is the check that a whole block has not
        been constructed, counted in the parameter total, and then left
        out of the forward pass.
        """
        model(**batch).sum().backward()
        unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        assert not unused

    def test_both_streams_reach_the_prediction(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        Perturbing the attributes or the history both move the output.

        A hybrid model in which one stream is disconnected still trains,
        still scores reasonably, and is not the model anyone signed off.
        """
        model.eval()
        with torch.no_grad():
            base = model(**batch)
            attributes = dict(batch)
            attributes["trade_features"] = attributes["trade_features"] + 1.0
            history = dict(batch)
            history["pnl_history"] = history["pnl_history"] + 1.0
            assert not torch.allclose(base, model(**attributes))
            assert not torch.allclose(base, model(**history))


@pytest.mark.usefixtures("requires_golden")
class TestNoHiddenState:
    """The model holds parameters and nothing else."""

    def test_two_forward_passes_agree(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        Calling twice gives the same answer.

        The original cached its graph embedding on the module and
        invalidated the cache on a mode change but not on a parameter
        change. A model that remembers anything between calls can return
        a stale number that looks entirely plausible.
        """
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(**batch), model(**batch))

    def test_nothing_is_stashed_on_the_module(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        The attribute set is unchanged by a forward pass.

        Model-held mutable state is how two jobs sharing a process see
        each other's results, and it is invisible until the day two jobs
        share a process.
        """
        model.eval()
        before = set(vars(model))
        with torch.no_grad():
            model(**batch)
        assert set(vars(model)) == before

    def test_the_precomputed_path_matches_the_ordinary_one(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        The capability replaces the cache without changing the answer.

        Reusing the graph embedding across an evaluation pass is a large
        saving -- the graph stream is recomputed identically for every
        batch otherwise -- but only if it is the same computation.
        """
        static = {name: value for name, value in batch.items() if name != "pnl_history"}
        model.eval()
        with torch.no_grad():
            direct = model(**batch)
            reused = model.forward_with_precomputed(batch, model.precompute(static))
        torch.testing.assert_close(direct, reused)

    def test_the_model_declares_that_it_handles_unseen_entities(self, model: HybridGnnRnn) -> None:
        """
        The capability is declared, and the output head backs it up.

        Declaring it without the head's borrowing machinery would let an
        inference pipeline hand the model a trade it cannot price.
        """
        assert model.supports_unseen_entities


@pytest.mark.usefixtures("requires_golden")
class TestParityAgainstTheBaseline:
    """Level 3: the forward pass, against the captured original."""

    def test_the_parameter_tensors_match_by_name_and_shape(self, model: HybridGnnRnn) -> None:
        """
        All 62 of them.

        Names as well as shapes, because the saved weights are restored
        by name: a port whose shapes matched but whose names differed
        would build a model that cannot load its own predecessor's
        checkpoint, and the failure would surface only on reload.
        """
        golden = load_golden(FIXTURE)
        expected = torch.load(golden.directory / "level3_forward/state_dict.pt", weights_only=True)
        produced = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        assert produced == {name: tuple(t.shape) for name, t in expected.items()}

    def test_the_forward_output_matches(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        The whole network, on the captured batch, under the original's weights.

        This is the test the entire port exists to pass. It is checked
        under ``eval`` and ``no_grad``, because the fixture was captured
        that way -- dropout active would make the comparison a record of
        the global random state rather than of the model.
        """
        golden = load_golden(FIXTURE)
        model.load_state_dict(
            torch.load(golden.directory / "level3_forward/state_dict.pt", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            output = model(**batch).numpy()

        report = compare_forward(np.asarray(output, dtype=np.float64), golden)
        assert report.passed, report.describe()
