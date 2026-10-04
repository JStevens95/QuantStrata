"""Tests for the recurrent block."""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.lifecycle.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.layers.rnn import RnnBlock
from src.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import BATCH, N_ELEMENTARY, SEQUENCE, UNITS


class TestShape:
    """What comes out, for each recurrence type."""

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
    def test_one_summary_per_sample(self, rnn_type: str, pnl_history: torch.Tensor) -> None:
        """
        The window collapses to a single vector.

        There is no node axis: the recent period is the same period for
        every instrument, and giving each one its own copy here would
        multiply the work by the universe size for no information gained.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS, rnn_type=rnn_type), in_features=N_ELEMENTARY)
        assert block(pnl_history).shape == (BATCH, UNITS)

    def test_a_bidirectional_recurrence_is_twice_as_wide(self, pnl_history: torch.Tensor) -> None:
        """
        Both directions' final states are kept, so the output doubles.

        The width is read from the block rather than assumed by whatever
        consumes it, which is why ``out_features`` exists. A consumer that
        assumed ``units`` would build a layer of the wrong shape and fail
        only on the first forward pass.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS, rnn_type="bilstm"), in_features=N_ELEMENTARY)
        assert block.out_features == 2 * UNITS
        assert block(pnl_history).shape == (BATCH, 2 * UNITS)

    def test_an_unknown_recurrence_is_refused(self) -> None:
        """
        Rather than defaulting to one and reporting another.

        The spec's literal already excludes this, so reaching it means a
        spec was bypassed -- and silently substituting a GRU for a
        requested temporal convolution would be worse than failing.
        """
        spec = HybridModelSpec(units=UNITS)
        object.__setattr__(spec, "rnn_type", "tcn")
        with pytest.raises(ContractError, match="tcn"):
            RnnBlock(spec, in_features=N_ELEMENTARY)


class TestBehaviour:
    """What the block does with its input."""

    def test_every_parameter_receives_a_gradient(self, pnl_history: torch.Tensor) -> None:
        """No weight is disconnected from the output."""
        block = RnnBlock(HybridModelSpec(units=UNITS), in_features=N_ELEMENTARY)
        block(pnl_history).sum().backward()
        unused = [name for name, parameter in block.named_parameters() if parameter.grad is None]
        assert not unused

    def test_the_summary_depends_on_the_order_of_the_window(
        self, pnl_history: torch.Tensor
    ) -> None:
        """
        Reversing time changes the answer.

        The whole claim of the recurrent stream is that *when* a move
        happened matters. A block whose output survived a time reversal
        would be an expensive way to compute a summary statistic, and
        nothing else in the suite would notice.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS), in_features=N_ELEMENTARY)
        block.eval()
        with torch.no_grad():
            forward = block(pnl_history)
            backward = block(pnl_history.flip(dims=(1,)))
        assert not torch.allclose(forward, backward)

    def test_the_last_step_of_the_window_matters_most(self, pnl_history: torch.Tensor) -> None:
        """
        Perturbing the most recent day moves the summary more than the oldest.

        Not a law of recurrences -- an untrained network could behave
        either way -- but at initialisation an LSTM's forget gate damps
        earlier steps, and if that ordering inverted it would mean the
        window is being read backwards. That is the bug this catches.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS), in_features=N_ELEMENTARY)
        block.eval()
        with torch.no_grad():
            base = block(pnl_history)

            oldest = pnl_history.clone()
            oldest[:, 0, :] += 1.0
            newest = pnl_history.clone()
            newest[:, SEQUENCE - 1, :] += 1.0

            moved_by_oldest = (block(oldest) - base).abs().mean()
            moved_by_newest = (block(newest) - base).abs().mean()
        assert moved_by_newest > moved_by_oldest

    def test_dropout_between_layers_only_applies_to_a_stack(self) -> None:
        """
        A single-layer recurrence gets no inter-layer dropout.

        Torch warns and ignores it, which is noise in every log of every
        single-layer run. Suppressing it here rather than tolerating the
        warning means a real warning stays visible.
        """
        single = RnnBlock(
            HybridModelSpec(units=UNITS, rnn_layers=1, dropout=0.5),
            in_features=N_ELEMENTARY,
        )
        stacked = RnnBlock(
            HybridModelSpec(units=UNITS, rnn_layers=2, dropout=0.5),
            in_features=N_ELEMENTARY,
        )
        assert single.rnn.dropout == 0.0
        assert stacked.rnn.dropout == 0.5
