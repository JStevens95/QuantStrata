"""
Compresses a window of P&L history into one vector per sample.

Where the GNN answers "what is this instrument like", this block answers
"what has the book been doing lately". It consumes the scaled P&L of every
selected elementary instrument over the window and returns a single
fixed-width summary of the whole period.

Why a recurrence rather than a flattened window
------------------------------------------------
The obvious alternative is to flatten the window into one long vector and
feed it to a dense layer. That works, and it has two properties that make
it the wrong default here.

It makes the parameter count scale with the window length, so lengthening
the history from four days to twenty multiplies the first layer by five and
the model starts overfitting the length rather than learning from it. And
it gives the model no notion of ordering: a flattened window is a bag of
numbers, so the network has to learn from scratch that column 3 is one day
after column 2. A recurrence has that structure built in and its parameter
count does not depend on the window length at all.

Why the last hidden state
-------------------------
The recurrence emits one output per timestep; this block keeps only the
final hidden state. That state is the network's running summary of
everything it has read, and it is the only one conditioned on the whole
window. Taking the mean over timesteps instead would weight the oldest day
as heavily as the most recent, which for predicting tomorrow is backwards.

For a bidirectional recurrence the final states of both directions are
concatenated, which is why that setting doubles the output width -- the
fusion block accounts for it rather than assuming a width.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import HybridModelSpec

__all__ = ["RnnBlock"]

#: Recurrences whose final state is a ``(hidden, cell)`` pair rather than a
#: bare tensor. GRU keeps no cell state, so it returns the tensor directly.
_CARRIES_CELL_STATE = frozenset({"lstm", "bilstm"})


class RnnBlock(nn.Module):
    """
    A stacked recurrence over the P&L window.

    Parameters
    ----------
    spec
        The model specification.
    in_features
        How many instruments the window carries, taken from the input
        signature so that the recurrence is sized at build time rather
        than materialised by a first forward pass -- defect 6.

    Notes
    -----
    Stacking is delegated to PyTorch's ``num_layers`` rather than looped
    here. That is not only shorter: the fused multi-layer kernel is
    substantially faster than a Python loop over single-layer modules, and
    on CUDA it is the difference between using cuDNN's recurrent kernels and
    not.
    """

    def __init__(self, spec: HybridModelSpec, *, in_features: int) -> None:
        super().__init__()
        self.rnn_type = spec.rnn_type
        width = spec.width("rnn")

        # PyTorch applies this dropout *between* stacked layers, so it is
        # meaningless with one layer and warns if passed anyway.
        between_layers = spec.dropout if spec.rnn_layers > 1 else 0.0

        shared = {
            "input_size": in_features,
            "hidden_size": width,
            "num_layers": spec.rnn_layers,
            # The window arrives as (batch, time, instrument), which is the
            # layout the data module produces and the one a reader expects.
            "batch_first": True,
            "dropout": between_layers,
        }
        if spec.rnn_type == "lstm":
            self.rnn: nn.Module = nn.LSTM(**shared)
        elif spec.rnn_type == "bilstm":
            self.rnn = nn.LSTM(**shared, bidirectional=True)
        elif spec.rnn_type == "gru":
            self.rnn = nn.GRU(**shared)
        else:
            raise ContractError(
                f"unknown recurrence type {spec.rnn_type!r}; the model implements "
                f"'lstm', 'bilstm' and 'gru'"
            )

    @property
    def out_features(self) -> int:
        """
        Width of the summary this block produces.

        Doubled for a bidirectional recurrence, since the two directions'
        final states are concatenated. Exposed as a property so the fusion
        block can size itself from it rather than re-deriving the doubling
        rule and getting it wrong once.
        """
        width = self.rnn.hidden_size
        return width * 2 if self.rnn_type == "bilstm" else width

    def forward(self, history: Tensor) -> Tensor:
        """
        Summarise the window.

        Parameters
        ----------
        history
            Scaled P&L, shape ``(batch, window, n_instruments)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_features)``.
        """
        _, final = self.rnn(history)

        # An LSTM returns (hidden, cell); the cell state is the memory the
        # gates write to, not the output, so only the hidden state leaves.
        hidden = final[0] if self.rnn_type in _CARRIES_CELL_STATE else final

        if self.rnn_type == "bilstm":
            # `hidden` is (num_layers * 2, batch, width), with the two
            # directions of the final layer in the last two positions:
            # -2 is the forward pass, -1 the backward one.
            return torch.cat([hidden[-2], hidden[-1]], dim=-1)
        # The top layer's state, which is the one conditioned on everything
        # the stack has read.
        return hidden[-1]
