"""
What the recurrent baseline computes: one recurrence and one linear head.

This file is pure PyTorch. It imports no registry, no run specification and
no pipeline, and it would work unchanged pasted into a notebook. That is
the tier 1 and tier 2 rule and it is worth stating plainly, because the
temptation in a model this small is to collapse everything into one file and
lose the property that the mathematics can be read, reviewed and unit-tested
without the framework in the way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from ...core.lifecycle.errors import ContractError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature
    from .spec import LstmTabularSpec

__all__ = ["LstmTabular", "build"]

#: Rank of a batched sequence input: ``(samples, timesteps, features)``.
_SEQUENCE_RANK = 3


class LstmTabular(nn.Module):
    """
    One recurrence and one linear head.

    Sized from the signature at construction rather than lazily, for the
    reason the flagship states at length: a model with no parameters yet
    hands an optimiser an empty parameter group, which trains nothing while
    reporting a decreasing loss from the layers that did exist.

    Parameters
    ----------
    settings
        The architecture specification.
    n_features
        Width of the input's feature axis.
    """

    def __init__(self, settings: LstmTabularSpec, *, n_features: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=settings.units,
            num_layers=settings.layers,
            dropout=settings.dropout if settings.layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(settings.units, 1)

    def forward(self, **inputs: Tensor) -> Tensor:
        """
        Read the window and predict one value per sample.

        Takes its inputs as keywords because that is the engine's calling
        convention, and by name rather than position because the flagship
        does -- a model with four inputs cannot afford positional ones.
        Variadic rather than a fixed ``features=`` parameter, because the
        input's name is the source's choice and this baseline is meant to
        run against whatever a user already has.

        The target is not among them: the loader removes it before calling,
        so there is nothing here to filter out.

        Parameters
        ----------
        **inputs
            The batch's input tensors.

        Returns
        -------
        torch.Tensor
            Shape ``(samples, 1)``.

        Raises
        ------
        ContractError
            If the batch carries no recognisable sequence input.
        """
        sequence = _sequence_of(inputs)
        output, _ = self.rnn(sequence)
        # The last timestep only. A window exists to summarise the recent
        # past into the present, so the state after reading all of it is the
        # summary; pooling across timesteps would blur the ordering the
        # recurrence was chosen for.
        return self.head(output[:, -1, :])


def build(settings: LstmTabularSpec, signature: InputSignature) -> LstmTabular:
    """
    Return an untrained network sized for the declared interface.

    Takes the signature rather than a bare width so that the sizing rule
    lives beside the architecture it sizes. ``register.py`` stays free of
    anything that could be called a modelling decision.

    Parameters
    ----------
    settings
        The validated architecture settings.
    signature
        The declared interface from the data build.

    Returns
    -------
    LstmTabular
        Untrained.

    Raises
    ------
    ContractError
        If the signature declares no dynamic input.
    """
    return LstmTabular(settings, n_features=_feature_width(signature))


def _sequence_of(inputs: Mapping[str, Tensor]) -> Tensor:
    """
    Take the batch's single input and give it a time axis.

    The input is taken rather than searched for. ``REQUIRES`` in this
    package's ``data.py`` declares exactly one dynamic input, and the
    pipeline checks that before the model is built, so by the time a batch
    arrives there is nothing to choose between.

    That is a deliberate change from what this function used to do, which
    was to scan the batch and return the first tensor it found. Handed two
    feature blocks it trained on whichever one the dictionary yielded
    first: reordering the data build changed the model, with no exception,
    no warning and a perfectly plausible loss curve. The count is asserted
    here as well as declared, because a silent wrong answer is worth two
    lines of belt and braces.

    A two-dimensional input -- one row per sample, no window -- is read as
    a window of length one rather than refused, which is the other half of
    what ``REQUIRES`` declares.

    Parameters
    ----------
    inputs
        One batch's inputs.

    Returns
    -------
    torch.Tensor
        Shape ``(samples, timesteps, features)``.

    Raises
    ------
    ContractError
        If the batch does not carry exactly one input tensor.
    """
    tensors = [value for value in inputs.values() if isinstance(value, Tensor)]
    if len(tensors) != 1:
        raise ContractError(
            f"a recurrence reads one sequence, but the batch carries "
            f"{len(tensors)} input tensor(s): {sorted(inputs)}. This should "
            f"have been caught when the signature was checked against "
            f"REQUIRES -- see rade_qnet.models.lstm_tabular.data"
        )
    sequence = tensors[0]
    if sequence.dim() == _SEQUENCE_RANK:
        return sequence
    # Insert the axis at position one, not with ``atleast_3d``, which
    # appends it: a ``(samples, features)`` input would become
    # ``(samples, features, 1)``, i.e. one timestep per *feature* of a
    # single scalar. That is a valid shape the recurrence accepts and a
    # silently wrong model, so the axis is placed explicitly.
    return sequence.unsqueeze(1)


def _feature_width(signature: InputSignature) -> int:
    """
    Return the width of the signature's single dynamic input.

    Parameters
    ----------
    signature
        The declared interface.

    Returns
    -------
    int
        Size of the last axis.

    Raises
    ------
    ContractError
        If the signature declares no dynamic input.
    """
    for tensor in signature.dynamic.values():
        return int(tensor.shape[-1])
    raise ContractError(
        "the signature declares no dynamic input, so there is nothing for a recurrence to read"
    )
