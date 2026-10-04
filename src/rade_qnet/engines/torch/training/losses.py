"""
The loss registry.

A loss is named in the specification and looked up here, rather than chosen by
an ``if`` chain inside the learner. That buys two things worth the indirection:
a loss can be registered by a model package without touching the engine, and
an unrecognised name fails when the spec is read -- listing what is available
-- instead of partway into the first epoch.

The asymmetric losses, and why they are in the framework
--------------------------------------------------------
``asymmetric`` and ``quantile`` look domain-specific, but neither is: each is
a general statement about how to weigh errors -- over-prediction against
under-prediction, or one quantile of the error distribution -- and both are
used well outside P&L work.

They live here because a loss is consumed by the *learner*, and the learner
is engine code. A loss that only one model needs can still be registered from
that model's package without touching this module; the registry is what makes
that possible.

What is problem-specific is the *choice* of loss and the value of its
asymmetry, and those live in a configuration file where they belong.

Both reduce to a mean over the batch rather than a sum, so a loss value is
comparable between runs with different batch sizes. A sum is not: halving the
batch size halves the reported loss while changing nothing about the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from ....core.runtime.errors import EngineError
from ....core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "LossFunction",
    "asymmetric_loss",
    "build_loss",
    "quantile_loss",
    "register_loss",
    "registered_losses",
]

_LOGGER = get_logger(__name__)

#: Default penalty multiplier applied to under-prediction by ``asymmetric``.
#: Chosen as a neutral-but-visible default: 1.0 would make the loss identical
#: to ``mae`` and hide the fact that an asymmetric loss was configured at all.
_DEFAULT_UNDER_PENALTY = 2.0

#: Default quantile for ``quantile``. The median, which makes the loss equal
#: to half the mean absolute error and so gives a recognisable baseline.
_DEFAULT_QUANTILE = 0.5


class LossFunction(Protocol):
    """
    A loss: predictions and a target in, one scalar out.

    A protocol rather than a type alias so that a registered loss carrying
    configuration -- a quantile level, an asymmetry factor -- satisfies the
    same contract as a plain function.
    """

    def __call__(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Return the loss for one batch.

        Parameters
        ----------
        predictions
            Model output.
        target
            Observed values, broadcastable to ``predictions``.

        Returns
        -------
        torch.Tensor
            A zero-dimensional tensor, so that ``backward`` can be called on
            it without a further reduction.
        """
        ...


def asymmetric_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    under_penalty: float = _DEFAULT_UNDER_PENALTY,
) -> torch.Tensor:
    """
    Absolute error with under-prediction penalised more heavily.

    The objective for a quantity where the two directions of error cost
    different amounts. Under-predicting a risk figure leaves a position
    unhedged; over-predicting it costs some funding. A symmetric loss asserts
    those are equally bad, which for a risk number is simply false.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.
    under_penalty
        Multiplier applied where the prediction falls below the target. One
        reduces this to the mean absolute error.

    Returns
    -------
    torch.Tensor
        Scalar mean loss.
    """
    error = target - predictions
    # `error > 0` is exactly the under-prediction case: the target exceeded
    # what was predicted.
    weight = torch.where(error > 0, torch.full_like(error, under_penalty), torch.ones_like(error))
    return (weight * error.abs()).mean()


def quantile_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile: float = _DEFAULT_QUANTILE,
) -> torch.Tensor:
    """
    Pinball loss, which fits a conditional quantile rather than a mean.

    Used when the useful output is an interval rather than a point: training
    at 0.05 and 0.95 gives a band, and a band is what a risk limit is set
    against.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.
    quantile
        The quantile to fit, strictly between zero and one.

    Returns
    -------
    torch.Tensor
        Scalar mean loss.

    Raises
    ------
    EngineError
        If the quantile is not in ``(0, 1)``. At zero or one the loss is
        one-sided and the minimiser is unbounded, so it would train to an
        extreme value rather than failing visibly.
    """
    if not 0.0 < quantile < 1.0:
        raise EngineError(
            f"quantile must be strictly between 0 and 1, received {quantile}; "
            f"at the boundary the loss is one-sided and has no finite minimiser"
        )
    error = target - predictions
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def _mse(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return the mean squared error.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.nn.functional.mse_loss(predictions, target)


def _mae(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return the mean absolute error.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.nn.functional.l1_loss(predictions, target)


def _huber(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return the Huber loss with the Torch default transition point.

    Squared near zero and absolute in the tails, which is the usual choice
    for a P&L series: it keeps the smooth gradient of a squared error without
    letting one outlying day dominate the epoch.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.nn.functional.huber_loss(predictions, target)


#: The registry. Mutable by `register_loss` only, so every entry arrives
#: through the duplicate check there.
_REGISTRY: dict[str, LossFunction] = {
    "mse": _mse,
    "mae": _mae,
    "huber": _huber,
    "asymmetric": asymmetric_loss,
    "quantile": quantile_loss,
}


def register_loss(name: str, loss: LossFunction, *, replace: bool = False) -> None:
    """
    Add a loss to the registry.

    Parameters
    ----------
    name
        Name the specification will use.
    loss
        The loss function.
    replace
        Whether to overwrite an existing entry.

    Raises
    ------
    EngineError
        If the name is taken and ``replace`` is false. Silently overwriting
        would mean two packages registering the same name produce different
        objectives depending on import order, which is a difference nothing
        would report.
    """
    if name in _REGISTRY and not replace:
        raise EngineError(
            f"a loss named {name!r} is already registered; pass replace=True to "
            f"override it deliberately"
        )
    _REGISTRY[name] = loss
    _LOGGER.debug("registered loss %r", name)


def registered_losses() -> tuple[str, ...]:
    """
    Return the registered loss names, sorted.

    Returns
    -------
    tuple of str
        Available names.
    """
    return tuple(sorted(_REGISTRY))


def build_loss(name: str, *, options: Mapping[str, object] | None = None) -> LossFunction:
    """
    Return a loss by name, with its options bound.

    Parameters
    ----------
    name
        A registered loss name.
    options
        Keyword arguments to bind -- ``quantile`` for the pinball loss,
        ``under_penalty`` for the asymmetric one. Bound here rather than
        passed per call, so the learner calls every loss identically.

    Returns
    -------
    LossFunction
        The loss, ready to call with predictions and a target.

    Raises
    ------
    EngineError
        If the name is unknown, listing what is available, or if an option is
        not accepted by the chosen loss -- which is usually a quantile set on
        a loss that has no quantile, and would otherwise be ignored in
        silence.
    """
    if name not in _REGISTRY:
        raise EngineError(
            f"unknown loss {name!r}; registered losses are {list(registered_losses())}"
        )

    loss = _REGISTRY[name]
    if not options:
        return loss

    import inspect  # noqa: PLC0415  Needed only on the configured-options path.

    accepted = set(inspect.signature(loss).parameters)
    rejected = sorted(set(options) - accepted)
    if rejected:
        raise EngineError(
            f"loss {name!r} does not accept option(s) {rejected}; it accepts "
            f"{sorted(accepted - {'predictions', 'target'})}"
        )

    def configured(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Return the configured loss for one batch.

        Parameters
        ----------
        predictions
            Model output.
        target
            Observed values.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return loss(predictions, target, **options)

    return configured
