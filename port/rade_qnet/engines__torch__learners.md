# `src/rade_qnet/engines/torch/learners`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 29 | 944 | `dc9200473af4132e` |
| 2 | `supervised.py` | 290 | 10072 | `5306ebe1215c4405` |

---

## 1. `src/rade_qnet/engines/torch/learners/__init__.py`

944 bytes · SHA-256 `dc9200473af4132e`

```python
"""
Update rules -- what one optimisation step means.

A learner receives a batch and the current parameters and returns the losses
and metrics for that step.  It knows nothing about epochs, checkpoints,
schedules, logging or devices; the loop owns all of those.  This is what allows
a new algorithm to be added as one focused module.

Modules
-------
``supervised.py``
    Forward pass, loss against a target, backward pass.  [Phase 2]

Planned modules
---------------
``dqn.py``
    Temporal-difference update with a target network.  [Phase 7]
``ppo.py``
    Clipped surrogate objective with generalised advantage estimation.
    [Phase 7]
``sac.py``
    Soft actor-critic with entropy regularisation.  [Phase 7]
``pathwise.py``
    Back-propagates a risk measure directly through a differentiable
    simulation.  No value function and no policy-gradient estimator -- the
    gradient is exact.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/engines/torch/learners/supervised.py`

10072 bytes · SHA-256 `5306ebe1215c4405`

```python
"""
The supervised update rule: forward, loss, backward.

A learner knows what one optimisation step means and nothing else. It has no
notion of epochs, checkpoints, schedules, logging or devices -- the loop owns
all of those. That separation is what lets Phase 7 add four reinforcement-
learning learners against an unchanged loop.

Reconciling prediction and target shapes
----------------------------------------
:func:`align_with_target` exists because of a failure that is easy to
reproduce and hard to notice. A regression head usually emits ``(batch, 1)``
while the target is ``(batch,)``. Those two broadcast against each other, so
``mse_loss`` computes a ``(batch, batch)`` matrix of every prediction against
every target and returns its mean.

The run does not fail. It trains, the loss decreases, and the number it is
minimising is roughly the variance of the target plus the squared error --
dominated by a term the model cannot affect. The loss curve looks like a
model that learned a little and plateaued, which is also what a genuinely
mediocre model looks like.

Torch emits a warning. Warnings in a training loop that logs per epoch are
not read. So the shapes are reconciled explicitly when the element counts
match, and refused with a message naming both shapes when they do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....core.runtime.components import learner as register_learner
from ....core.runtime.errors import EngineError
from ....core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...torch.callbacks import GradientNorms
    from ...torch.hardware import ResolvedHardware
    from ...torch.losses import LossFunction

__all__ = ["SupervisedLearner", "align_with_target"]

_LOGGER = get_logger(__name__)

#: Reported when a step produced a non-finite loss, so the epoch summary can
#: show how many steps were lost rather than only a NaN mean.
_NON_FINITE_KEY = "non_finite_steps"


def align_with_target(
    predictions: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reshape predictions to the target's shape when they describe the same values.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    tuple
        Predictions and target, with matching shapes.

    Raises
    ------
    EngineError
        If the two hold different numbers of elements. Refused rather than
        broadcast, for the reason in the module docstring: broadcasting here
        produces a plausible loss curve for a meaningless objective.
    """
    if predictions.shape == target.shape:
        return predictions, target

    if predictions.numel() != target.numel():
        raise EngineError(
            f"the model produced {tuple(predictions.shape)} "
            f"({predictions.numel()} element(s)) but the target is "
            f"{tuple(target.shape)} ({target.numel()} element(s)). These cannot "
            f"describe the same predictions. Check the model's output dimension "
            f"against the target declared in the input signature"
        )

    # Same elements, different arrangement -- the `(batch, 1)` against
    # `(batch,)` case.  Reshaped rather than broadcast, which is the whole
    # point of this function.
    return predictions.reshape(target.shape), target


@register_learner("supervised", engine="torch")
class SupervisedLearner:
    """
    One gradient step against a target.

    Parameters
    ----------
    loss
        The objective, from the loss registry.
    optimiser
        The optimiser, built over materialised parameters.
    hardware
        Resolved device, precision and gradient scaler.
    gradient_clip_norm
        Global gradient-norm clip, or ``None`` to disable clipping.
    gradient_norms
        Tracker to record each step's norm in. ``None`` disables tracking.
    """

    def __init__(
        self,
        *,
        loss: LossFunction,
        optimiser: torch.optim.Optimizer,
        hardware: ResolvedHardware,
        gradient_clip_norm: float | None = None,
        gradient_norms: GradientNorms | None = None,
    ) -> None:
        self.loss = loss
        self.optimiser = optimiser
        self.hardware = hardware
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_norms = gradient_norms

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Run one forward and backward pass and update the parameters.

        Parameters
        ----------
        model
            The prepared model, in training mode.
        inputs
            Batch inputs, passed to ``forward`` as keyword arguments so that a
            model declares what it consumes in its own signature rather than
            depending on positional order.
        target
            Observed values for the batch.

        Returns
        -------
        dict
            ``loss`` for the step, plus a non-finite marker when the step was
            skipped.
        """
        # `set_to_none` releases the gradient tensors rather than filling them
        # with zeros.  It is the Torch default, named here because the
        # difference is visible: a zeroed gradient still participates in
        # weight decay, a released one does not.
        self.optimiser.zero_grad(set_to_none=True)

        with self.hardware.autocast():
            predictions = model(**inputs)
            aligned, observed = align_with_target(predictions, target)
            loss = self.loss(aligned, observed)

        if not torch.isfinite(loss):
            # Skipped rather than propagated.  A single NaN batch -- a bad row,
            # an unlucky initialisation -- would otherwise turn every
            # parameter into NaN on the backward pass, after which every
            # subsequent epoch is NaN and the cause is invisible.
            _LOGGER.warning(
                "skipping a step whose loss was %s; parameters left unchanged",
                loss.item(),
            )
            return {"loss": float(loss.item()), _NON_FINITE_KEY: 1.0}

        self._backward(loss)
        self._clip_and_record(model)
        self._step()
        return {"loss": float(loss.item())}

    def _backward(self, loss: torch.Tensor) -> None:
        """
        Run the backward pass, scaling the loss under ``fp16``.

        Parameters
        ----------
        loss
            The step's loss.
        """
        if self.hardware.scaler is not None:
            self.hardware.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_and_record(self, model: torch.nn.Module) -> None:
        """
        Clip the gradient norm and record it.

        Parameters
        ----------
        model
            The model whose gradients are being clipped.
        """
        if self.hardware.scaler is not None:
            # Unscaled first, or the recorded norm would be the scaled one and
            # the clip threshold would mean something different at every
            # scaler value.
            self.hardware.scaler.unscale_(self.optimiser)

        if self.gradient_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.gradient_clip_norm
            )
        elif self.gradient_norms is not None:
            # Measured without clipping, by asking for an effectively infinite
            # threshold.  `clip_grad_norm_` returns the pre-clip norm either
            # way, so this is the cheapest way to observe it.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        else:
            return

        if self.gradient_norms is not None:
            self.gradient_norms.record(float(norm))

    def _step(self) -> None:
        """Apply the optimiser update, through the scaler when one is in use."""
        if self.hardware.scaler is not None:
            self.hardware.scaler.step(self.optimiser)
            # Updates the scale factor, and in doing so decides whether the
            # step just attempted was skipped for overflow.
            self.hardware.scaler.update()
        else:
            self.optimiser.step()

    def eval_step(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute the loss for one batch without updating anything.

        Parameters
        ----------
        model
            The prepared model, in evaluation mode.
        inputs
            Batch inputs.
        target
            Observed values for the batch.

        Returns
        -------
        dict
            ``loss`` for the batch.
        """
        with torch.no_grad(), self.hardware.autocast():
            predictions = model(**inputs)
            aligned, observed = align_with_target(predictions, target)
            loss = self.loss(aligned, observed)
        return {"loss": float(loss.item())}

    def predict_step(
        self, model: torch.nn.Module, inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Produce predictions for one batch.

        Parameters
        ----------
        model
            The prepared model, in evaluation mode.
        inputs
            Batch inputs.

        Returns
        -------
        torch.Tensor
            Model output, detached and in full precision. Cast back from any
            reduced precision here rather than at the caller, because a
            ``bfloat16`` array reaching NumPy is an error and a ``float16``
            one silently loses three decimal digits of a P&L figure.
        """
        with torch.no_grad(), self.hardware.autocast():
            predictions = model(**inputs)
        return predictions.detach().to(dtype=torch.float32)
```

