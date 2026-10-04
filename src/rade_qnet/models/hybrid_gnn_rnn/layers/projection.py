"""
Turns one representation per target into one PnL number per target.

Everything upstream is shared machinery: the same GNN, the same recurrence,
the same attention, applied to every instrument. This layer is where the
model stops being general and becomes specific to the instruments in front
of it, and it is the only place in the network that holds per-instrument
parameters.

Baseline plus residual
----------------------
The prediction is a sum of two terms, and the split is the point of the
layer.

The *baseline* is a dedicated weight vector and bias per target: a linear
read of that one instrument's representation. It is the model saying "this
particular swaption's PnL is roughly this function of its state", and it is
what you would get from fitting each target separately.

The *residual* is a small shared network over the representation
concatenated with the instrument's static attributes. It corrects the
baseline using patterns learnt across the whole book -- the part a
per-instrument fit cannot see.

Together: per-instrument accuracy where there is history to fit, shared
structure where there is not. Neither alone is adequate. A purely shared
head cannot express that two instruments with near-identical attributes
behave differently; purely per-instrument baselines cannot say anything
at all about an instrument that was not in the training set.

Instruments that were not in the training set
----------------------------------------------
Which is the harder half of the problem. A trade booked after the model
was fitted has no baseline kernel, and the whole point of a replication
model is to price it anyway.

Targets arrive with the fitted ones first and the new ones after. A new
target borrows a baseline from its ``k`` nearest fitted targets in
attribute space -- a 10y EUR payer inherits from the other 10y EUR payers
-- and its residual is damped, because the residual network never saw it
and its correction is extrapolation rather than fit.

This is interpolation in attribute space, and it is only as good as the
attributes. If a new trade is genuinely unlike anything fitted, the blend
will return a confident-looking number built from poor neighbours. The
graph diagnostics are what tell you whether that is happening; this layer
cannot know.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import HybridModelSpec
from .gnn import activation_function

__all__ = ["ProjectionLayer"]

#: Guard against dividing by a zero norm or a zero distance.
_EPSILON = 1e-8

#: Keeps the inverse-distance weights from summing to zero.
_WEIGHT_SUM_FLOOR = 1e-12


class ProjectionLayer(nn.Module):
    """
    Per-target output head: a fitted baseline plus a shared residual correction.

    Parameters
    ----------
    spec
        The model specification.
    in_features
        Width of the attended representation.
    attribute_features
        Width of the static attribute vector.
    n_fitted_targets
        How many targets get a dedicated baseline. Sized here rather than
        on the first forward pass so that the optimiser sees every
        parameter when it is constructed.
    """

    def __init__(
        self,
        spec: HybridModelSpec,
        *,
        in_features: int,
        attribute_features: int,
        n_fitted_targets: int,
    ) -> None:
        super().__init__()
        self.n_fitted_targets = n_fitted_targets
        self.blend = spec.new_target_blend
        self.n_neighbours = spec.new_target_neighbours
        self.temperature = spec.new_target_temperature
        self.distance_power = spec.new_target_distance_power
        self.residual_damping = spec.new_target_residual_damping
        self.use_weight_norm = spec.baseline_weight_norm

        # One row per fitted target. Xavier rather than the default
        # initialisation because each row is read in isolation by a single
        # dot product, so its scale sets that target's output scale
        # directly.
        self._baseline_kernels = nn.Parameter(torch.empty(n_fitted_targets, in_features))
        nn.init.xavier_uniform_(self._baseline_kernels)
        self._baseline_biases = nn.Parameter(torch.zeros(n_fitted_targets))

        # Initialised so softplus(gain) == 1, leaving the layer equivalent
        # to the plain dot product at step zero. Starting a reparametrised
        # layer anywhere else changes the function the optimiser begins
        # from, which is not what a reparametrisation is for.
        self._baseline_gain = (
            nn.Parameter(torch.full((n_fitted_targets,), math.log(math.expm1(1.0))))
            if spec.baseline_weight_norm
            else None
        )

        width = spec.width("projection")
        self._residual_fc_1 = nn.Linear(in_features + attribute_features, width)
        self._residual_fc_2 = nn.Linear(width, 1)
        self._activation = activation_function(spec.projection_activation)
        self._residual_dropout = nn.Dropout(spec.dropout) if spec.dropout > 0.0 else None

    def forward(self, attended: Tensor, attributes: Tensor) -> Tensor:
        """
        Predict PnL for every target.

        Parameters
        ----------
        attended
            Attended representations, shape ``(batch, n_targets, in_features)``.
        attributes
            Static attributes of the targets, shape
            ``(n_targets, attribute_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets)``, in scaled space.

        Raises
        ------
        ContractError
            If the two inputs disagree on how many targets there are.
        """
        n_samples, n_targets, _ = attended.shape
        if attributes.shape[0] != n_targets:
            raise ContractError(
                f"{n_targets} attended target(s) but {attributes.shape[0]} attribute "
                f"row(s); the attribute table must be restricted to the targets"
            )

        # Attributes are shared across the batch, so `expand` gives a view
        # rather than materialising a copy per sample.
        residual = self._residual_fc_1(
            torch.cat([attended, attributes.unsqueeze(0).expand(n_samples, -1, -1)], dim=-1)
        )
        residual = self._activation(residual)
        if self._residual_dropout is not None:
            residual = self._residual_dropout(residual)
        residual = self._residual_fc_2(residual).squeeze(-1)

        # Fewer targets than kernels is legitimate: a cluster can be
        # evaluated on a subset of what it was fitted on.
        n_fitted = min(self.n_fitted_targets, n_targets)
        fitted_baseline = self._fitted_baseline(attended[:, :n_fitted, :], n_fitted)

        n_new = n_targets - n_fitted
        if n_new > 0:
            new_baseline = self._borrowed_baseline(
                fitted_baseline,
                fitted_attributes=attributes[:n_fitted, :],
                new_attributes=attributes[n_fitted:, :],
            )
            residual = torch.cat(
                [residual[:, :n_fitted], residual[:, n_fitted:] * self.residual_damping],
                dim=1,
            )
        else:
            new_baseline = attended.new_zeros((n_samples, 0))

        return torch.cat([fitted_baseline, new_baseline], dim=1) + residual

    def _fitted_baseline(self, attended: Tensor, n_fitted: int) -> Tensor:
        """
        Read each fitted target's own kernel.

        Parameters
        ----------
        attended
            Shape ``(batch, n_fitted, in_features)``.
        n_fitted
            How many kernels to use.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_fitted)``.
        """
        kernels = self._baseline_kernels[:n_fitted, :]
        if self._baseline_gain is not None:
            direction = kernels / (torch.norm(kernels, dim=1, keepdim=True) + _EPSILON)
            kernels = direction * nn.functional.softplus(self._baseline_gain[:n_fitted]).unsqueeze(
                1
            )
        # einsum rather than a matmul because each target reads only its
        # own kernel: this is a row-wise dot product, not a matrix product.
        return (
            torch.einsum("bna,na->bn", attended, kernels) + self._baseline_biases[None, :n_fitted]
        )

    def _borrowed_baseline(
        self,
        fitted_baseline: Tensor,
        *,
        fitted_attributes: Tensor,
        new_attributes: Tensor,
    ) -> Tensor:
        """
        Blend fitted targets' baselines for targets that have none.

        Parameters
        ----------
        fitted_baseline
            Shape ``(batch, n_fitted)``.
        fitted_attributes, new_attributes
            Attribute rows, shape ``(n_fitted, p)`` and ``(n_new, p)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_new)``.
        """
        neighbours, weights = self._neighbour_weights(new_attributes, fitted_attributes)
        # The blend is over baselines, not over kernels. Mixing kernels
        # and then reading the new target's own representation would
        # apply a neighbour's weights to a different instrument's state;
        # mixing outputs asks "what would my neighbours have predicted",
        # which is the question actually being answered.
        return torch.sum(fitted_baseline[:, neighbours] * weights[None, :, :], dim=-1)

    def _neighbour_weights(
        self, new_attributes: Tensor, fitted_attributes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Find each new target's nearest fitted targets and how much to weight them.

        Parameters
        ----------
        new_attributes, fitted_attributes
            Attribute rows.

        Returns
        -------
        tuple of torch.Tensor
            Neighbour indices and weights, both ``(n_new, k)``; the weights
            sum to one along the last axis.

        Raises
        ------
        ContractError
            If the blend mode is not one this layer implements.
        """
        k = min(self.n_neighbours, fitted_attributes.shape[0])

        if self.blend == "cosine_softmax":
            # Cosine rather than Euclidean: attribute vectors carry one-hot
            # blocks whose magnitude reflects how many categories a field
            # has, so direction is comparable across instruments where
            # distance is not.
            similarity = (
                nn.functional.normalize(new_attributes, p=2, dim=1)
                @ nn.functional.normalize(fitted_attributes, p=2, dim=1).T
            )
            scores, neighbours = torch.topk(similarity, k=k, dim=1, sorted=False)
            return neighbours, torch.softmax(scores * self.temperature, dim=1)

        if self.blend == "inverse_distance":
            distances = torch.cdist(new_attributes, fitted_attributes)
            nearest, neighbours = torch.topk(-distances, k=k, dim=1, sorted=False)
            raw = 1.0 / torch.clamp(-nearest, min=_EPSILON).pow(self.distance_power)
            return neighbours, raw / (raw.sum(dim=1, keepdim=True) + _WEIGHT_SUM_FLOOR)

        raise ContractError(
            f"unknown new-target blend {self.blend!r}; expected 'cosine_softmax' "
            f"or 'inverse_distance'"
        )
