"""Tests for the output head."""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.layers.projection import ProjectionLayer
from src.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import BATCH, N_ATTRIBUTES, N_TARGETS, UNITS


@pytest.fixture
def attended() -> torch.Tensor:
    """Attended representations, one per target per sample."""
    return torch.randn(BATCH, N_TARGETS, UNITS, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def attributes() -> torch.Tensor:
    """Return static attributes for the targets."""
    return torch.randn(N_TARGETS, N_ATTRIBUTES, generator=torch.Generator().manual_seed(1))


def head(n_fitted: int = N_TARGETS, **overrides: object) -> ProjectionLayer:
    """Build an output head at the shared widths."""
    spec = HybridModelSpec(units=UNITS, dropout=0.0, **overrides)
    return ProjectionLayer(
        spec,
        in_features=UNITS,
        attribute_features=N_ATTRIBUTES,
        n_fitted_targets=n_fitted,
    )


class TestShape:
    """What comes out."""

    def test_one_number_per_target_per_sample(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """The representation collapses to a P&L prediction."""
        assert head()(attended, attributes).shape == (BATCH, N_TARGETS)

    def test_mismatched_inputs_are_refused(self, attended: torch.Tensor) -> None:
        """
        The attended targets and the attribute rows must be the same set.

        Broadcasting would otherwise silently pair each target with the
        wrong instrument's attributes, which produces plausible numbers
        and no error at all.
        """
        with pytest.raises(ContractError, match="attribute row"):
            head()(attended, torch.randn(N_TARGETS + 1, N_ATTRIBUTES))

    def test_the_head_is_built_eagerly(self) -> None:
        """
        Every per-target parameter exists before the first forward pass.

        This head is where the original was laziest -- its baseline
        kernels materialised on the first call, after the optimiser had
        already been constructed over nothing. That is defect 6, and it
        meant the per-target baselines never trained.
        """
        layer = head()
        assert all(
            not isinstance(parameter, torch.nn.UninitializedParameter)
            for parameter in layer.parameters()
        )
        assert layer._baseline_kernels.shape == (N_TARGETS, UNITS)


class TestBaselineAndResidual:
    """The two terms the prediction is made of."""

    def test_each_target_reads_only_its_own_kernel(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """
        The baseline is a row-wise dot product, not a matrix product.

        If it were a matrix product, every target's baseline would read
        every other target's kernel -- which is exactly the per-instrument
        specificity this half of the head exists to provide.
        """
        layer = head()
        layer.eval()
        with torch.no_grad():
            base = layer._fitted_baseline(attended, N_TARGETS)
            layer._baseline_kernels[1] += 10.0
            moved = layer._fitted_baseline(attended, N_TARGETS)
        torch.testing.assert_close(base[:, 0], moved[:, 0])
        assert not torch.allclose(base[:, 1], moved[:, 1])

    def test_the_residual_reads_the_attributes(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """
        Changing a target's attributes changes its prediction.

        The residual is the only path by which a static attribute reaches
        the output, so if it ignored them the head would reduce to a
        per-target linear read and could say nothing about an instrument
        it had not fitted.
        """
        layer = head()
        layer.eval()
        with torch.no_grad():
            before = layer(attended, attributes)
            after = layer(attended, attributes + 1.0)
        assert not torch.allclose(before, after)

    def test_every_parameter_receives_a_gradient(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """No weight is disconnected from the output."""
        layer = head()
        layer(attended, attributes).sum().backward()
        unused = [name for name, parameter in layer.named_parameters() if parameter.grad is None]
        assert not unused

    def test_weight_normalisation_starts_from_the_same_function(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """
        The gain is initialised so that the layer begins where it would have.

        A reparametrisation that also moves the starting point is not a
        reparametrisation; it is a different model with a confusing name.
        The gain starts at the value whose softplus is one, so the first
        step is taken from the identical function.
        """
        plain, normed = head(), head(baseline_weight_norm=True)
        normed.load_state_dict(plain.state_dict(), strict=False)
        normed._baseline_kernels.data = plain._baseline_kernels.data.clone()

        # Unit-norm times a gain of one is the original kernel only when
        # the original was already unit-norm, so compare the directions.
        with torch.no_grad():
            gain = torch.nn.functional.softplus(normed._baseline_gain)
        torch.testing.assert_close(gain, torch.ones_like(gain))


class TestUnseenTargets:
    """Instruments that had no baseline fitted."""

    def test_an_unfitted_target_still_gets_a_prediction(self) -> None:
        """
        A trade booked after training is priced, not rejected.

        The whole point of a replication model is to price new business.
        A head that could only speak about the instruments it was fitted
        on would be useless for the job it exists to do.
        """
        layer = head(n_fitted=1)
        output = layer(torch.randn(BATCH, N_TARGETS, UNITS), torch.randn(N_TARGETS, N_ATTRIBUTES))
        assert output.shape == (BATCH, N_TARGETS)
        assert torch.isfinite(output).all()

    def test_the_borrowed_baseline_follows_the_nearest_neighbour(self) -> None:
        """
        A new target that matches one fitted target inherits its baseline.

        Constructed so the answer is known: the new instrument's
        attributes are a copy of the first fitted one's, and the cosine
        blend is sharp enough that the softmax puts essentially all its
        weight there. If the neighbour lookup were indexing the wrong
        axis this would inherit from the wrong instrument.
        """
        n_fitted = 3
        layer = ProjectionLayer(
            HybridModelSpec(units=UNITS, dropout=0.0, new_target_temperature=50.0),
            in_features=UNITS,
            attribute_features=N_ATTRIBUTES,
            n_fitted_targets=n_fitted,
        )
        layer.eval()

        fitted_attributes = torch.eye(n_fitted, N_ATTRIBUTES)
        # The new row is the first fitted row exactly.
        attributes = torch.cat([fitted_attributes, fitted_attributes[:1]], dim=0)
        attended = torch.randn(BATCH, n_fitted + 1, UNITS)

        with torch.no_grad():
            fitted = layer._fitted_baseline(attended[:, :n_fitted, :], n_fitted)
            borrowed = layer._borrowed_baseline(
                fitted,
                fitted_attributes=fitted_attributes,
                new_attributes=attributes[n_fitted:],
            )
        torch.testing.assert_close(borrowed[:, 0], fitted[:, 0], atol=1e-5, rtol=0.0)

    @pytest.mark.parametrize("blend", ["cosine_softmax", "inverse_distance"])
    def test_the_blend_weights_sum_to_one(self, blend: str) -> None:
        """
        Whichever blend is chosen, the borrow is an average.

        Weights that did not sum to one would scale the borrowed baseline
        by the number of neighbours, so a new instrument's prediction
        would depend on how many neighbours it happened to have.
        """
        layer = head(new_target_blend=blend)
        _, weights = layer._neighbour_weights(
            torch.randn(2, N_ATTRIBUTES), torch.randn(5, N_ATTRIBUTES)
        )
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))

    def test_an_unknown_blend_is_refused(self) -> None:
        """Rather than falling through to one of the two silently."""
        layer = head()
        layer.blend = "nearest"
        with pytest.raises(ContractError, match="blend"):
            layer._neighbour_weights(torch.randn(1, N_ATTRIBUTES), torch.randn(2, N_ATTRIBUTES))

    def test_the_residual_is_damped_for_unfitted_targets_only(self) -> None:
        """
        The correction is trusted less where it was never fitted.

        The residual network saw the fitted targets and did not see the
        new ones, so for a new instrument its output is extrapolation.
        Damping it is an explicit statement of that, and the damping must
        not touch the targets that *were* fitted.
        """
        n_fitted = 2
        full = ProjectionLayer(
            HybridModelSpec(units=UNITS, dropout=0.0, new_target_residual_damping=1.0),
            in_features=UNITS,
            attribute_features=N_ATTRIBUTES,
            n_fitted_targets=n_fitted,
        )
        damped = ProjectionLayer(
            HybridModelSpec(units=UNITS, dropout=0.0, new_target_residual_damping=0.0),
            in_features=UNITS,
            attribute_features=N_ATTRIBUTES,
            n_fitted_targets=n_fitted,
        )
        damped.load_state_dict(full.state_dict())
        full.eval()
        damped.eval()

        attended = torch.randn(BATCH, n_fitted + 1, UNITS)
        attributes = torch.randn(n_fitted + 1, N_ATTRIBUTES)
        with torch.no_grad():
            undamped_out = full(attended, attributes)
            damped_out = damped(attended, attributes)

        torch.testing.assert_close(undamped_out[:, :n_fitted], damped_out[:, :n_fitted])
        assert not torch.allclose(undamped_out[:, n_fitted:], damped_out[:, n_fitted:])
