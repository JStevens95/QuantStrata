"""
Unit tests for the recurrent baseline's architecture.

Pure PyTorch, run directly against the network. No specification, no
registry, no engine -- which is the point of ``model.py`` being
framework-free, and is what makes the shape regression below a two-line
test instead of a training run.
"""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.models.lstm_tabular.model import LstmTabular, _sequence_of
from src.rade_qnet.models.lstm_tabular.spec import LstmTabularSpec


class TestTheNetworkReadsAWindow:
    """One recurrence, one head, one value per sample."""

    def test_a_windowed_batch_produces_one_prediction_per_sample(self) -> None:
        """The headline shape contract, asserted on the network itself."""
        network = LstmTabular(LstmTabularSpec(units=8), n_features=4)
        output = network(features=torch.zeros(5, 3, 4))
        assert output.shape == (5, 1)

    def test_an_unwindowed_batch_is_read_as_a_single_timestep(self) -> None:
        """
        A source with no sequence transform still runs.

        Degrading to a one-step recurrence makes this simply a worse
        model, which is a better outcome than a run that cannot start on
        data a user already has.
        """
        network = LstmTabular(LstmTabularSpec(units=8), n_features=4)
        assert network(features=torch.zeros(5, 4)).shape == (5, 1)

    def test_the_input_name_is_not_hard_coded(self) -> None:
        """
        The batch key is the source's choice, so the model may not assume it.

        A fixed ``features=`` parameter would work against every fixture in
        this repository and fail against the first user whose column block
        is called something else.
        """
        network = LstmTabular(LstmTabularSpec(units=8), n_features=4)
        assert network(whatever_the_source_called_it=torch.zeros(2, 3, 4)).shape == (
            2,
            1,
        )


class TestTheTimeAxisGoesInTheRightPlace:
    """
    A regression test for a bug that ran, trained, and was wrong.

    ``torch.atleast_3d`` *appends* the new axis, turning ``(samples,
    features)`` into ``(samples, features, 1)`` -- one timestep per feature
    of a single scalar. The recurrence accepts that shape happily. The
    model it produces is nonsense, and nothing in a training run says so.
    """

    def test_the_axis_is_inserted_rather_than_appended(self) -> None:
        """``(5, 4)`` must become ``(5, 1, 4)``, never ``(5, 4, 1)``."""
        assert _sequence_of({"x": torch.zeros(5, 4)}).shape == (5, 1, 4)

    def test_an_already_windowed_input_is_left_alone(self) -> None:
        """Three dimensions are already correct and must not be touched."""
        assert _sequence_of({"x": torch.zeros(5, 3, 4)}).shape == (5, 3, 4)

    def test_an_empty_batch_is_refused_with_an_explanation(self) -> None:
        """
        Silence here would surface as an unrelated error much later.

        The message names the keys that *were* present, because the usual
        cause is a source whose input block is called something the user
        did not expect.
        """
        with pytest.raises(ContractError, match="one sequence"):
            _sequence_of({})

    def test_two_inputs_are_refused_rather_than_silently_chosen_between(
        self,
    ) -> None:
        """
        The bug this model's input contract was introduced to kill.

        This used to return whichever tensor the dictionary yielded first,
        so reordering the data build changed which feature block the
        recurrence trained on -- no exception, no warning, a plausible
        loss curve and a different model.

        The contract in ``data.py`` now rejects the pairing before
        training starts; this asserts the model refuses it even if it
        somehow gets that far, because a silent wrong answer is worth
        catching twice.
        """
        with pytest.raises(ContractError, match="one sequence"):
            _sequence_of({"prices": torch.zeros(2, 3, 4), "vols": torch.zeros(2, 3, 4)})
