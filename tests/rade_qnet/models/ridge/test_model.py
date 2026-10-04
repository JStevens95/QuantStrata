"""
Unit tests for the ridge architecture, with no framework in sight.

These are the tests the four-file layout exists to make possible. Nothing
below constructs a run specification, touches the registry or starts an
engine, because ``model.py`` imports none of those -- which is a property
``test_model_layout.py`` enforces rather than hopes for.

The practical consequence is speed and blame. These run in milliseconds,
and a failure here is unambiguously about the model; a failure in
``test_register.py`` is unambiguously about the wiring.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.models.ridge.model import build
from src.rade_qnet.models.ridge.spec import RidgeSpec


class TestTheSpecRefusesNonsense:
    """Validation is the cheapest place to fail, so it should fail here."""

    def test_a_zero_penalty_is_rejected(self) -> None:
        """
        ``alpha`` must be strictly positive.

        Zero is ordinary least squares, which on the near-collinear design
        a flattened sequence window produces is unstable in a way that
        reads as a model excellent in sample and useless out of it. The
        constraint turns a subtle modelling failure into a parse error.
        """
        with pytest.raises(ValidationError):
            RidgeSpec(alpha=0.0)

    def test_the_defaults_are_usable_on_their_own(self) -> None:
        """
        An empty ``model.params`` block produces a working model.

        This is what lets a specification name ``ridge`` and nothing else.
        """
        assert RidgeSpec().alpha > 0


class TestBuildIsAPureFunctionOfTheSettings:
    """No data, no signature, no state: that is what makes bundles reloadable."""

    def test_the_settings_reach_the_estimator(self) -> None:
        """
        What the spec says is what the estimator gets.

        Worth asserting because the mapping is written out by hand, and a
        transposed pair of keyword arguments would train a perfectly
        plausible, differently configured model.
        """
        estimator = build(RidgeSpec(alpha=0.25, fit_intercept=False))
        assert estimator.alpha == 0.25
        assert estimator.fit_intercept is False

    def test_the_estimator_comes_back_unfitted(self) -> None:
        """
        Construction and fitting are separate, and the engine owns fitting.

        A ``build`` that fitted anything would make the model's behaviour
        depend on when it was called, and would put data inside a function
        the bundle reload path calls with no data available.
        """
        assert not hasattr(build(RidgeSpec()), "coef_")
