"""
Unit tests for the boosted-tree model's settings and construction.

Short, because the model is three declarations and a factory. That is the
honest amount of test for the honest amount of code, and the structural
guarantees -- that it registers, that it is under the tier 1 budget, that
``model.py`` stays free of wiring -- are asserted once for every model in
``test_model_layout.py`` rather than restated here.
"""

from __future__ import annotations

import importlib.util

import pytest
from pydantic import ValidationError

from src.rade_xl.models.xgb_tabular.spec import XgbTabularSpec

if importlib.util.find_spec("xgboost") is None:  # pragma: no cover
    pytest.skip("xgboost is not installed", allow_module_level=True)

# Imported after the skip, and from the engine package rather than the
# library, because importing the package is what forces Torch's OpenMP
# runtime to load first. See PHASE_6 §8.2: the reverse order deadlocks with
# no error at all, which is why this is not a plain top-level import.
from src.rade_xl.models.xgb_tabular.model import build


class TestTheModelDeclaresNoSettingsOfItsOwn:
    """
    The ownership question, pinned.

    Two schemas describe a run. ``XGBoostTrainingSpec`` already describes
    the booster completely, and this model has no architecture beyond it,
    so ``model.params`` is empty. The tests below are what stop that
    answer being quietly reversed.
    """

    def test_the_spec_is_empty(self) -> None:
        """
        Every booster setting lives in the training spec.

        Not a style preference. An earlier version re-declared
        ``max_depth``, ``min_child_weight`` and ``reg_lambda`` here with
        the same defaults, and the engine merges model params *last*: a
        user who set ``training.max_depth: 12`` trained at depth 6 and was
        told nothing. Both numbers were plausible and the run succeeded.

        If this fails, check the engine's merge order before adding the
        field -- a setting declared in two places has a winner, and it is
        rarely the one the user typed.
        """
        assert not XgbTabularSpec.model_fields

    @pytest.mark.parametrize(
        "field", ["max_depth", "min_child_weight", "reg_lambda", "n_estimators"]
    )
    def test_a_booster_setting_is_refused_here(self, field: str) -> None:
        """
        Putting one in ``model.params`` is an error, not a silent override.

        This is the behaviour that makes the division safe to rely on: a
        user who guesses the wrong side gets a validation error naming the
        field, rather than a run that ignores them.
        """
        with pytest.raises(ValidationError):
            XgbTabularSpec.model_validate({field: 3})


class TestBuildCarriesNoOverrides:
    """A booster does not exist until training ends, so this holds a slot."""

    def test_the_holder_comes_back_empty_and_unfitted(self) -> None:
        """
        Nothing is injected between the training spec and the booster.

        A non-empty ``params`` here would override the training spec for
        whatever key it contained, which is exactly the defect above.
        """
        holder = build(XgbTabularSpec())
        assert holder.params == {}
        assert not holder.is_fitted
