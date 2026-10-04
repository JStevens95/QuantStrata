"""Tests for the model pipeline-override resolver."""

from __future__ import annotations

import pytest

from src.rade_qnet.core.runtime.errors import ComponentError
from src.rade_qnet.models.hybrid_gnn_rnn.pipelines.eval import HybridEvalPipeline
from src.rade_qnet.models.hybrid_gnn_rnn.pipelines.train import HybridTrainPipeline
from src.rade_qnet.models.hybrid_gnn_rnn.pipelines.tune import HybridTunePipeline
from src.rade_qnet.models.hybrid_gnn_rnn.register import HybridGnnRnnModel
from src.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.orchestration.pipelines.tune import TunePipeline
from src.rade_qnet.orchestration.stages.resolve import LIFECYCLES, pipeline_for


class Base:
    """Stands in for a framework pipeline."""


class Override(Base):
    """Stands in for a model's override of it."""


class Unrelated:
    """A class that extends nothing."""


def definition(**pipelines: type):
    """
    Build a stand-in model definition declaring the given overrides.

    Parameters
    ----------
    **pipelines
        Lifecycle name to pipeline class.

    Returns
    -------
    object
        The definition.
    """
    return type("Definition", (), {"pipelines": dict(pipelines)})()


class TestResolvingAnOverride:
    """The lookup itself."""

    def test_a_declared_override_is_returned(self) -> None:
        """
        The whole point, and the thing that was missing.

        Before this resolver existed the declaration had no reader, so a
        model's override ran only when a caller imported and instantiated
        it by hand. Every documented entry point got the base pipeline
        and no error -- the override simply did not happen.
        """
        assert pipeline_for(definition(train=Override), "train", Base) is Override

    def test_a_model_with_no_overrides_gets_the_framework_s_pipeline(self) -> None:
        """Which is the ordinary case and must cost nothing."""
        assert pipeline_for(definition(), "train", Base) is Base

    def test_a_definition_without_the_attribute_is_fine(self) -> None:
        """
        Declaring overrides is optional, so its absence is not an error.

        A simple model has no reason to carry an empty mapping.
        """

        class Bare:
            """Declares nothing."""

        assert pipeline_for(Bare(), "train", Base) is Base

    def test_no_definition_at_all_is_fine(self) -> None:
        """
        Reached when the caller could not resolve one.

        An unregistered model behind a bundle, for instance. Falling back is right: the pipeline that follows reports the
        missing model far better than a helper can, and pre-empting it
        here would replace a good diagnostic with a worse one.
        """
        assert pipeline_for(None, "train", Base) is Base

    def test_an_override_for_another_lifecycle_is_not_used(self) -> None:
        """A model may customise training without customising evaluation."""
        assert pipeline_for(definition(train=Override), "eval", Base) is Base


class TestWhatIsRefused:
    """The two mistakes that would otherwise be silent."""

    def test_an_override_that_is_not_a_subclass_is_refused(self) -> None:
        """
        Because the caller was promised the base pipeline's return type.

        A model that returns something else breaks every caller that
        never asked for a custom pipeline, which makes it a framework
        failure rather than a model's own business.
        """
        with pytest.raises(ComponentError, match=r"not a subclass of Base"):
            pipeline_for(definition(train=Unrelated), "train", Base)

    def test_an_unknown_lifecycle_key_is_refused(self) -> None:
        """
        Because the failure is otherwise invisible.

        A model declaring ``evaluate`` where the framework reads ``eval``
        gets no error and no override: it appears to work, using the
        framework's pipeline, which is the shape of defect this resolver
        was written to end rather than to reproduce.
        """
        with pytest.raises(ComponentError, match=r"which no lifecycle reads"):
            pipeline_for(definition(evaluate=Override), "train", Base)

    def test_asking_for_an_unknown_lifecycle_is_refused(self) -> None:
        """A caller's typo is as worth catching as a model's."""
        with pytest.raises(ComponentError, match=r"is not a lifecycle"):
            pipeline_for(definition(), "evaluate", Base)

    def test_the_known_lifecycles_are_the_four_pipelines(self) -> None:
        """Pinned: a fifth means every caller of this function must learn it."""
        assert {"train", "eval", "infer", "tune"} == LIFECYCLES


class TestTheFlagshipIsWiredUp:
    """The regression this whole module exists to prevent."""

    def test_every_override_the_flagship_declares_resolves(self) -> None:
        """
        Asserted against the real model, not a stand-in.

        A stand-in would have kept passing throughout the period the
        flagship's overrides were unreachable.
        """
        model = HybridGnnRnnModel()
        assert pipeline_for(model, "train", TrainPipeline) is HybridTrainPipeline
        assert pipeline_for(model, "eval", EvaluatePipeline) is HybridEvalPipeline
        assert pipeline_for(model, "tune", TunePipeline) is HybridTunePipeline
