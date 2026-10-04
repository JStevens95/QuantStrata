"""Tests for the hybrid model's train-pipeline override."""

from __future__ import annotations

from src.rade_qnet.core.runtime.components import get_report
from src.rade_qnet.models.hybrid_gnn_rnn.pipelines.train import (
    HYBRID_REPORTS,
    HybridTrainPipeline,
)
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline

from ..test_register import run_spec


class TestTrainOverride:
    """The tier-2 pipeline override."""

    def test_the_stage_sequence_is_untouched(self) -> None:
        """
        The override extends a hook; it does not replace a step.

        If this ever has to change, that is a finding about the
        framework's step granularity rather than about this model, and
        the Phase 3 notes say to raise it rather than absorb it.
        """
        assert HybridTrainPipeline.stages == TrainPipeline.stages
        overridden = set(vars(HybridTrainPipeline)) & set(TrainPipeline.stages)
        assert not overridden

    def test_the_model_report_is_added_to_whatever_the_spec_asked_for(self) -> None:
        """
        A user narrowing the report set still gets the graph diagnostics.

        The graph is the model's largest assumption and the one least
        visible in any metric, so switching it off to save a few seconds
        should not be something a user can do by accident.
        """
        pipeline = HybridTrainPipeline.__new__(HybridTrainPipeline)
        pipeline.spec = run_spec(enabled=("curves",))
        assert pipeline.report_names() == ("curves", "hybrid_graph")

    def test_naming_it_explicitly_does_not_run_it_twice(self) -> None:
        """
        A duplicate would render twice and overwrite its own output.

        Dropped rather than rejected: a user who asks for something
        already guaranteed has made a harmless redundancy, not an error
        worth failing a run over.
        """
        pipeline = HybridTrainPipeline.__new__(HybridTrainPipeline)
        pipeline.spec = run_spec(enabled=("hybrid_graph", "summary"))
        assert pipeline.report_names() == ("hybrid_graph", "summary")

    def test_the_spec_ordering_is_preserved(self) -> None:
        """
        Appended, not prepended.

        Order is occasionally meaningful: a summary report that lists the
        figures produced has to run after them, so reordering the user's
        selection would break it.
        """
        pipeline = HybridTrainPipeline.__new__(HybridTrainPipeline)
        pipeline.spec = run_spec(enabled=("curves", "summary"))
        assert pipeline.report_names()[:2] == ("curves", "summary")

    def test_the_base_hook_returns_the_spec_unchanged(self) -> None:
        """
        A model that does not override it sees exactly the old behaviour.

        The hook was added to the base for this model's sake, so the
        first thing to check is that it changed nothing for anyone else.
        """
        pipeline = TrainPipeline.__new__(TrainPipeline)
        pipeline.spec = run_spec(enabled=("summary", "curves"))
        assert pipeline.report_names() == ("summary", "curves")

    def test_every_name_the_override_adds_resolves(self) -> None:
        """
        Otherwise the run fails at its first stage, after validating cleanly.

        The pipeline's ``resolve`` stage instantiates each report for
        precisely this reason, and this is the same check one import
        earlier.
        """
        for name in HYBRID_REPORTS:
            assert get_report(name)
