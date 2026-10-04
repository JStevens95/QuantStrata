"""Tests for the hybrid model's tune-pipeline override."""

from __future__ import annotations

import pytest

from src.rade_qnet.core.lifecycle.errors import ContractError, SpecError
from src.rade_qnet.core.spec.tune import parse_tune_spec
from src.rade_qnet.models.hybrid_gnn_rnn.pipelines.tune import (
    HybridTunePipeline,
    is_buildable,
)
from src.rade_qnet.orchestration.pipelines.tune import TunePipeline
from src.rade_qnet.orchestration.stages.search import expand

from ..test_register import FIXTURE, run_spec


def variant(**params: object):
    """
    Build a run specification with the model parameters overridden.

    Parameters
    ----------
    **params
        Values to set on the model reference's parameter bag.

    Returns
    -------
    SupervisedRunSpec
        The validated specification.
    """
    spec = run_spec()
    reference = spec.model.model_copy(update={"params": {**spec.model.params, **params}})
    return spec.model_copy(update={"model": reference})


def tune_spec(space: dict[str, object], *, trials: int = 64):
    """
    Build a validated search specification over the hybrid model.

    Parameters
    ----------
    space
        Dotted paths to the values to try.
    trials
        The budget, defaulted high enough that a grid is exhausted.

    Returns
    -------
    TuneSpec
        The validated specification.
    """
    return parse_tune_spec(
        {
            "model": "hybrid_gnn_rnn",
            "base": {
                "task": "supervised",
                "model": {"name": "hybrid_gnn_rnn", "params": {"units": 8}},
                "source": {"kind": "model", "params": {"directory": str(FIXTURE)}},
                "training": {"engine": "torch"},
                "reports": {"enabled": []},
            },
            "space": space,
            "trials": trials,
            "sampler": "grid",
        }
    )


def proposals_for(space: dict[str, object], **kwargs):
    """
    Return what the override proposes for a search space.

    Parameters
    ----------
    space
        Dotted paths to the values to try.
    **kwargs
        Passed to :func:`tune_spec`.

    Returns
    -------
    list of dict
        One flat proposal per surviving trial.
    """
    pipeline = HybridTunePipeline.__new__(HybridTunePipeline)
    pipeline.spec = tune_spec(space, **kwargs)
    return pipeline.propose()


class TestTheFeasibilityTest:
    """`is_buildable`, which previews what the layers will say."""

    def test_a_width_divisible_by_its_heads_is_buildable(self) -> None:
        """The ordinary case, and the one a default configuration hits."""
        assert is_buildable(run_spec())

    def test_the_fusion_block_is_checked(self) -> None:
        """
        Because the fusion reshape is the first one a forward pass reaches.

        Eight features over three heads leaves two features that the head
        reshape would drop, which the layer refuses -- correctly, and too
        late to save the trial.
        """
        assert not is_buildable(variant(fusion_heads=3))

    def test_the_attention_block_is_checked_too(self) -> None:
        """
        A proposal can be feasible for one block and not the other.

        Checking only fusion would admit this configuration and spend the
        trial on it, which is the entire failure this file prevents.
        """
        assert not is_buildable(variant(attention_heads=5))

    def test_per_block_widths_are_read_rather_than_the_shared_one(self) -> None:
        """
        ``units`` is only a default; a block may override it.

        Testing the shared knob instead of the resolved width would call a
        configuration feasible whenever its default happened to divide,
        regardless of the width the block will actually build with.
        """
        assert not is_buildable(variant(fusion_units=9, fusion_heads=2))

    def test_another_model_s_spec_is_not_judged(self) -> None:
        """
        Answering True is right: this function knows of no reason it fails.

        Reached when the pipeline is attached to the wrong definition,
        where refusing every trial would be a confusing way to report a
        wiring mistake.
        """
        spec = run_spec()
        foreign = spec.model.model_copy(update={"name": "some_other_model"})
        assert is_buildable(spec.model_copy(update={"model": foreign}))


class TestTheOverride:
    """What the pipeline does with the framework's proposals."""

    def test_the_stage_sequence_is_untouched(self) -> None:
        """
        ``propose`` is extended, not added to or removed from.

        The override replaces a stage's body, which is a tier-3 override
        and the heaviest kind this model uses. What it must not do is
        change the sequence: a model whose lifecycle differs from the
        framework's is a finding about the framework's step granularity
        rather than about this model.
        """
        assert HybridTunePipeline.stages == TunePipeline.stages

    def test_a_wholly_feasible_space_is_passed_through_unchanged(self) -> None:
        """Nothing is filtered when nothing needs to be."""
        space = {"model.params.units": [8, 16, 32]}
        assert len(proposals_for(space)) == 3

    def test_the_infeasible_cells_of_a_grid_are_dropped(self) -> None:
        """
        Three widths crossed with three head counts is nine cells.

        Written as the explicit surviving set rather than as "fewer than
        nine" so that a change to the divisibility rule fails here, rather
        than quietly altering how much of a budget a search spends.
        """
        space = {
            "model.params.units": [8, 16, 24],
            "model.params.fusion_heads": [1, 3, 4],
        }
        proposals = proposals_for(space)

        feasible = {(p["model.params.units"], p["model.params.fusion_heads"]) for p in proposals}
        assert feasible == {(8, 1), (16, 1), (24, 1), (24, 3), (8, 4), (16, 4), (24, 4)}

    def test_every_surviving_proposal_actually_builds(self) -> None:
        """
        The property the filter exists to guarantee, asserted directly.

        Checked against the feasibility test rather than against a
        recomputed expectation, so this still holds if the rule changes.
        """
        space = {
            "model.params.units": [8, 12, 16, 18],
            "model.params.attention_heads": [1, 2, 3],
        }
        pipeline = HybridTunePipeline.__new__(HybridTunePipeline)
        pipeline.spec = tune_spec(space)

        for index, proposal in enumerate(pipeline.propose()):
            assert is_buildable(pipeline.spec.run_spec_for(expand(proposal), trial=index))

    def test_a_path_the_spec_does_not_have_is_the_framework_s_error(self) -> None:
        """
        Validation runs before filtering, so a typo stays a user error.

        Filtering first would turn a mistyped path into a silently
        smaller search -- defect 7 wearing a different hat.
        """
        with pytest.raises(SpecError, match=r"valid run specification"):
            proposals_for({"no_such_block.no_such_field": [1, 2]})

    def test_a_model_parameter_the_schema_rejects_is_caught_here(self) -> None:
        """
        Which the framework's own check cannot do.

        A run spec carries the model as a name and an untyped bag of
        values; the bag meets this model's schema only at ``resolve``, by
        which point the trial is spent. Parsing the params to test
        feasibility means a mistyped knob is found before the search
        starts rather than sixty-four tracebacks into it.
        """
        with pytest.raises(SpecError, match=r"(?s)do not parse.*no_such_knob"):
            proposals_for({"model.params.no_such_knob": [1, 2]})

    def test_a_space_that_can_never_build_is_refused(self) -> None:
        """
        Because a search with no trials has no winner and no error.

        Unlike a partially-infeasible space, this one is a genuine
        contradiction between what was asked for and what exists, and
        silently returning nothing would surface as a confusing failure
        several stages later.
        """
        space = {
            "model.params.units": [8, 16],
            "model.params.fusion_heads": [3, 5],
        }
        with pytest.raises(ContractError, match=r"none of the .* can be built"):
            proposals_for(space)

    def test_the_refusal_names_the_knobs_involved(self) -> None:
        """So the message is actionable without reading this file."""
        space = {"model.params.units": [8], "model.params.fusion_heads": [3]}
        with pytest.raises(ContractError, match=r"fusion_heads.*attention_heads"):
            proposals_for(space)

    def test_the_base_pipeline_filters_nothing(self) -> None:
        """
        A model that does not override this sees the old behaviour.

        The whole override is additive, so the first thing to check is
        that it changed nothing for anybody else.
        """
        pipeline = TunePipeline.__new__(TunePipeline)
        pipeline.spec = tune_spec({"model.params.units": [8, 16], "model.params.fusion_heads": [3]})
        assert len(pipeline.propose()) == 2
