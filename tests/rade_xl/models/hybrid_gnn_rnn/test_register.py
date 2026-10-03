"""
Tests for the model's framework declaration.

These check the wiring rather than the mathematics: that the name in a
specification resolves to this model, that the engine and the reports the
declaration names are actually registered by the time anything asks for
them, and that the definition builds its parts from the spec alone.

Registration by accident is the failure worth naming. A declaration whose
subject happens to be registered -- because some other import pulled it in
-- passes every test run from a session that imported everything, and
fails the first real run that imports only what it needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rade_xl.core.runtime.components import (
    ENGINES,
    MODELS,
    REPORTS,
    get_model,
)
from src.rade_xl.core.spec.run import parse_run_spec
from src.rade_xl.models.hybrid_gnn_rnn.data import HybridDataModule
from src.rade_xl.models.hybrid_gnn_rnn.model import HybridGnnRnn
from src.rade_xl.models.hybrid_gnn_rnn.pipelines.train import (
    HYBRID_REPORTS,
    HybridTrainPipeline,
)
from src.rade_xl.models.hybrid_gnn_rnn.register import HybridGnnRnnModel
from src.rade_xl.models.hybrid_gnn_rnn.spec import HybridDataSpec, HybridModelSpec
from src.rade_xl.models.hybrid_gnn_rnn.state import HybridState
from src.rade_xl.orchestration.pipelines.train import TrainPipeline

from .test_model import signature

FIXTURE = Path("tests/fixtures/rade_xl/golden/hybrid_gnn_rnn/input")


def run_spec(**report_overrides: object):
    """
    Build a minimal validated run specification naming the model.

    Parameters
    ----------
    **report_overrides
        Fields to set on the reports block.

    Returns
    -------
    SupervisedRunSpec
        The validated specification.
    """
    return parse_run_spec(
        {
            "task": "supervised",
            "model": {"name": "hybrid_gnn_rnn", "params": {"units": 8}},
            "source": {"kind": "model", "params": {"directory": str(FIXTURE)}},
            "training": {"engine": "torch"},
            "reports": dict(report_overrides) or {},
        }
    )


class TestRegistration:
    """What importing the model package makes resolvable."""

    def test_the_model_resolves_by_the_name_a_spec_uses(self) -> None:
        """
        A specification names a string; this is what turns it into a class.

        Resolved through the registry rather than imported, because that
        is the path a real run takes.
        """
        assert "hybrid_gnn_rnn" in MODELS
        assert get_model("hybrid_gnn_rnn") is HybridGnnRnnModel

    def test_the_declared_engine_is_registered(self) -> None:
        """
        The decorator declares ``torch``; importing the model provides it.

        A declaration whose subject may or may not be registered,
        depending on what else the process happened to import, is the
        classic source of "no engine named 'torch'" from a configuration
        that is perfectly correct.
        """
        assert HybridGnnRnnModel.component_engine == "torch"
        assert "torch" in ENGINES

    def test_the_model_report_is_registered(self) -> None:
        """
        For the same reason, and by the same mechanism.

        The train pipeline adds ``hybrid_graph`` to every run, so a name
        that resolved only when something else had imported the report
        module would fail the run at its first stage.
        """
        assert "hybrid_graph" in REPORTS
        assert all(name in REPORTS for name in HYBRID_REPORTS)

    def test_the_declared_types_are_the_real_ones(self) -> None:
        """
        The specs, state and pipelines a reader would look for.

        These are the only place the wiring is written down, so a wrong
        one here is a wrong one everywhere.
        """
        assert HybridGnnRnnModel.spec is HybridModelSpec
        assert HybridGnnRnnModel.data_spec is HybridDataSpec
        assert HybridGnnRnnModel.state_cls is HybridState
        assert HybridGnnRnnModel.pipelines["train"] is HybridTrainPipeline

    def test_the_pipeline_mapping_cannot_be_mutated(self) -> None:
        """
        It is class-level, so a write would affect every run in the process.

        Not a hypothetical: a test or a notebook rebinding a stage for
        one run would silently rebind it for all of them.
        """
        with pytest.raises(TypeError):
            HybridGnnRnnModel.pipelines["train"] = TrainPipeline  # type: ignore[index]


class TestDefinition:
    """What the definition builds."""

    def test_the_data_module_is_fresh_each_time(self) -> None:
        """
        Not held on the definition, so two jobs cannot reach the same object.

        The module carries no state -- every setting comes from the spec
        it is handed -- so a fresh one costs nothing and a shared one is
        a thing that could accumulate something.
        """
        definition = HybridGnnRnnModel()
        first = definition.data_module(run_spec())
        second = definition.data_module(run_spec())
        assert isinstance(first, HybridDataModule)
        assert first is not second

    def test_the_model_is_built_from_the_spec_and_signature_alone(self) -> None:
        """
        No data reaches the constructor.

        This is what makes a six-month-old bundle reloadable: the saved
        signature plus the saved weights are sufficient to rebuild the
        identical object, with no need to reproduce the dataset that
        originally shaped it.
        """
        spec = run_spec()
        built = HybridGnnRnnModel().build_model(spec, signature())
        assert isinstance(built, HybridGnnRnn)
        assert built.spec.units == 8
