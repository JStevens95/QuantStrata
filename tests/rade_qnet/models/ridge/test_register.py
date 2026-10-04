"""
Unit tests for the ridge wiring.

Separate from ``test_model.py`` for the same reason the source files are
separate: a failure here means the model is plugged in wrongly, and a
failure there means the model is wrong. Keeping them apart means the test
that fails tells you which.
"""

from __future__ import annotations

from src.rade_qnet.core.runtime.components import MODELS, get_model
from src.rade_qnet.core.spec.run import parse_run_spec
from src.rade_qnet.models.ridge.register import RidgeModel
from src.rade_qnet.models.ridge.spec import RidgeSpec
from src.rade_qnet.sources.dataset.module import TabularDataModule


def specification(path: str) -> dict:
    """
    Return a minimal valid run specification naming this model.

    Parameters
    ----------
    path
        Any path; nothing here reads the file.

    Returns
    -------
    dict
        A specification ready for :func:`~rade_qnet.core.spec.run.parse_run_spec`.
    """
    return {
        "task": "supervised",
        "model": {"name": "ridge", "params": {"alpha": 0.5}},
        "source": {"kind": "tabular", "path": path},
        "training": {"engine": "sklearn"},
        "reports": {"enabled": []},
        "hardware": {"device": "cpu"},
    }


class TestTheDeclaration:
    """The four things ``register.py`` is responsible for saying."""

    def test_the_model_resolves_by_name(self) -> None:
        """
        ``"ridge"`` in a specification finds this class.

        The import at the top of this module is what registers it, which
        is the whole discovery mechanism.
        """
        assert get_model("ridge") is RidgeModel

    def test_the_declared_engine_is_recorded_as_metadata(self) -> None:
        """
        The registry knows which engine the model needs.

        That is what lets a pipeline reject a specification pairing this
        model with the Torch engine before building anything, rather than
        failing partway through construction with a less obvious message.
        """
        assert MODELS.entry("ridge").metadata["engine"] == "sklearn"

    def test_the_spec_class_is_attached(self) -> None:
        """The open-ended ``model.params`` block has a validator."""
        assert RidgeModel.spec is RidgeSpec


class TestTheTwoMethods:
    """What the framework calls, and what it is entitled to get back."""

    def test_the_data_module_is_the_frameworks_own(self) -> None:
        """
        A tier 1 model adds no data code.

        If this ever returns a custom module, the model has moved to tier
        3 and needs a ``data.py`` -- which is a layout change, not a
        one-line change here.
        """
        spec = parse_run_spec(specification("ignored.csv"))
        assert isinstance(RidgeModel().data_module(spec), TabularDataModule)

    def test_params_from_the_specification_reach_the_estimator(self) -> None:
        """
        End to end through the wiring: YAML value to configured estimator.

        The unit tests in ``test_model.py`` prove ``build`` honours its
        settings; this proves the settings in the specification are the
        ones it is handed. Both can pass individually while the pair is
        broken, which is why this exists separately.
        """
        spec = parse_run_spec(specification("ignored.csv"))
        estimator = RidgeModel().build_model(spec, signature=None)  # type: ignore[arg-type]
        assert estimator.alpha == 0.5
