"""
How the recurrent baseline plugs into the framework.

Same shape as the other two tier 1 packages, with a third engine. Note that
``build_model`` here forwards the signature where ridge discards it -- that
is the only material difference between a model that must be sized in
advance and one that is not, and it is a two-word difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.capability.supervised import SupervisedModel
from ...core.runtime.components import model

# Imported for its registration side effect: this is what puts the Torch
# engine in the registry, so the `engine="torch"` declaration below resolves
# rather than depending on what else the process happened to import.
# Defect 12 in the architecture notes.
from ...engines import torch as _engine  # noqa: F401
from .data import REQUIRES, data_module
from .model import build
from .spec import LstmTabularSpec

if TYPE_CHECKING:
    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec
    from ...sources.dataset.module import TabularDataModule
    from .model import LstmTabular

__all__ = ["LstmTabularModel"]


@model("lstm_tabular", engine="torch")
class LstmTabularModel(SupervisedModel):
    """
    Framework declaration for the flagship's temporal stream, alone.

    Attributes
    ----------
    requires
        What this model consumes, declared in ``data.py`` and checked
        against the data build before the model is constructed.
    spec
        Validates the ``model.params`` block.
    """

    requires = REQUIRES
    spec = LstmTabularSpec

    def data_module(self, spec: SupervisedRunSpec) -> TabularDataModule:
        """
        Return the data module that builds this model's dataset.

        Delegated to ``data.py``, which is where every question about this
        model's data is answered -- including the requirement bound above.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        TabularDataModule
            Whatever ``data.py`` builds.
        """
        return data_module(spec)

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> LstmTabular:
        """
        Return a fully parameterised network.

        Parameters
        ----------
        spec
            The run specification, for the model's parameters.
        signature
            The declared interface, which supplies the input width.

        Returns
        -------
        LstmTabular
            Untrained.
        """
        settings = LstmTabularSpec.model_validate(dict(spec.model.params))
        return build(settings, signature)
