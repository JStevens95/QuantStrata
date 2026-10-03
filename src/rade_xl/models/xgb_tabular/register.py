"""
How the boosted-tree model plugs into the framework.

Structurally identical to :mod:`rade_xl.models.ridge.register`, which is the
point of having a template: a different engine and a different spec, in the
same four declarations and the same two methods. A reader who has
understood one model package has understood every tier 1 and tier 2 package
in the repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.capability.simple import TabularModel
from ...core.runtime.components import model
from .data import REQUIRES, data_module
from .model import build
from .spec import XgbTabularSpec

if TYPE_CHECKING:
    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec
    from ...engines.xgboost import BoosterModel
    from ...sources.dataset.module import TabularDataModule

__all__ = ["XgbTabularModel"]


@model("xgb_tabular", engine="xgboost")
class XgbTabularModel(TabularModel):
    """
    Framework declaration for the boosted-tree model.

    Attributes
    ----------
    requires
        What this model consumes, declared in ``data.py`` and checked
        against the data build before the model is constructed.
    spec
        Validates the ``model.params`` block of a run specification.
    """

    requires = REQUIRES
    spec = XgbTabularSpec

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

    def build_model(
        self, spec: SupervisedRunSpec, signature: InputSignature
    ) -> BoosterModel:
        """
        Construct the unfitted booster holder.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from the data build. Unused: a tree
            takes its width from the matrix it is handed.

        Returns
        -------
        BoosterModel
            Unfitted.
        """
        del signature
        return build(XgbTabularSpec.model_validate(dict(spec.model.params)))
