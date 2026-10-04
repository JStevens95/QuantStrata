"""
How the ridge regression plugs into the framework.

Every model package has a ``register.py`` and it always does the same four
things: name the model, name its engine, bind the spec that validates its
settings and the contract its data must meet, and say how to build its data
and itself. It contains no
mathematics. A platform engineer maintaining the framework reads this file;
a quant arguing with the model reads ``model.py``; neither has to skim the
other's half.

Importing this module is what makes ``"ridge"`` resolvable by name in a run
specification. The package's ``__init__`` imports it for exactly that
reason, so a user who has imported the model package can name the model in
YAML without knowing this file exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.capability.supervised import SupervisedModel
from ...core.runtime.components import model

# Imported for its registration side effect: this is what puts the
# scikit-learn engine in the registry. The decorator below declares
# `engine="sklearn"`, and a declaration whose subject may or may not be
# registered depending on what else the process happened to import is the
# classic source of "no engine named 'sklearn'" from a correct
# specification. Defect 12 in the architecture notes.
from ...engines import sklearn as _engine  # noqa: F401
from .data import REQUIRES, data_module
from .model import build
from .spec import RidgeSpec

if TYPE_CHECKING:
    from sklearn.linear_model import Ridge

    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec
    from ...sources.dataset.module import TabularDataModule

__all__ = ["RidgeModel"]


@model("ridge", engine="sklearn")
class RidgeModel(SupervisedModel):
    """
    Framework declaration for the ridge regression.

    Attributes
    ----------
    requires
        What this model consumes, declared in ``data.py`` and checked
        against the data build before the model is constructed.
    spec
        Validates the ``model.params`` block of a run specification.
    """

    requires = REQUIRES
    spec = RidgeSpec

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

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> Ridge:
        """
        Construct the unfitted estimator.

        A pure function of the spec, with no data in sight. That is what
        makes a bundle reloadable six months later: the saved signature
        plus the saved weights are sufficient to rebuild the identical
        object with no need to reproduce the dataset.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from the data build. Unused: a linear
            model takes its width from the matrix it is handed rather than
            being sized in advance.

        Returns
        -------
        sklearn.linear_model.Ridge
            Unfitted.
        """
        del signature
        return build(RidgeSpec.model_validate(dict(spec.model.params)))
