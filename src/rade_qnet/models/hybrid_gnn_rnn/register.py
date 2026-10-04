"""
The framework declaration for the hybrid graph-temporal network.

``model.py`` answers *what does this compute?*. This file answers *how does
it plug in?* -- which spec validates its settings, which data module builds
its dataset, which engine trains it, and which pipelines it overrides.

Keeping the two apart is not tidiness for its own sake. The network is the
part a quant reads and argues with; the wiring is the part a platform
engineer reads and maintains. Interleaving them means neither reader can
skim their own half, and in practice the wiring wins -- a reader looking
for the mathematics finds registration boilerplate first and stops.

Importing this module is what makes ``hybrid_gnn_rnn`` resolvable by name
in a run specification. The package's ``__init__`` imports it for exactly
that reason, so a user who has imported the model package can name the
model in YAML without knowing this file exists.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from ...core.capability.simple import TabularModel
from ...core.runtime.components import model

# Imported for its import side effect: this is what puts the Torch engine
# and its supervised learner in the registry. The decorator below declares
# `engine="torch"`, and a declaration whose subject may or may not be
# registered depending on what else the process happened to import is the
# classic source of "no engine named 'torch'" from a correct specification.
# Importing this model already implies Torch is installed, so nothing is
# paid by a host that does not use it.
from ...engines import torch as _torch_engine  # noqa: F401
from .data import REQUIRES, HybridDataModule
from .model import HybridGnnRnn
from .pipelines.eval import HybridEvalPipeline
from .pipelines.train import HybridTrainPipeline
from .pipelines.tune import HybridTunePipeline
from .spec import HybridDataSpec, HybridModelSpec
from .state import HybridState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["HybridGnnRnnModel"]


@model("hybrid_gnn_rnn", engine="torch")
class HybridGnnRnnModel(TabularModel):
    """
    Framework declaration for the hybrid graph-temporal network.

    Attributes
    ----------
    spec
        Validates the ``model.params`` block of a run specification.
    data_spec
        Validates the ``source.params`` block. Separate from ``spec``
        because the data build and the network are configured
        independently: changing the recurrent width should not invalidate
        a cached dataset, and the two specs being distinct types is what
        makes that structurally true rather than merely intended.
    state_cls
        The fitted state this model's data build produces.
    pipelines
        Which lifecycles this model overrides: ``train`` to add reports,
        ``eval`` to break the error down per target, and ``tune`` to drop
        infeasible trials before they are spent. Inference is deliberately
        absent and inherits the framework's -- see
        :mod:`~rade_qnet.models.hybrid_gnn_rnn.pipelines`. Read-only so that
        a caller cannot reach into the class and rebind a stage for every
        run in the process.
    """

    requires = REQUIRES
    spec = HybridModelSpec
    data_spec = HybridDataSpec
    state_cls = HybridState
    pipelines: Mapping[str, type] = MappingProxyType(
        {
            "train": HybridTrainPipeline,
            "eval": HybridEvalPipeline,
            "tune": HybridTunePipeline,
        }
    )

    def data_module(self, spec: SupervisedRunSpec) -> HybridDataModule:
        """
        Return the data module that builds this model's dataset.

        Constructed per call rather than held on the definition. The module
        carries no state of its own -- every setting it uses is read from
        the source spec it is handed -- so a fresh one costs nothing, and a
        shared one would be an object two concurrent jobs could reach.

        Parameters
        ----------
        spec
            The validated run specification. Unused: the module reads its
            settings from ``spec.source`` when the framework calls it,
            which keeps this method a pure constructor.

        Returns
        -------
        HybridDataModule
            The data module.
        """
        del spec
        return HybridDataModule()

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> HybridGnnRnn:
        """
        Construct the untrained network.

        A pure function of the spec and the signature, with no data in
        sight. That is what makes a bundle reloadable six months later: the
        saved signature plus the saved weights are sufficient to rebuild
        the identical object, with no need to reproduce the dataset that
        originally shaped it.

        The network sizes every layer here rather than lazily on the first
        forward pass, so the object this returns is already fully
        parameterised and the engine's materialise step finds nothing left
        to do.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from the data build.

        Returns
        -------
        HybridGnnRnn
            An untrained, fully parameterised network.
        """
        return HybridGnnRnn(HybridModelSpec.model_validate(dict(spec.model.params)), signature)
