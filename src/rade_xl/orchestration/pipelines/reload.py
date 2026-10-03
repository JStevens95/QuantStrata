"""
Opening a saved bundle and turning it back into something runnable.

A bundle on disk is a directory of files. Evaluation, inference and any later
comparison all need the same thing from it: the specification it was trained
from, the model definition that interprets its weights, the fitted state its
transforms live in, and the lineage that records how its data was split.

Shared as a module rather than as a pipeline base class. Two pipelines
calling the same function stay in step; two pipelines inheriting a common
base diverge the first time one of them needs a stage the other does not --
and evaluation and inference differ in exactly that way, because one has
targets and the other does not.

Why a bundle is not self-describing
-----------------------------------
The manifest records a model *name*, not a Python import path. Phase 1 chose
that deliberately: a recorded class path means renaming a class breaks every
bundle written before the rename. The cost is that loading a bundle requires
the model's package to be importable, so that the name resolves through the
registry.

That cost is not worth engineering away, because it is not really a cost.
Weights are meaningless without the architecture that interprets them, so any
process that can use a bundle has the model code anyway. What *is* worth
getting right is the error. A registry miss from a bundle load should say
that the bundle names a model this process has not imported -- which is the
same fault as Phase 4's defect 12, reached from the other direction -- rather
than reporting a bare lookup failure that leaves the reader guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.capability.definition import PredictorDefinition
from ...core.capability.simple import RebuildableDataModule
from ...core.runtime.components import MODELS, ComponentError, get_model
from ...core.runtime.errors import BundleError
from ...core.runtime.logging import get_logger
from ...core.spec.run import SupervisedRunSpec
from ...storage.bundle import (
    load_fitted_state,
    load_lineage,
    load_signature,
    load_spec,
    open_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ...core.contract.bundle import SavedBundle
    from ...core.contract.data import DataLineage
    from ...core.contract.signature import InputSignature
    from ...core.contract.state import FittedState

__all__ = ["LoadedBundle", "load_bundle"]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    """
    Everything a pipeline needs to put a saved model back to work.

    Parameters
    ----------
    saved
        The located bundle: its directory and verified manifest.
    spec
        The run specification the model was trained from.
    definition
        The model definition, resolved from the manifest's model name.
    state
        The fitted state, to be applied and never re-fitted.
    lineage
        The data lineage, supplying the split that must not be re-derived.
    signature
        The input signature the model was built against, used to check that
        rebuilt data still presents the interface the weights expect.
    """

    saved: SavedBundle
    spec: SupervisedRunSpec
    definition: PredictorDefinition
    state: FittedState
    lineage: DataLineage
    signature: InputSignature

    @property
    def model_name(self) -> str:
        """
        Return the name the model is registered under.

        Returns
        -------
        str
            From the manifest, so it is what was recorded rather than what
            the resolved class happens to be called now.
        """
        return self.saved.manifest.model_name

    def describe(self) -> str:
        """
        Return a one-line description, for a log and a report header.

        Returns
        -------
        str
            Model name, bundle version and the split sizes it was trained
            against.
        """
        sizes = {name: len(values) for name, values in self.lineage.split_indices.items()}
        return (
            f"{self.model_name} v{self.saved.manifest.version} "
            f"(spec {self.saved.manifest.spec_digest[:8]}, splits {sizes})"
        )


def load_bundle(directory: Path, *, verify: bool = True) -> LoadedBundle:
    """
    Open a saved bundle and resolve everything needed to run it again.

    Parameters
    ----------
    directory
        The bundle directory.
    verify
        Whether to re-hash every file against the manifest. Defaults to
        true, because a bundle about to produce numbers someone will act on
        is exactly the case where the check is worth its cost.

    Returns
    -------
    LoadedBundle
        The specification, definition, fitted state, lineage and signature.

    Raises
    ------
    BundleError
        If the bundle is missing or inconsistent, if its model is not
        registered in this process, or if its definition cannot supply the
        fitted-state type needed to read ``fitted_state/``.
    """
    saved = open_bundle(directory, verify=verify)
    spec = load_spec(saved)

    if not isinstance(spec, SupervisedRunSpec):
        raise BundleError(
            f"the bundle at {directory} holds a {type(spec).__name__}; only "
            f"supervised runs can be evaluated or served through this path"
        )

    definition = _definition_for(saved.manifest.model_name, directory)
    state = load_fitted_state(saved, _state_type_for(definition, spec, directory))
    lineage = load_lineage(saved)
    signature = load_signature(saved)

    loaded = LoadedBundle(
        saved=saved,
        spec=spec,
        definition=definition,
        state=state,
        lineage=lineage,
        signature=signature,
    )
    _LOGGER.info("loaded %s", loaded.describe())
    return loaded


def _definition_for(model_name: str, directory: Path) -> PredictorDefinition:
    """
    Resolve a manifest's model name into an instantiated definition.

    Parameters
    ----------
    model_name
        The name recorded in the manifest.
    directory
        The bundle directory, named in errors so the reader knows which
        bundle could not be opened.

    Returns
    -------
    PredictorDefinition
        A fresh instance of the registered definition.

    Raises
    ------
    BundleError
        If the name is not registered here, or is registered to something
        that is not a predictor.
    """
    try:
        model_type = get_model(model_name)
    except ComponentError as error:
        raise BundleError(
            f"the bundle at {directory} was trained by a model named "
            f"{model_name!r}, which is not registered in this process. A "
            f"model registers when its package is imported, so import the "
            f"package that defines it before loading the bundle. "
            f"Registered here: {sorted(MODELS.names()) or 'nothing'}"
        ) from error

    definition = model_type()
    if not isinstance(definition, PredictorDefinition):
        raise BundleError(
            f"the bundle at {directory} names the model {model_name!r}, which "
            f"is registered as a {type(definition).__name__}; only a predictor "
            f"can be evaluated or served through this path"
        )
    return definition


def _state_type_for(
    definition: PredictorDefinition, spec: SupervisedRunSpec, directory: Path
) -> type[FittedState]:
    """
    Ask a definition for the concrete type its fitted state loads into.

    The type is not recorded on disk, by the same decision that keeps class
    paths out of the manifest. The data module declares it, so the data
    module is asked -- which is why loading a bundle needs the model package
    and not merely the framework.

    Parameters
    ----------
    definition
        The resolved model definition.
    spec
        The bundle's run specification, needed because a definition may
        choose its data module from the spec.
    directory
        The bundle directory, named in errors.

    Returns
    -------
    type[FittedState]
        The class to read ``fitted_state/`` into.

    Raises
    ------
    BundleError
        If the definition cannot supply a data module that declares one.
    """
    data_module = getattr(definition, "data_module", None)
    if data_module is None:
        raise BundleError(
            f"the model {type(definition).__name__} has no data_module(), so "
            f"nothing can say what type the fitted state in {directory} loads "
            f"into. A model that builds its data by a bespoke route must "
            f"expose the module that knows how to rebuild it"
        )

    module = data_module(spec)
    if not isinstance(module, RebuildableDataModule):
        raise BundleError(
            f"{type(definition).__name__}.data_module() returned a "
            f"{type(module).__name__}, which has no rebuild(); the bundle at "
            f"{directory} can be opened but its model cannot be re-fed the "
            f"way it was trained, so any metric from it would be meaningless"
        )

    state_type = getattr(module, "state_type", None)
    if not isinstance(state_type, type):
        raise BundleError(
            f"{type(module).__name__} declares no state_type, so the fitted "
            f"state in {directory} cannot be read back"
        )
    return state_type
