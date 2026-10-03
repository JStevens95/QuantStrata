"""
Writing and reading a versioned bundle on disk.

Two disciplines govern this module, and both exist because of specific ways
model stores go wrong in production.

**Writes are atomic.** A bundle is written into a temporary directory beside
its destination and then renamed into place in one filesystem operation. A
crash, a cancellation or a full disk therefore leaves either no bundle or a
complete one -- never a directory containing weights but no manifest, which
looks loadable and is not. The implementation this replaces wrote files
in place, so an interrupted run left behind a bundle that failed at inference
time rather than at write time.

**Weights are serialised by the engine, not here.** This module writes bytes
it is handed. It never imports a training library, which is what the layering
test enforces and what lets the same bundle format hold a network's tensors
and a boosted-tree dump.

A note on pickled modules
-------------------------
The format deliberately stores *parameters*, not objects. Pickling a model
object embeds the class's import path and its source semantics into the file,
so refactoring the class breaks every bundle ever written -- and loading one
with ``weights_only=False`` executes whatever the file says. Writing
parameters plus a signature means a bundle can be loaded after the model's
code has been reorganised, which is the normal course of events over the life
of a model.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel

from ..core.contract.bundle import (
    FITTED_STATE_DIRNAME,
    LINEAGE_FILENAME,
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    SIGNATURE_FILENAME,
    SPEC_FILENAME,
    WEIGHTS_FILENAME,
    Manifest,
    ModelBundle,
    SavedBundle,
)
from ..core.contract.data import DataLineage
from ..core.contract.result import TrainingResult
from ..core.contract.signature import InputSignature
from ..core.runtime.errors import BundleError, SpecError
from ..core.runtime.logging import get_logger
from ..core.spec.run import RunSpec, parse_run_spec
from .manifest import build_manifest, verify_manifest

if TYPE_CHECKING:
    from ..core.contract.state import FittedState

__all__ = [
    "bundle_directory",
    "load_fitted_state",
    "load_lineage",
    "load_manifest",
    "load_result",
    "load_signature",
    "load_spec",
    "open_bundle",
    "write_bundle",
]

_LOGGER = get_logger(__name__)

#: Prefix for in-progress bundle directories. Recognisable so a cleanup task
#: can distinguish abandoned partial writes from real bundles.
TEMPORARY_PREFIX = ".writing-"


def bundle_directory(root: Path, *, model_name: str, version: int, job_id: str | None) -> Path:
    """
    Return the canonical directory for one bundle.

    The layout is ``<root>/<model>/<job>/v<N>``, with the job level omitted for
    a single run. Nesting by job rather than flattening to ``model-job-vN``
    keeps a job set's output navigable with ordinary tools when it has three
    hundred members.

    Parameters
    ----------
    root
        The model store root.
    model_name
        Registered model name.
    version
        Version number.
    job_id
        Job identifier, or ``None``.

    Returns
    -------
    Path
        The bundle directory, which may not exist yet.
    """
    base = root / model_name
    if job_id is not None:
        base = base / job_id
    return base / f"v{version}"


def write_bundle(
    bundle: ModelBundle,
    *,
    root: Path,
    version: int,
    framework_version: str,
    engine: str,
    model_name: str,
    write_weights: Callable[[Path], None],
    job_id: str | None = None,
    tags: tuple[str, ...] = (),
) -> SavedBundle:
    """
    Write a bundle atomically and return its location.

    The sequence matters. Everything is written into a temporary directory;
    the manifest is built last, over the files actually present; and only then
    is the directory renamed into place. Building the manifest before the last
    file was written would produce a manifest that does not describe the
    bundle, and renaming before the manifest exists would briefly expose a
    bundle that cannot be verified.

    Parameters
    ----------
    bundle
        The finished run, in memory.
    root
        The model store root.
    version
        Version number, obtained from the catalog so it is unique.
    framework_version
        Version of ``rade_xl`` writing this bundle.
    engine
        Registered engine name.
    model_name
        Registered model name.
    write_weights
        Callback that writes the model's parameters to the path it is given.
        Supplied by the engine, because only the engine knows how to
        serialise its own model -- and because ``storage`` must not import a
        training library.
    job_id
        Job identifier, or ``None`` for a single run.
    tags
        Free-form labels recorded in the manifest.

    Returns
    -------
    SavedBundle
        The written bundle's directory and manifest.

    Raises
    ------
    BundleError
        If the destination already exists. A version is never overwritten:
        a bundle that has been recorded in the catalog, evaluated, and perhaps
        promoted must not change underneath those decisions.
    """
    destination = bundle_directory(root, model_name=model_name, version=version, job_id=job_id)
    if destination.exists():
        raise BundleError(
            f"refusing to overwrite the existing bundle at {destination}; "
            f"request a new version from the catalog instead"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Beside the destination, not in the system temporary directory, so the
    # final rename stays on one filesystem and therefore stays atomic.
    staging = destination.parent / f"{TEMPORARY_PREFIX}{destination.name}-{uuid4().hex[:8]}"
    staging.mkdir(parents=True)

    try:
        _write_payload(bundle, staging, write_weights=write_weights)
        manifest = build_manifest(
            staging,
            model_name=model_name,
            engine=engine,
            version=version,
            framework_version=framework_version,
            spec_digest=bundle.lineage.spec_digest,
            job_id=job_id,
            metrics=dict(bundle.result.headline()),
            tags=tags,
        )
        (staging / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        # The one operation that makes the bundle visible. Everything above
        # this line is discardable; everything below it is committed.
        staging.rename(destination)
    except BaseException:
        # Covers KeyboardInterrupt and SystemExit too: an interrupted write
        # should not leave staging directories accumulating in the store.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _LOGGER.info("wrote bundle %s to %s", manifest.identifier, destination)
    return SavedBundle(directory=destination, manifest=manifest)


def _write_payload(
    bundle: ModelBundle,
    directory: Path,
    *,
    write_weights: Callable[[Path], None],
) -> None:
    """
    Write every part of a bundle except the manifest.

    Separated from :func:`write_bundle` so the atomic-rename logic reads as
    one short sequence, and so the set of files a bundle contains is stated in
    one place.

    Parameters
    ----------
    bundle
        The finished run.
    directory
        The staging directory.
    write_weights
        The engine's parameter-serialisation callback.
    """
    (directory / SPEC_FILENAME).write_text(bundle.spec.model_dump_json(indent=2), encoding="utf-8")
    (directory / SIGNATURE_FILENAME).write_text(
        bundle.signature.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / LINEAGE_FILENAME).write_text(
        bundle.lineage.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / RESULT_FILENAME).write_text(
        bundle.result.model_dump_json(indent=2), encoding="utf-8"
    )

    state_directory = directory / FITTED_STATE_DIRNAME
    state_directory.mkdir()
    bundle.state.save(state_directory)

    write_weights(directory / WEIGHTS_FILENAME)


def open_bundle(directory: Path, *, verify: bool = True) -> SavedBundle:
    """
    Locate a bundle on disk and read its manifest.

    Parameters
    ----------
    directory
        The bundle directory.
    verify
        Whether to re-hash every file and check it against the manifest.
        Defaults to true: the cost is worth paying by default, and a caller
        that is only listing bundles can opt out explicitly.

    Returns
    -------
    SavedBundle
        The bundle's directory and manifest.

    Raises
    ------
    BundleError
        If the directory or its manifest is missing, unreadable, or -- when
        verifying -- does not match what is on disk.
    """
    manifest = load_manifest(directory)
    if verify:
        verify_manifest(directory, manifest)
    return SavedBundle(directory=directory, manifest=manifest)


def load_manifest(directory: Path) -> Manifest:
    """
    Read a bundle's manifest.

    Parameters
    ----------
    directory
        The bundle directory.

    Returns
    -------
    Manifest
        The manifest.

    Raises
    ------
    BundleError
        If the manifest is missing or invalid.
    """
    return _read_contract(directory / MANIFEST_FILENAME, Manifest)


def load_signature(saved: SavedBundle) -> InputSignature:
    """
    Read a bundle's input signature.

    This is what makes a model rebuildable: the signature plus the spec is
    enough to reconstruct the model object without re-running the data build.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    InputSignature
        The saved signature.

    Raises
    ------
    BundleError
        If the file is missing or invalid.
    """
    return _read_contract(saved.signature_path, InputSignature)


def load_lineage(saved: SavedBundle) -> DataLineage:
    """
    Read a bundle's data lineage.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    DataLineage
        Where the training data came from, including the exact split indices.

    Raises
    ------
    BundleError
        If the file is missing or invalid.
    """
    return _read_contract(saved.lineage_path, DataLineage)


def load_result(saved: SavedBundle) -> TrainingResult:
    """
    Read a bundle's training result.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    TrainingResult
        Metrics and training history.

    Raises
    ------
    BundleError
        If the file is missing or invalid.
    """
    return _read_contract(saved.result_path, TrainingResult)


def load_spec(saved: SavedBundle) -> RunSpec:
    """
    Read the run specification a bundle was produced from.

    The counterpart to writing ``spec.json``, which Phase 1 did without
    providing a way to read it back -- invisible until something re-loaded a
    bundle, which nothing did until evaluation existed.

    Validated through the same loader a configuration file goes through, not
    merely parsed as JSON. A bundle that was written by a version whose
    schema has since changed should fail here, naming the field, rather than
    produce a specification object with a field quietly missing.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    BundleError
        If the file is missing, unreadable, or no longer validates.
    """
    path = saved.spec_path
    if not path.is_file():
        raise BundleError(f"expected {path.name} in the bundle at {path.parent}, but it is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"could not read {path}: {error}") from error
    try:
        return parse_run_spec(payload, origin=str(path))
    except SpecError as error:
        raise BundleError(
            f"{path} no longer validates as a run specification: {error}"
        ) from error


def load_fitted_state[StateT: FittedState](
    saved: SavedBundle,
    state_type: type[StateT],
) -> StateT:
    """
    Read a bundle's fitted state.

    The state type is supplied by the caller rather than recorded in the
    bundle, deliberately. Recording a class path would mean a bundle names a
    Python import, and renaming the class would break every bundle written
    before the rename -- the exact fragility this format avoids. The model
    definition knows its own state type, so it passes it in.

    Parameters
    ----------
    saved
        The located bundle.
    state_type
        The concrete state class to load into.

    Returns
    -------
    StateT
        The loaded state.

    Raises
    ------
    BundleError
        If the state directory is missing.
    """
    directory = saved.fitted_state_directory
    if not directory.is_dir():
        raise BundleError(
            f"bundle at {saved.directory} has no fitted state directory "
            f"({FITTED_STATE_DIRNAME}/); it may have been written by an "
            f"incompatible version"
        )
    return state_type.load(directory)


def _read_contract[ContractT: BaseModel](path: Path, contract_type: type[ContractT]) -> ContractT:
    """
    Read and validate one JSON contract file.

    One helper for every contract read, so the error message is uniform and
    always names the file -- a validation failure reported without the path is
    nearly useless when a bundle holds five JSON files.

    Parameters
    ----------
    path
        The file to read.
    contract_type
        The pydantic model to validate against.

    Returns
    -------
    ContractT
        The validated contract.

    Raises
    ------
    BundleError
        If the file is missing, unreadable, or fails validation.
    """
    if not path.is_file():
        raise BundleError(f"expected {path.name} in the bundle at {path.parent}, but it is absent")
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise BundleError(f"could not read {path}: {error}") from error
    try:
        return contract_type.model_validate_json(payload)
    except ValueError as error:
        raise BundleError(f"{path} is not a valid {contract_type.__name__}: {error}") from error
