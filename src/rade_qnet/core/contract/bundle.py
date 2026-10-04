"""
What a finished run produces, in memory and on disk.

A bundle is **self-describing**. Given only a bundle directory, the framework
can state which model produced it, from which spec, against which data
fingerprint, at which code version -- and can rebuild the model and invert its
target transforms without consulting the original run or the script that
launched it.

That property is what makes a six-month-old model usable rather than
archaeological, and it is why :class:`Manifest` carries a content hash for
every file. Corruption and silent drift then fail at load time with a message
naming the file, instead of surfacing as predictions that are subtly wrong.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import Field, model_validator

from ..lifecycle.errors import BundleError, SpecError
from ..spec.run import ReinforcementRunSpec, SupervisedRunSpec
from .base import ContractModel
from .data import DataLineage
from .result import TrainingResult
from .signature import InputSignature, PolicySignature
from .state import FittedState

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "Manifest",
    "ManifestEntry",
    "ModelBundle",
    "SavedBundle",
]

#: Schema version of the on-disk bundle layout. Incremented when the layout
#: changes incompatibly, so a reader can refuse a bundle it cannot understand
#: rather than loading it partially.
BUNDLE_SCHEMA_VERSION = 1

#: Canonical file and directory names inside a bundle. Named constants rather
#: than literals scattered across the writer and the reader, because the two
#: disagreeing is a failure that only shows up at load time.
MANIFEST_FILENAME = "manifest.json"
SPEC_FILENAME = "spec.json"
SIGNATURE_FILENAME = "signature.json"
LINEAGE_FILENAME = "lineage.json"
RESULT_FILENAME = "result.json"
WEIGHTS_FILENAME = "weights.bin"
FITTED_STATE_DIRNAME = "fitted_state"


class ManifestEntry(ContractModel):
    """
    One file in a bundle, with its content hash.

    Parameters
    ----------
    relative_path
        Path relative to the bundle directory, in POSIX form so a bundle
        written on one platform verifies on another.
    sha256
        Hexadecimal digest of the file's contents.
    size_bytes
        File size, recorded so a truncated file is detectable without reading
        it.
    """

    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)


class Manifest(ContractModel):
    """
    The bundle's own description of itself.

    Parameters
    ----------
    schema_version
        Layout version. A reader refuses a version it does not know.
    model_name
        Registered name of the model, which is how it is rebuilt.
    engine
        Registered name of the engine that trained it.
    version
        Monotonic version within the model and job, assigned by the catalog.
    created_at
        When the bundle was written.
    framework_version
        Version of ``rade_qnet`` that wrote it.
    spec_digest
        Digest of the run spec, so two bundles claiming the same
        configuration can be shown to have had it.
    job_id
        Job identifier within a job set, or ``None`` for a single run.
    files
        Every file in the bundle, with hashes.
    metrics
        Headline metrics, duplicated here so a catalog listing need not open
        the result file for every bundle it displays.
    tags
        Free-form labels for querying.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a file is listed more than once.
    """

    schema_version: int = Field(default=BUNDLE_SCHEMA_VERSION, ge=1)
    model_name: str
    engine: str
    version: int = Field(ge=1)
    created_at: datetime
    framework_version: str
    spec_digest: str
    job_id: str | None = None
    files: tuple[ManifestEntry, ...] = ()
    metrics: Mapping[str, float] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_files_are_unique(self) -> Manifest:
        """
        Reject a duplicated file path.

        Returns
        -------
        Manifest
            The validated manifest.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any path appears twice, which would make verification
            ambiguous about which hash is authoritative.
        """
        counts = Counter(entry.relative_path for entry in self.files)
        duplicates = sorted(path for path, count in counts.items() if count > 1)
        if duplicates:
            raise SpecError(f"manifest lists file(s) more than once: {duplicates}")
        return self

    @property
    def file_paths(self) -> tuple[str, ...]:
        """Every listed path, sorted."""
        return tuple(sorted(entry.relative_path for entry in self.files))

    def entry_for(self, relative_path: str) -> ManifestEntry:
        """
        Return the entry for one path.

        Parameters
        ----------
        relative_path
            Path relative to the bundle directory.

        Returns
        -------
        ManifestEntry
            The entry.

        Raises
        ------
        BundleError
            If the path is not listed.
        """
        for entry in self.files:
            if entry.relative_path == relative_path:
                return entry
        raise BundleError(
            f"{relative_path!r} is not listed in the manifest; "
            f"listed files are {list(self.file_paths)}"
        )

    @property
    def identifier(self) -> str:
        """
        A human-readable identifier, as used on the command line.

        Returns
        -------
        str
            ``model/job/vN``, or ``model/vN`` for a single run.
        """
        parts = [self.model_name]
        if self.job_id is not None:
            parts.append(self.job_id)
        parts.append(f"v{self.version}")
        return "/".join(parts)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """
    A finished run, in memory, before or after being written.

    A dataclass rather than a contract model because it holds a live model
    object and a live fitted state. Its serialisable parts are written as
    separate files, each hashed in the manifest.

    Parameters
    ----------
    model
        The trained model. Typed as ``object`` because ``core`` cannot name an
        engine's model type; the engine that produced it knows what it is.
    state
        What was fitted beyond the model's parameters. A policy fits nothing
        beyond its parameters, so an interactive run supplies
        :class:`~.state.IdentityFittedState`.
    signature
        The declared interface of the thing that was trained, sufficient to
        rebuild it.

        An :class:`~.signature.InputSignature` for a supervised model --
        inputs and a target. A :class:`~.signature.PolicySignature` for a
        policy -- an observation space and an action space, and no target,
        because a policy has none.

        A union rather than one type, because this field means "what does the
        saved object expect", and for a policy the honest answer is the
        second. The alternative was to record the experience stream's input
        signature instead, which would have type-checked and been wrong in a
        way that only shows up much later: a tensor description loses the
        number of discrete actions and the bounds of a continuous space, so
        the policy could no longer be rebuilt from its own bundle -- the one
        thing the field exists for.

        Readers that only make sense for one paradigm check which they have
        and refuse the other by name; see ``orchestration.stages.reload``.
    spec
        The run specification that produced this bundle.
    lineage
        Where the experience came from. For a supervised run that is the
        dataset and its split; for an interactive one it is the environment
        and how much was collected, recorded in the same shape so two runs
        remain comparable.
    result
        Metrics and training history.
    manifest
        Set once the bundle has been written. ``None`` beforehand, which is
        how "has this been persisted?" is answered without touching the
        filesystem.
    """

    model: object
    state: FittedState
    signature: InputSignature | PolicySignature
    spec: SupervisedRunSpec | ReinforcementRunSpec
    lineage: DataLineage
    result: TrainingResult
    manifest: Manifest | None = None

    @property
    def is_persisted(self) -> bool:
        """Whether this bundle has been written to disk."""
        return self.manifest is not None


@dataclass(frozen=True, slots=True)
class SavedBundle:
    """
    A bundle on disk, located and verified.

    Parameters
    ----------
    directory
        The bundle directory.
    manifest
        The manifest read from it.
    """

    directory: Path
    manifest: Manifest

    @property
    def spec_path(self) -> Path:
        """Path to the saved run specification."""
        return self.directory / SPEC_FILENAME

    @property
    def signature_path(self) -> Path:
        """Path to the saved input signature."""
        return self.directory / SIGNATURE_FILENAME

    @property
    def lineage_path(self) -> Path:
        """Path to the saved data lineage."""
        return self.directory / LINEAGE_FILENAME

    @property
    def result_path(self) -> Path:
        """Path to the saved training result."""
        return self.directory / RESULT_FILENAME

    @property
    def weights_path(self) -> Path:
        """Path to the saved model weights."""
        return self.directory / WEIGHTS_FILENAME

    @property
    def fitted_state_directory(self) -> Path:
        """Directory holding the saved fitted state."""
        return self.directory / FITTED_STATE_DIRNAME
