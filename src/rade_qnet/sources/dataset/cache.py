"""
Caching the expensive result of preparing a dataset.

What the cache key covers, and why
----------------------------------
:class:`DatasetCache` is content-addressed on three things: the digest of the
source specification, the fingerprint of the raw input -- from
:func:`~rade_qnet.sources.dataset.tables.fingerprint_source` -- and the
framework version. Any of them changing invalidates the entry.

The framework version is in the key for a reason that is easy to leave out.
A cached build is the *output of this code*, not just of the spec and the
input. Fixing a leakage defect in the splitter changes what a correct build
looks like without changing either the spec or the data, so a cache keyed only
on those two would keep serving the leaky arrays after the fix shipped. That
failure is invisible: the run succeeds, the metrics look like yesterday's, and
nothing reports that the fix did not take effect.

What is cached is the *prepared* dataset -- transformed arrays, splits, fitted
state -- and not a :class:`~rade_qnet.core.contract.data.DataBundle`. A bundle
holds live loaders, which are not serialisable, and packaging prepared arrays
into loaders is cheap. Caching the stage before the cheap stage is the whole
point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import DataLineage, SplitIndices
from ...core.contract.signature import InputSignature
from ...core.lifecycle.errors import BundleError, ContractError
from ...core.provenance.hashing import abbreviate_digest, digest_payload
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.state import FittedState

__all__ = ["DatasetCache", "PreparedDataset"]

_LOGGER = get_logger(__name__)

#: Filenames inside a cache entry. Part of the on-disk format.
_ARRAYS_FILENAME = "arrays.npz"
_METADATA_FILENAME = "metadata.json"
_STATE_DIRECTORY = "state"

#: Where a build's static inputs are cached, separate from the arrays file
#: so that a model's signature keys cannot collide with the fixed ones.
_STATIC_FILENAME = "static.npz"

#: Length of the cache subdirectory name. Long enough that a collision across
#: every build a person will ever inspect is not a practical concern.
_KEY_LENGTH = 16


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """
    A data build after the expensive work and before the engine-specific work.

    This is the cacheable unit: transformed arrays, the splits, the fitted
    state, the signature and the lineage. Turning it into a
    :class:`~rade_qnet.core.contract.data.DataBundle` means constructing loaders
    or array payloads, which is cheap and engine-specific -- so it happens on
    every run rather than being stored.

    Parameters
    ----------
    features
        Transformed feature matrix over the whole scenario axis, not split.
        Held whole because a sequence window near a split boundary needs rows
        from before that split's first index, and slicing per split first
        would make those rows unreachable.
    target
        Transformed target, aligned with ``features``.
    splits
        Which scenario indices belong to each split.
    signature
        The declared interface between this data and a model.
    state
        What was fitted while preparing the data.
    lineage
        Where the data came from.
    feature_names
        Column names after reduction, in column order.
    entity_ids
        Identifier per row, where the problem has an entity axis.
    static_inputs
        Inputs that are identical for every sample, keyed as the signature
        declares them. A graph is the motivating case: the adjacency does
        not vary by scenario, so collating a copy of it into every sample
        multiplies its memory by the batch size and its transfer cost by
        the number of batches -- defect 4. Empty for a model without any,
        which is most of them.

    Raises
    ------
    ContractError
        If the arrays disagree on row count, or a split index falls outside
        the scenario axis -- which would otherwise surface as a confusing
        out-of-bounds error during windowing.
    """

    features: NDArray[np.floating]
    target: NDArray[np.floating]
    splits: SplitIndices
    signature: InputSignature
    state: FittedState
    lineage: DataLineage
    feature_names: tuple[str, ...] | None = None
    entity_ids: tuple[str, ...] | None = None
    static_inputs: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate row alignment and index bounds."""
        n_rows = self.features.shape[0]
        if self.target.shape[0] != n_rows:
            raise ContractError(
                f"features have {n_rows} row(s) but target has {self.target.shape[0]}"
            )
        for name in ("train", "validation", "test"):
            indices = self.splits[name]
            if indices.size and (indices.min() < 0 or indices.max() >= n_rows):
                raise ContractError(
                    f"{name} split indices span [{indices.min()}, {indices.max()}] "
                    f"but the scenario axis has {n_rows} row(s)"
                )

    @property
    def n_scenarios(self) -> int:
        """Length of the scenario axis before splitting."""
        return int(self.features.shape[0])


class DatasetCache:
    """
    A content-addressed store of prepared datasets.

    Parameters
    ----------
    directory
        Root of the store. One subdirectory per entry, named for the key.
    enabled
        When false, every lookup misses and every write is skipped. A disabled
        cache rather than an absent one, so a caller never branches on whether
        caching is configured -- there is one code path and it is the cached
        one.
    """

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled

    def key(self, *, spec_digest: str, source_fingerprint: str, framework_version: str) -> str:
        """
        Return the cache key for a build.

        Parameters
        ----------
        spec_digest
            Digest of the source specification.
        source_fingerprint
            Digest of the raw input, from :func:`fingerprint_source`.
        framework_version
            Version of ``rade_qnet``. In the key for the reason given in the
            module docstring: a cached build is the output of this code, so a
            fix to the splitter must invalidate it.

        Returns
        -------
        str
            A short hexadecimal key, usable as a directory name.
        """
        return abbreviate_digest(
            digest_payload(
                {
                    "spec": spec_digest,
                    "source": source_fingerprint,
                    "version": framework_version,
                }
            ),
            length=_KEY_LENGTH,
        )

    def path_for(self, key: str) -> Path:
        """
        Return the directory an entry occupies.

        Parameters
        ----------
        key
            A cache key.

        Returns
        -------
        pathlib.Path
            The entry directory, which may not exist.
        """
        return self.directory / key

    def load(self, key: str, *, state_type: type[FittedState]) -> PreparedDataset | None:
        """
        Return a cached dataset, or ``None`` on a miss.

        A miss is a return value rather than an exception: missing is the
        normal first-run condition, and a caller that has to catch an
        exception for the normal case writes worse code.

        A *corrupt* entry is different, and is treated as a miss with a
        warning rather than as a failure. A half-written entry -- an
        interrupted run, a full disk -- should cost a rebuild, not a crash
        hours into a job set. The warning is what keeps it from being silent.

        Parameters
        ----------
        key
            A cache key, from :meth:`key`.
        state_type
            The concrete fitted-state class to load. Passed in rather than
            recorded on disk, for the reason given in
            :class:`~rade_qnet.core.contract.state.FittedState`.

        Returns
        -------
        PreparedDataset or None
            The cached dataset, or ``None``.
        """
        if not self.enabled:
            return None

        entry = self.path_for(key)
        if not (entry / _METADATA_FILENAME).is_file():
            _LOGGER.debug("dataset cache miss for key %s", key)
            return None

        try:
            return self._read_entry(entry, state_type=state_type)
        except (OSError, ValueError, KeyError, BundleError, ContractError) as error:
            _LOGGER.warning(
                "ignoring unreadable dataset cache entry %s (%s: %s); rebuilding",
                key,
                type(error).__name__,
                error,
            )
            return None

    def _read_entry(self, entry: Path, *, state_type: type[FittedState]) -> PreparedDataset:
        """
        Read one entry from disk.

        Separated from :meth:`load` so that the error handling there wraps a
        single call, and every failure mode inside it is treated the same way.

        Parameters
        ----------
        entry
            The entry directory.
        state_type
            The concrete fitted-state class to load.

        Returns
        -------
        PreparedDataset
            The stored dataset.
        """
        metadata = json.loads((entry / _METADATA_FILENAME).read_text(encoding="utf-8"))
        with np.load(entry / _ARRAYS_FILENAME, allow_pickle=False) as arrays:
            features = arrays["features"]
            target = arrays["target"]
            splits = SplitIndices(
                train=arrays["train"],
                validation=arrays["validation"],
                test=arrays["test"],
            )

        with np.load(entry / _STATIC_FILENAME, allow_pickle=False) as stored:
            static_inputs = {name: stored[name] for name in stored.files}

        names = metadata["feature_names"]
        entities = metadata["entity_ids"]
        return PreparedDataset(
            features=features,
            target=target,
            splits=splits,
            signature=InputSignature.model_validate(metadata["signature"]),
            state=state_type.load(entry / _STATE_DIRECTORY),
            lineage=DataLineage.model_validate(metadata["lineage"]),
            feature_names=tuple(names) if names is not None else None,
            entity_ids=tuple(entities) if entities is not None else None,
            static_inputs=static_inputs,
        )

    def save(self, key: str, dataset: PreparedDataset) -> None:
        """
        Write a dataset into the store.

        Writes with ``allow_pickle=False``, which is not a detail. A cache
        directory is an ordinary writable path, and loading a pickle from one
        executes whatever it contains -- so an entry that an attacker, or a
        stale incompatible version, could place there would run as code. The
        arrays here are plain numeric data and need nothing more.

        Parameters
        ----------
        key
            A cache key, from :meth:`key`.
        dataset
            The dataset to store.
        """
        if not self.enabled:
            return

        entry = self.path_for(key)
        entry.mkdir(parents=True, exist_ok=True)
        (entry / _STATE_DIRECTORY).mkdir(parents=True, exist_ok=True)

        np.savez(
            entry / _ARRAYS_FILENAME,
            features=dataset.features,
            target=dataset.target,
            train=dataset.splits.train,
            validation=dataset.splits.validation,
            test=dataset.splits.test,
        )
        # Written to their own file rather than into the arrays file,
        # because their names come from the model's signature and could
        # otherwise collide with `features`, `target` or a split name.
        np.savez(entry / _STATIC_FILENAME, **dict(dataset.static_inputs))
        dataset.state.save(entry / _STATE_DIRECTORY)

        metadata = {
            "signature": dataset.signature.model_dump(mode="json"),
            "lineage": dataset.lineage.model_dump(mode="json"),
            "feature_names": list(dataset.feature_names)
            if dataset.feature_names is not None
            else None,
            "entity_ids": list(dataset.entity_ids) if dataset.entity_ids is not None else None,
        }
        # Metadata is written last, and `load` keys presence off it.  That
        # ordering is what makes an interrupted write read as a miss rather
        # than as an entry with arrays and no description of them.
        (entry / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        _LOGGER.info("cached prepared dataset under key %s", key)
