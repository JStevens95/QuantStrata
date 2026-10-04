"""
Reading raw tabular data, and caching the expensive result of preparing it.

Two responsibilities that belong together because they are the two ends of the
same decision: what counts as "the input" is exactly what the cache key must
cover.

Why CSV is parsed here instead of by pandas
-------------------------------------------
``rade_qnet`` does not depend on pandas. That is a deliberate cost: parsing CSV
with the standard library is more code than one ``read_csv`` call.

The reason is the framework's stated promise that a host which only reads a
bundle need not install a training stack -- and pandas is a large dependency to
impose on every consumer for the benefit of the one source kind that reads
files. A deployment that already has pandas loses nothing, because
:func:`read_table` dispatches Parquet through it when a Parquet path is
supplied, and says precisely what to install when it is absent.

The alternative -- depending on pandas and using it everywhere -- was rejected
rather than overlooked. It would be less code here and a heavier install for
everyone.

What the cache key covers, and why
----------------------------------
:class:`DatasetCache` is content-addressed on three things: the digest of the
source specification, the fingerprint of the raw input, and the framework
version. Any of them changing invalidates the entry.

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

import csv
import json
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import DataLineage, SplitIndices
from ...core.contract.signature import InputSignature
from ...core.runtime.errors import BundleError, ContractError, SpecError
from ...core.runtime.hashing import abbreviate_digest, digest_file, digest_payload
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.state import FittedState

__all__ = [
    "DatasetCache",
    "PreparedDataset",
    "TableData",
    "fingerprint_source",
    "read_table",
]

_LOGGER = get_logger(__name__)

#: File suffixes handled natively, by the standard library CSV reader.
_DELIMITED_SUFFIXES: Mapping[str, str] = {".csv": ",", ".tsv": "\t", ".txt": ","}

#: File suffixes handled through pandas, when it is installed.
_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})

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
class TableData:
    """
    One tabular file, read into arrays and still in its original units.

    Deliberately not a data frame. Everything downstream of here wants a float
    matrix, a target vector and a handful of string columns, and naming those
    four things in a type means a transform cannot quietly depend on a column
    that was never declared.

    Parameters
    ----------
    features
        Samples by features, in ``feature_names`` order.
    target
        One value per sample.
    feature_names
        Column names in column order. Carried rather than reconstructed,
        because a feature importance table reported by index is nearly
        useless to the person reading it.
    attributes
        Non-numeric columns kept aside, by name: an entity identifier, a group
        key for a grouped split. Strings rather than codes, because encoding
        them is a fitted transform's job and doing it during reading would put
        the fit in the wrong place.

    Raises
    ------
    ContractError
        If the row counts disagree. Checked here rather than left to the first
        transform, because a feature matrix misaligned with its target trains
        without complaint and learns a permutation.
    """

    features: NDArray[np.float64]
    target: NDArray[np.float64]
    feature_names: tuple[str, ...]
    attributes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that every column has one entry per sample."""
        n_rows = self.features.shape[0]
        if self.target.shape[0] != n_rows:
            raise ContractError(
                f"features have {n_rows} row(s) but target has "
                f"{self.target.shape[0]}; they must correspond one to one"
            )
        if len(self.feature_names) != self.features.shape[1]:
            raise ContractError(
                f"feature_names has {len(self.feature_names)} entry(ies) but "
                f"features has {self.features.shape[1]} column(s)"
            )
        for name, values in self.attributes.items():
            if len(values) != n_rows:
                raise ContractError(
                    f"attribute column {name!r} has {len(values)} value(s) but "
                    f"there are {n_rows} row(s)"
                )

    @property
    def n_rows(self) -> int:
        """Number of samples."""
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.features.shape[1])

    def attribute(self, name: str) -> tuple[str, ...]:
        """
        Return one attribute column.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        tuple of str
            One value per row.

        Raises
        ------
        ContractError
            If the column was not read. The message lists what was read,
            because the usual cause is a grouped split naming a ``group_key``
            that the reader was never told to keep.
        """
        try:
            return self.attributes[name]
        except KeyError:
            raise ContractError(
                f"attribute column {name!r} was not read; columns kept aside are "
                f"{sorted(self.attributes)}. Name it in attribute_columns when "
                f"reading if a split or transform needs it"
            ) from None


def read_table(
    path: Path,
    *,
    target_column: str = "target",
    feature_columns: Sequence[str] | None = None,
    attribute_columns: Sequence[str] = (),
) -> TableData:
    """
    Read a tabular file into a :class:`TableData`.

    Parameters
    ----------
    path
        File to read. The suffix selects the reader.
    target_column
        Column holding the target.
    feature_columns
        Columns to use as features, in the order given. ``None`` means every
        remaining numeric column, sorted by name -- sorted rather than
        file order, so two files with the same columns in a different order
        produce the same matrix and therefore the same cache key.
    attribute_columns
        Columns to keep aside as strings.

    Returns
    -------
    TableData
        The parsed table.

    Raises
    ------
    SpecError
        If the suffix is not supported, or a named column is absent.
    BundleError
        If the file does not exist.
    """
    if not path.is_file():
        raise BundleError(f"no such file: {path}")

    suffix = path.suffix.lower()
    if suffix in _DELIMITED_SUFFIXES:
        columns = _read_delimited(path, delimiter=_DELIMITED_SUFFIXES[suffix])
    elif suffix in _PARQUET_SUFFIXES:
        columns = _read_parquet(path)
    else:
        supported = sorted({*_DELIMITED_SUFFIXES, *_PARQUET_SUFFIXES})
        raise SpecError(
            f"cannot read {path.name}: suffix {suffix!r} is not supported (supported: {supported})"
        )

    return _assemble(
        columns,
        source=path.name,
        target_column=target_column,
        feature_columns=feature_columns,
        attribute_columns=attribute_columns,
    )


def _read_delimited(path: Path, *, delimiter: str) -> dict[str, list[str]]:
    """
    Read a delimited text file into a column-per-key mapping of raw strings.

    Values are left as strings so that one place -- :func:`_assemble` --
    decides what is numeric and what is an attribute. Converting during the
    read would mean deciding before the caller's column choices are known.

    Parameters
    ----------
    path
        File to read.
    delimiter
        Field separator.

    Returns
    -------
    dict
        Column name to raw values.

    Raises
    ------
    SpecError
        If the file has no header row, or a row's field count disagrees with
        the header. A short row usually means an unquoted delimiter inside a
        value, and silently padding it would shift every subsequent column.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise SpecError(f"{path.name} is empty; expected a header row") from None

        columns: dict[str, list[str]] = {name.strip(): [] for name in header}
        if len(columns) != len(header):
            duplicated = sorted({name for name in header if header.count(name) > 1})
            raise SpecError(
                f"{path.name} has duplicate column name(s) {duplicated}; "
                f"column names must be unique"
            )

        names = list(columns)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue  # A trailing blank line is not an error.
            if len(row) != len(names):
                raise SpecError(
                    f"{path.name} line {line_number} has {len(row)} field(s) but "
                    f"the header declares {len(names)}; check for an unquoted "
                    f"{delimiter!r} inside a value"
                )
            for name, value in zip(names, row, strict=True):
                columns[name].append(value.strip())

    return columns


def _read_parquet(path: Path) -> dict[str, list[str]]:
    """
    Read a Parquet file through pandas, if pandas is installed.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    dict
        Column name to values as strings, matching the delimited reader so
        that :func:`_assemble` has one input shape to handle.

    Raises
    ------
    SpecError
        If pandas is not installed, naming what to install. An import error
        from six frames down would not tell the reader that the fix is a pip
        install and that CSV would work without one.
    """
    if find_spec("pandas") is None:
        raise SpecError(
            f"reading {path.name} needs pandas, which rade_qnet does not require "
            f"(see the note in sources.dataset.io). Install it with "
            f"'pip install pandas pyarrow', or supply the data as CSV"
        )

    import pandas  # noqa: PLC0415  Imported lazily: see the check above.

    frame = pandas.read_parquet(path)
    return {str(name): [str(value) for value in frame[name]] for name in frame.columns}


def _assemble(
    columns: Mapping[str, Sequence[str]],
    *,
    source: str,
    target_column: str,
    feature_columns: Sequence[str] | None,
    attribute_columns: Sequence[str],
) -> TableData:
    """
    Turn raw string columns into a :class:`TableData`.

    Parameters
    ----------
    columns
        Column name to raw values.
    source
        Filename, for error messages.
    target_column
        Column holding the target.
    feature_columns
        Chosen feature columns, or ``None`` to infer.
    attribute_columns
        Columns to keep aside as strings.

    Returns
    -------
    TableData
        The assembled table.

    Raises
    ------
    SpecError
        If a named column is absent, if a chosen feature column is not
        numeric, or if no feature column remains.
    """
    available = sorted(columns)
    _require_columns([target_column, *attribute_columns], available=available, source=source)

    if feature_columns is None:
        # Inferred: everything that is neither the target nor an attribute and
        # that parses as numbers.  Sorted for a stable cache key.
        candidates = sorted(set(columns) - {target_column, *attribute_columns})
        chosen = tuple(name for name in candidates if _parse_floats(columns[name]) is not None)
        if not chosen:
            raise SpecError(
                f"{source}: no numeric feature column found among {candidates}; "
                f"name the feature columns explicitly if they need conversion first"
            )
        _LOGGER.debug("inferred %d feature column(s) from %s: %s", len(chosen), source, chosen)
    else:
        _require_columns(feature_columns, available=available, source=source)
        chosen = tuple(feature_columns)

    feature_arrays = []
    for name in chosen:
        values = _parse_floats(columns[name])
        if values is None:
            raise SpecError(
                f"{source}: feature column {name!r} is not numeric; drop it, or "
                f"keep it aside as an attribute column and encode it"
            )
        feature_arrays.append(values)

    target = _parse_floats(columns[target_column])
    if target is None:
        raise SpecError(f"{source}: target column {target_column!r} is not numeric")

    # `column_stack` on an empty list raises; the no-column case is already
    # refused above, so reaching here guarantees at least one column.
    return TableData(
        features=np.column_stack(feature_arrays),
        target=target,
        feature_names=chosen,
        attributes={name: tuple(columns[name]) for name in attribute_columns},
    )


def _require_columns(requested: Sequence[str], *, available: Sequence[str], source: str) -> None:
    """
    Raise if any requested column is absent.

    Parameters
    ----------
    requested
        Column names the caller asked for.
    available
        Column names present in the file.
    source
        Filename, for the error message.

    Raises
    ------
    SpecError
        Naming the absent columns and listing what is present, because the
        usual cause is a typo or a renamed upstream column and the fix is
        obvious once the real names are visible.
    """
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SpecError(
            f"{source}: column(s) {missing} are not present; the file has {list(available)}"
        )


def _parse_floats(values: Sequence[str]) -> NDArray[np.float64] | None:
    """
    Parse a column as floats, or return ``None`` if it is not numeric.

    An empty field becomes ``nan`` rather than failing, because a missing
    value is a data condition the quality report is designed to surface,
    whereas a non-numeric *column* is a configuration mistake.

    Parameters
    ----------
    values
        Raw strings.

    Returns
    -------
    numpy.ndarray or None
        The parsed column, or ``None`` if any non-empty value is not a number.
    """
    parsed = np.empty(len(values), dtype=np.float64)
    for index, raw in enumerate(values):
        if raw == "":
            parsed[index] = np.nan
            continue
        try:
            parsed[index] = float(raw)
        except ValueError:
            return None
    return parsed


def fingerprint_source(path: Path | None, *, extra: Mapping[str, object] | None = None) -> str:
    """
    Return a digest identifying the raw input of a build.

    Hashes file *contents* rather than a path and modification time. A path is
    not the data -- two runs on two machines resolve the same path to different
    files -- and a modification time changes when nothing did, so a cache keyed
    on it misses after a checkout.

    Parameters
    ----------
    path
        The input file, or ``None`` for a build with no single file input --
        a model data module assembling several sources, which supplies
        ``extra`` instead.
    extra
        Additional material to fold in: query parameters, a list of upstream
        digests, a universe definition.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    material: dict[str, object] = {"extra": dict(extra or {})}
    material["file"] = digest_file(path) if path is not None else None
    return digest_payload(material)


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
