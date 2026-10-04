"""
Reading raw tabular data from disk, and saying what was read.

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

Why the fingerprint lives here
-------------------------------
:func:`fingerprint_source` is what :mod:`~rade_qnet.sources.dataset.cache`
keys its entries on, so the two modules are coupled -- and the coupling is
deliberately an import rather than an adjacency. What counts as "the input" is
a question about reading, answered once, here; the cache is one of its
callers. While both halves shared a module that relationship was true and
invisible, which is the kind of thing that survives a refactor by accident
rather than on purpose.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from ...core.lifecycle.errors import BundleError, ContractError, SpecError
from ...core.provenance.hashing import digest_file, digest_payload
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ["TableData", "fingerprint_source", "read_table"]

_LOGGER = get_logger(__name__)

#: File suffixes handled natively, by the standard library CSV reader.
_DELIMITED_SUFFIXES: Mapping[str, str] = {".csv": ",", ".tsv": "\t", ".txt": ","}

#: File suffixes handled through pandas, when it is installed.
_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})


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
            f"(see the note in sources.dataset.tables). Install it with "
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
