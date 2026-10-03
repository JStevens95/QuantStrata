"""
Stable content hashes for specifications, arrays and files.

These digests are used as cache keys and as bundle provenance, which imposes a
requirement stronger than it first appears: **the digest of equal inputs must
be equal in a different process, on a different day, on a different machine.**

Python's built-in ``hash()`` satisfies none of that. String hashing is salted
per interpreter, so ``hash("a")`` differs between two processes of the same
program. A cache keyed on it misses every time, and a hyper-parameter search
silently rebuilds its dataset once per trial instead of once in total. Every
digest here is therefore a SHA-256 over a canonical byte encoding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import BaseModel

__all__ = [
    "abbreviate_digest",
    "canonical_json",
    "digest_arrays",
    "digest_file",
    "digest_payload",
    "digest_spec",
]

#: Read size for file hashing. Large enough that the loop is not syscall-bound,
#: small enough that a multi-gigabyte checkpoint is not held in memory.
_FILE_CHUNK_BYTES = 1 << 20


def _json_default(value: object) -> Any:  # noqa: ANN401 - json hook signature
    """
    Encode values ``json`` cannot serialise on its own.

    Only types that genuinely appear in specifications and lineage are
    handled. Anything else raises, which is deliberate: silently encoding an
    unknown object via ``repr`` would produce a digest that changes when an
    unrelated ``__repr__`` changes.

    Parameters
    ----------
    value
        The object ``json.dumps`` could not encode.

    Returns
    -------
    Any
        A JSON-encodable replacement.

    Raises
    ------
    TypeError
        If the value has no canonical encoding.
    """
    if isinstance(value, Path):
        # Posix form so a digest computed on Windows matches one from Linux.
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        # Sets have no order, so one must be imposed or the digest is unstable.
        return sorted(str(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"no canonical JSON encoding for type {type(value).__name__!r}")


def canonical_json(payload: object) -> str:
    """
    Render a payload as JSON in a form that depends only on its content.

    Keys are sorted and separators are fixed, so two mappings that differ only
    in insertion order produce identical text -- and therefore identical
    digests.

    Parameters
    ----------
    payload
        Any JSON-encodable structure, plus the extra types handled by the
        module's encoder hook.

    Returns
    -------
    str
        Canonical JSON text.

    Raises
    ------
    TypeError
        If the payload contains a value with no canonical encoding.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def digest_payload(payload: object) -> str:
    """
    Return the SHA-256 digest of a payload's canonical JSON form.

    Parameters
    ----------
    payload
        Any structure acceptable to :func:`canonical_json`.

    Returns
    -------
    str
        Lowercase hexadecimal digest, 64 characters.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def digest_spec(spec: BaseModel) -> str:
    """
    Return the digest of a specification.

    The spec is dumped in JSON mode first, so that ``Path``, ``Enum`` and
    similar fields are reduced to their serialised form. This matters because
    the digest must match the one computed from the same spec after it has
    been written to disk and read back -- that round trip is what a step cache
    relies on.

    Parameters
    ----------
    spec
        Any pydantic model.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    return digest_payload(spec.model_dump(mode="json"))


def digest_file(path: Path, *, chunk_bytes: int = _FILE_CHUNK_BYTES) -> str:
    """
    Return the SHA-256 digest of a file's contents.

    Read in chunks so that hashing a large checkpoint does not require holding
    it in memory.

    Parameters
    ----------
    path
        File to hash.
    chunk_bytes
        Read size in bytes.

    Returns
    -------
    str
        Lowercase hexadecimal digest.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """
    Return a digest over a named collection of arrays.

    Shape and dtype are folded into the digest alongside the bytes, so two
    arrays holding the same values in different shapes or precisions hash
    differently. Without that, a ``float32`` and ``float64`` view of the same
    data would be treated as interchangeable by a cache, and they are not.

    Parameters
    ----------
    arrays
        Mapping of name to array. Names are hashed in sorted order, so the
        mapping's iteration order is irrelevant.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def abbreviate_digest(digest: str, *, length: int = 12) -> str:
    """
    Shorten a digest for use in a directory name or log line.

    Parameters
    ----------
    digest
        A full hexadecimal digest.
    length
        Number of leading characters to keep. Twelve hexadecimal characters is
        48 bits, which makes an accidental collision across the number of runs
        a person will ever inspect by hand effectively impossible.

    Returns
    -------
    str
        The leading ``length`` characters.
    """
    return digest[:length]
