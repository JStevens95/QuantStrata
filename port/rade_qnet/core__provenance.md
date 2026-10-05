# `tranql/models/rade/rade_qnet/rade_qnet/core/provenance`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 50 | 2311 | `b556b6e4ac00495d` |
| 2 | `hashing.py` | 231 | 6505 | `f6d9f4a484edae26` |
| 3 | `logging.py` | 275 | 9905 | `00e4746f8dd6ecf0` |
| 4 | `seeding.py` | 240 | 7831 | `030643171325087b` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/core/provenance/__init__.py`

2311 bytes · SHA-256 `b556b6e4ac00495d`

```python
"""
What every run can prove about itself, whatever it computed.

Three guarantees, and they are the three things a model-risk review asks for.
Together they are the difference between a number and a number you can defend.

``seeding``
    The same numbers, twice.  The same specification on the same data
    produces the same weights, and a fanned-out job set produces the same
    results run in parallel as run one after another.

``hashing``
    The same identity, on another machine, in a year.  A digest of equal
    inputs must be equal in a different process, on a different day --
    which Python's built-in ``hash`` does not give you, because string
    hashing is salted per interpreter.  These digests are cache keys and
    bundle provenance, so an unstable one quietly means a cache that never
    hits and a lineage that cannot be checked.

``logging``
    The narrative, legible among forty concurrent runs.  Identifiers live in
    context variables rather than in a logger threaded through every
    signature, so code that has nothing to do with logging does not acquire
    a logging parameter.

Why these are not "utilities"
------------------------------
They were, in a ``runtime`` package that also held the registry, the pipeline
base and the run context -- eight modules whose only shared property was
being needed everywhere.  That is a bin, and a bin is where things go to stop
being findable.

What separates these three from the rest of that package is *who reads them*.
:mod:`rade_qnet.core.lifecycle` is read by someone extending the framework:
how do I register a model, override a stage, attach a hook.  This package is
read by someone who has to answer for a result: why is this number what it
is, and can you produce it again.

Dependency note
---------------
``seeding`` seeds what is installed without importing it.  ``core`` may not
depend on a training library, yet seeding PyTorch is exactly the kind of
thing that belongs here, so an engine registers a seeding callback when it is
imported and ``seed_everything`` calls whatever has registered.  The torch
half lives in :mod:`rade_qnet.engines.torch.hardware.determinism`, because
what it seeds is CUDA generators and cuDNN algorithm selection -- device
facts, not framework facts.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/core/provenance/hashing.py`

6505 bytes · SHA-256 `f6d9f4a484edae26`

```python
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
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/core/provenance/logging.py`

9905 bytes · SHA-256 `00e4746f8dd6ecf0`

```python
"""
Structured logging with contextual run identifiers.

A forty-job run produces forty interleaved streams of log output. Without
identifiers attached to every record, that output is unreadable, and the usual
remedy -- threading a pre-configured logger through every function -- puts an
infrastructure parameter into the signature of code that has nothing to do
with logging.

Instead the identifiers live in :class:`~contextvars.ContextVar` slots. Any
module can call :func:`get_logger` and its records are automatically stamped
with the current run, job and stage.

Crossing a process boundary
---------------------------
Context variables do not survive ``fork`` or ``spawn``. A worker must re-bind
them, which is what :func:`context_payload` and :func:`apply_context_payload`
are for: the parent serialises the context into the job payload, and the
worker applies it before doing any work.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Final

__all__ = [
    "ContextFilter",
    "apply_context_payload",
    "bound_context",
    "configure_logging",
    "context_payload",
    "current_context",
    "get_logger",
]

#: Identifier of the current run. ``None`` outside a run.
RUN_ID: ContextVar[str | None] = ContextVar("rade_qnet_run_id", default=None)

#: Identifier of the current job within a job set. ``None`` for a single run.
JOB_ID: ContextVar[str | None] = ContextVar("rade_qnet_job_id", default=None)

#: Name of the pipeline stage currently executing. ``None`` between stages.
STAGE: ContextVar[str | None] = ContextVar("rade_qnet_stage", default=None)

_CONTEXT_VARIABLES: Final = {"run_id": RUN_ID, "job_id": JOB_ID, "stage": STAGE}

#: Root logger name. Every framework logger is a descendant, so a user can
#: raise or lower the framework's verbosity with a single call without
#: touching their own loggers.
ROOT_LOGGER_NAME: Final = "rade_qnet"

#: The name this package was actually imported under: ``tranql.models.rade.rade_qnet.rade_qnet`` from
#: the repository, ``rade_qnet`` when installed, or a deeper name such as
#: ``tranql.models.rade.rade_qnet.rade_qnet`` where it is vendored into a
#: larger tree. Read off this module's own ``__name__`` rather than assumed,
#: because the package's internal imports are all relative and so it can be
#: mounted anywhere -- and the logger names must not depend on where.
_IMPORTED_AS: Final = __name__.removesuffix(".core.provenance.logging")

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s [%(rade_qnet_context)s] %(name)s: %(message)s"
_TIME_FORMAT: Final = "%Y-%m-%d %H:%M:%S"


class ContextFilter(logging.Filter):
    """
    Attach the current run, job and stage identifiers to every record.

    Implemented as a filter rather than an adapter so that records emitted by
    code which knows nothing about this module -- including a third-party
    library logging under the framework's logger tree -- are stamped too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add a ``rade_qnet_context`` attribute to the record.

        Parameters
        ----------
        record
            The record being emitted. Mutated in place.

        Returns
        -------
        bool
            Always ``True``: this filter annotates, it never discards.
        """
        identifiers = current_context()
        # A compact single field rather than three, so the format string stays
        # readable and a line with no context does not carry empty brackets.
        record.rade_qnet_context = (
            " ".join(f"{key}={value}" for key, value in identifiers.items()) or "-"
        )
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Return the framework logger for a module.

    Parameters
    ----------
    name
        Usually ``__name__``. A leading ``rade_qnet`` is not duplicated, and any
        other name is placed under the framework's logger tree so that
        configuring one logger configures all of them.

    Returns
    -------
    logging.Logger
        A logger beneath :data:`ROOT_LOGGER_NAME`.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    # Replace whatever the package was imported as with the root name, so
    # `tranql.models.rade.rade_qnet.rade_qnet.core.spec.run`, `rade_qnet.core.spec.run` and a vendored
    # `tranql....rade_qnet.core.spec.run` all log as `rade_qnet.core.spec.run`.
    # A logging configuration written against `rade_qnet.core` must keep
    # matching wherever the package is mounted, and a log stream should not
    # reveal which mount produced it.
    if name.startswith(f"{_IMPORTED_AS}."):
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name.removeprefix(f'{_IMPORTED_AS}.')}")
    trimmed = name.removeprefix("src.")
    if trimmed.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(trimmed)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{trimmed}")


def configure_logging(
    *,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Install a handler on the framework's root logger.

    Called explicitly, never at import. Importing ``rade_qnet`` must not change
    how an application's logging behaves -- a library that configures logging
    on import is a library that silently redirects somebody else's output.

    Parameters
    ----------
    level
        Threshold for the framework's logger.
    stream
        Destination. Defaults to standard error, so log output does not
        contaminate a program's results on standard output.
    force
        Replace any handler this function previously installed. Without it the
        call is idempotent, so repeated calls cannot produce duplicated lines.

    Returns
    -------
    logging.Logger
        The configured framework root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Do not propagate to the root logger: an application that has configured
    # its own root handler would otherwise see every framework line twice.
    logger.propagate = False

    existing = [handler for handler in logger.handlers if getattr(handler, "_rade_qnet", False)]
    if existing and not force:
        return logger
    for handler in existing:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_TIME_FORMAT))
    handler.addFilter(ContextFilter())
    # Marked so a later call can recognise its own handler and leave any
    # handler the application added alone.
    handler._rade_qnet = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def current_context() -> dict[str, str]:
    """
    Return the identifiers currently bound, omitting those that are unset.

    Returns
    -------
    dict of str to str
        Mapping with any of ``run_id``, ``job_id`` and ``stage`` that are set.
    """
    return {
        name: value
        for name, variable in _CONTEXT_VARIABLES.items()
        if (value := variable.get()) is not None
    }


@contextmanager
def bound_context(
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    stage: str | None = None,
) -> Iterator[None]:
    """
    Bind identifiers for the duration of a block.

    Only the arguments given are changed; the rest keep their current values.
    That is what lets a stage bind ``stage`` without needing to know, or
    repeat, the run it belongs to.

    Every variable is restored on exit, including when the block raises, so a
    failed stage cannot leave its name attached to subsequent log lines.

    Parameters
    ----------
    run_id, job_id, stage
        Values to bind. ``None`` leaves the existing value in place.

    Yields
    ------
    None
    """
    updates = {"run_id": run_id, "job_id": job_id, "stage": stage}
    tokens = [
        _CONTEXT_VARIABLES[name].set(value) for name, value in updates.items() if value is not None
    ]
    try:
        yield
    finally:
        # Reset in reverse order so nested bindings unwind correctly.
        for token in reversed(tokens):
            token.var.reset(token)


def context_payload() -> dict[str, str]:
    """
    Serialise the current context for transport to a worker process.

    Returns
    -------
    dict of str to str
        A plain, picklable mapping suitable for inclusion in a job payload.
    """
    return current_context()


def apply_context_payload(payload: Mapping[str, str]) -> None:
    """
    Re-bind identifiers in a worker process.

    Unlike :func:`bound_context` this does not restore anything, because a
    worker's context should last for the whole of its task.

    This **replaces** the context rather than merging into it: an identifier
    absent from the payload is cleared. That matters because a process pool
    reuses its workers. If applying a payload merged, a worker that handled
    job ``EURUSD`` and then a run with no job identifier would keep logging
    ``EURUSD`` -- misattributing work to a job that had already finished,
    which is precisely the failure this mechanism exists to prevent.

    Unknown keys are ignored rather than raising: a newer parent sending an
    identifier an older worker does not recognise should degrade to slightly
    less informative logs, not to a crash.

    Parameters
    ----------
    payload
        A mapping produced by :func:`context_payload`. Pass an empty mapping
        to clear the context entirely.
    """
    for name, variable in _CONTEXT_VARIABLES.items():
        variable.set(payload.get(name))
```

---

## 4. `tranql/models/rade/rade_qnet/rade_qnet/core/provenance/seeding.py`

7831 bytes · SHA-256 `030643171325087b`

```python
"""
Deterministic seeding, and derived seeds for job sets.

Two problems are solved here, and the second is easy to overlook until a
parallel run produces different numbers from a sequential one.

**Seeding what is installed, from a package that imports nothing.**
``core`` may not import a training library, yet seeding PyTorch is exactly the
kind of thing this module should do. The resolution is a registry: an engine
registers a seeding callback when it is imported, and :func:`seed_everything`
invokes whatever has registered. ``core`` therefore seeds the libraries that
happen to be present without ever naming them.

**Per-job seeds that do not depend on execution order.**
Forty jobs cannot share one seed, and they must not draw seeds from a counter
or from the process identifier -- either makes the result depend on scheduling.
:func:`derive_seed` hashes the run seed together with a stable label, so job
``EURUSD`` gets the same seed whether it runs first, last, or in a worker on
another machine.

Determinism
-----------
``determinism`` is an explicit three-way choice rather than a boolean, because
full determinism costs real performance and some operations have no
deterministic implementation at all:

``off``
    Fastest. Seeds are set, but non-deterministic kernels are permitted.
``warn``
    Request deterministic kernels; where none exists, warn and continue.
``strict``
    Require deterministic kernels; a missing one is an error.

This replaces a blanket ``use_deterministic_algorithms(True)`` buried in a
swallowed ``try``/``except``, where neither the cost nor the failure was
visible to anyone.
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal, Protocol

import numpy as np

from ..lifecycle.errors import SpecError
from .logging import get_logger

__all__ = [
    "Determinism",
    "Seeder",
    "derive_seed",
    "register_seeder",
    "registered_seeders",
    "seed_everything",
    "unregister_seeder",
]

#: How hard to work for bit-for-bit reproducibility.
Determinism = Literal["off", "warn", "strict"]

#: Upper bound for a derived seed. Chosen because the numerical libraries this
#: framework wraps accept a 32-bit unsigned seed.
_SEED_MODULUS = 2**32

_LOGGER = get_logger(__name__)


class Seeder(Protocol):
    """
    A callback that seeds one library.

    Registered by whichever package owns the library, so ``core`` never
    imports it. An engine registers at module import; a user could register
    one for a library the framework has never heard of.
    """

    def __call__(self, seed: int, determinism: Determinism) -> None:
        """
        Seed the library and apply the requested determinism level.

        Parameters
        ----------
        seed
            A non-negative integer below ``2 ** 32``.
        determinism
            The requested level. An implementation that cannot honour
            ``strict`` must raise rather than degrade silently.
        """


# Process-global and deliberately so: there is one set of installed libraries
# per process. Registration is keyed by name and replacing an entry is
# explicit, so this cannot be mutated into meaning something different.
_SEEDERS: dict[str, Seeder] = {}


def register_seeder(name: str, seeder: Seeder, *, replace: bool = False) -> None:
    """
    Register a callback to be invoked by :func:`seed_everything`.

    Parameters
    ----------
    name
        Library name, used for logging and for replacement.
    seeder
        The callback.
    replace
        Permit overwriting an existing registration. Required explicitly so
        that two packages accidentally claiming one name is an error rather
        than a silent last-one-wins.

    Raises
    ------
    SpecError
        If ``name`` is already registered and ``replace`` is false.
    """
    if name in _SEEDERS and not replace:
        raise SpecError(
            f"a seeder named {name!r} is already registered; "
            f"pass replace=True to override it deliberately"
        )
    _SEEDERS[name] = seeder


def unregister_seeder(name: str) -> None:
    """
    Remove a registered seeder, if present.

    Parameters
    ----------
    name
        Library name. Absent names are ignored, so teardown is idempotent.
    """
    _SEEDERS.pop(name, None)


def registered_seeders() -> tuple[str, ...]:
    """
    Return the names of registered seeders, sorted.

    Returns
    -------
    tuple of str
        Registered library names.
    """
    return tuple(sorted(_SEEDERS))


def seed_everything(seed: int, *, determinism: Determinism = "off") -> int:
    """
    Seed the standard library, NumPy and every registered library.

    Parameters
    ----------
    seed
        Base seed. Must be a non-negative integer below ``2 ** 32``.
    determinism
        Passed through to each registered seeder.

    Returns
    -------
    int
        The seed that was applied, so a caller can record it in a run's
        lineage without recomputing it.

    Raises
    ------
    SpecError
        If the seed is out of range, or if a seeder fails under ``strict``.
    """
    if not 0 <= seed < _SEED_MODULUS:
        raise SpecError(f"seed must satisfy 0 <= seed < {_SEED_MODULUS}, received {seed}")

    random.seed(seed)
    # The legacy global NumPy generator is seeded in addition to any explicit
    # Generator a caller holds, because third-party code frequently uses
    # `np.random.*` directly and would otherwise be left unseeded.
    np.random.seed(seed)  # noqa: NPY002

    for name, seeder in sorted(_SEEDERS.items()):
        try:
            seeder(seed, determinism)
        except Exception as exc:
            if determinism == "strict":
                # Under `strict` the user asked for reproducibility and must
                # be told it was not achieved, rather than receiving results
                # that quietly are not reproducible.
                raise SpecError(
                    f"seeder {name!r} failed under determinism='strict': {exc}"
                ) from exc
            _LOGGER.warning(
                "seeder %r failed; results from that library will not be reproducible",
                name,
                exc_info=True,
            )
    return seed


def derive_seed(base_seed: int, *labels: str) -> int:
    """
    Derive a stable child seed from a base seed and one or more labels.

    The derivation is a hash, not a counter, which gives three properties a
    counter cannot. It is independent of execution order, so a job set is
    reproducible regardless of scheduling. It is identical across processes and
    machines, because it never touches ``hash()``. And it is stable under
    insertion, so adding a forty-first job leaves the other forty unchanged --
    without which every job would be retrained on a different seed whenever the
    portfolio grew.

    Parameters
    ----------
    base_seed
        The run's seed.
    *labels
        Stable identifiers, such as a job id or a trial number. Order
        matters: ``derive_seed(0, "a", "b")`` differs from
        ``derive_seed(0, "b", "a")``.

    Returns
    -------
    int
        A seed in ``[0, 2 ** 32)``.

    Raises
    ------
    SpecError
        If no label is given. A derived seed with no label would just be a
        reformatting of the base seed, which is never what the caller wants.
    """
    if not labels:
        raise SpecError("derive_seed requires at least one label")

    # NUL is used as the separator because it cannot occur in the identifiers
    # this is called with, so ("a", "bc") and ("ab", "c") cannot collide.
    material = "\0".join((str(base_seed), *labels)).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big") % _SEED_MODULUS
```

