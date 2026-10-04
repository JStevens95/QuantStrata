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
