"""
Seeds Torch's random number generators, and makes ``strict`` mean something.

``core.provenance.seeding`` seeds Python and NumPy and then calls out to whatever
seeders have registered. It cannot seed Torch itself: ``core`` has an empty
dependency set, so it cannot import a training library. The registration
therefore belongs here, in the package that owns the library -- which is the
same dependency inversion the engine registry uses, applied to a different
concern.

Without this module, ``seed_everything`` leaves Torch unseeded. Every weight
initialisation, every dropout mask and every shuffled batch order would differ
between two runs of the same configuration, and the framework would be
reporting a seed that it had not actually applied to the one library that
matters most. Nothing in the output would say so: both runs complete, both
report the seed they were given, and the scores differ.

**What ``strict`` does and does not buy.** Seeding makes the *sampling*
reproducible. It does not make the *arithmetic* reproducible: a GPU reduction
sums in an order that depends on how the work was scheduled, and cuDNN selects
among algorithms by benchmarking them. ``strict`` therefore also asks Torch to
use deterministic algorithm implementations and turns the benchmark selector
off. That costs throughput, which is exactly why it is not the default.

Some operations have no deterministic implementation at all. Under ``strict``
Torch raises when one is reached, and that error propagates rather than being
caught: a user who asked for reproducibility must be told it is unavailable
for their model rather than receiving results that quietly are not
reproducible. Under ``warn`` the same situation logs and continues, which is
the level to use when reproducibility is desirable but not worth refusing a
run over.
"""

from __future__ import annotations

import os

import torch

from ....core.provenance.logging import get_logger
from ....core.provenance.seeding import Determinism, register_seeder

__all__ = ["SEEDER_NAME", "seed_torch"]

#: Name this seeder registers under. Exported so a test can unregister it and
#: so a user replacing it does not have to guess the string.
SEEDER_NAME = "torch"

#: Environment variable cuBLAS reads to select a deterministic reduction
#: workspace. It is read by the library at its first allocation, so setting it
#: later in a process has no effect -- which is why it is set here, at seeding
#: time, rather than inside the fit.
_CUBLAS_WORKSPACE_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"

#: The configuration cuBLAS requires for deterministic reductions. ``:4096:8``
#: is the larger of the two documented settings; it costs a little more memory
#: than ``:16:8`` and does not restrict the stream count.
_CUBLAS_WORKSPACE_VALUE = ":4096:8"

_LOGGER = get_logger(__name__)


def seed_torch(seed: int, determinism: Determinism = "off") -> None:
    """
    Seed every Torch generator and apply the requested determinism level.

    Parameters
    ----------
    seed
        A non-negative integer below ``2 ** 32``.
    determinism
        ``off`` seeds and nothing more. ``warn`` additionally asks for
        deterministic algorithms but tolerates an operation that has none.
        ``strict`` refuses such an operation, so that a run claiming
        reproducibility has it.

    Raises
    ------
    RuntimeError
        Propagated from Torch under ``strict`` when an operation reached
        during the run has no deterministic implementation. Deliberately not
        caught: ``seed_everything`` turns it into a specification error naming
        this seeder, and the alternative is a run that reports a seed it could
        not honour.
    """
    # `manual_seed` covers the CPU generator and every visible CUDA device, so
    # a per-device loop is unnecessary and would miss a device that appears
    # later.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Seeds all devices, including ones not yet initialised.
        torch.cuda.manual_seed_all(seed)

    if determinism == "off":
        return

    # cuDNN benchmarks its convolution algorithms on the first call and caches
    # the winner. The benchmark is timing-dependent, so the chosen algorithm --
    # and therefore the arithmetic -- can differ between two runs on the same
    # machine. Turning it off is a throughput cost and a prerequisite for
    # reproducibility.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if determinism == "strict":
        # Must be set before cuBLAS allocates its workspace, which happens on
        # the first matrix multiply. Setting it here means seeding is the last
        # moment it can still take effect.
        os.environ.setdefault(_CUBLAS_WORKSPACE_VARIABLE, _CUBLAS_WORKSPACE_VALUE)

    # `warn_only=True` logs and proceeds where no deterministic implementation
    # exists; under `strict` the same situation raises. That is the whole
    # difference between the two levels, and it is a difference in what the
    # user is promised rather than in what the framework attempts.
    torch.use_deterministic_algorithms(True, warn_only=determinism == "warn")
    _LOGGER.debug("seeded torch with %d at determinism=%r", seed, determinism)


# Registered at import, so importing the engine is enough to make
# `seed_everything` cover Torch. The train pipeline imports the engine before
# it seeds, which is the ordering this relies on -- and the ordering is tested,
# because a seeder registered after seeding is a seeder that did nothing.
register_seeder(SEEDER_NAME, seed_torch, replace=True)
