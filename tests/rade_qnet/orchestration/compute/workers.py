"""
Work functions for the executor tests.

In their own module, and module-level within it, because that is the
constraint being tested. A function defined inside a test -- a closure, a
lambda, a local `def` -- cannot cross a process boundary under the spawn
start method, so a test that used one would pass sequentially and fail under
the pool for a reason that has nothing to do with the executor.

Keeping them here also means the test module itself needs nothing picklable
in it, so a future test can use a fixture or a closure freely without
accidentally breaking the pool tests next to it.

Every function here is trivial and fast. A pool test slow enough to be
annoying is a pool test somebody eventually skips.
"""

from __future__ import annotations

import os

from src.rade_qnet.orchestration.compute.processes import THREAD_VARIABLES


def double(value: int) -> int:
    """
    Return twice the value.

    Parameters
    ----------
    value
        Any integer.

    Returns
    -------
    int
        Twice it.
    """
    return value * 2


def explode(message: str) -> int:
    """
    Raise a ``ValueError`` carrying the message.

    Parameters
    ----------
    message
        What the error should say.

    Returns
    -------
    int
        Never; the annotation exists so the item type-checks alongside the
        functions it is mixed with.

    Raises
    ------
    ValueError
        Always.
    """
    raise ValueError(message)


def fail_unless_even(value: int) -> int:
    """
    Return the value, failing on odd inputs.

    Used for partial-failure tests, where some items must succeed and others
    must not within a single call.

    Parameters
    ----------
    value
        Any integer.

    Returns
    -------
    int
        The value, if it is even.

    Raises
    ------
    ValueError
        If the value is odd.
    """
    if value % 2:
        message = f"{value} is odd"
        raise ValueError(message)
    return value


def read_thread_budget(_: object) -> dict[str, str | None]:
    """
    Report the thread-budget variables as this worker sees them.

    The only way to check a budget that is applied in another process. An
    assertion made in the parent would be testing the parent's environment,
    which the executor never touches.

    Parameters
    ----------
    _
        Ignored; the item interface requires a payload.

    Returns
    -------
    dict
        Variable name to value, or ``None`` where unset.
    """
    return {variable: os.environ.get(variable) for variable in THREAD_VARIABLES}


def read_environment_variable(name: str) -> str | None:
    """
    Report one environment variable as this worker sees it.

    Parameters
    ----------
    name
        The variable to read.

    Returns
    -------
    str or None
        Its value, or ``None`` if unset.
    """
    return os.environ.get(name)


def read_process_id(_: object) -> int:
    """
    Report this worker's process identifier.

    Used to confirm that work actually left the calling process, which no
    other observation establishes -- a pool that silently ran everything
    in-process would pass every other test here.

    Parameters
    ----------
    _
        Ignored; the item interface requires a payload.

    Returns
    -------
    int
        The process identifier.
    """
    return os.getpid()
