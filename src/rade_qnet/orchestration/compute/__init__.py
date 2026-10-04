"""
Where a unit of work executes.

An executor takes a list of work items and runs them.  That is the whole
interface, and its narrowness is what keeps parallelism from leaking into
pipeline logic.  A pipeline cannot tell which executor is running it, so
switching from sequential to eight processes is a configuration change that
cannot alter results -- which parity level 5 verifies by running both and
comparing what they produced.

Modules
-------
``base.py``
    ``WorkItem``, ``WorkResult``, ``WorkFailure`` and the ``Executor``
    protocol.  Failures are returned, not raised, so one bad job cannot abort
    a set; results come back in input order, so a manifest never depends on
    which job happened to finish first.
``local.py``
    In-process sequential execution.  The reference implementation that all
    others must reproduce exactly, and the only sane way to debug.
``processes.py``
    A process pool using the spawn start method, with per-worker thread
    budgets set in the pool initialiser so N workers do not each claim every
    core and collapse throughput through oversubscription.
``gpus.py``
    One worker per visible device, each pinned by device-visibility
    environment variable before the training library is imported -- the only
    point at which such pinning reliably takes effect.
``policy.py``
    Chooses an executor and worker count from the hardware actually present
    and the size of the job set, so placement need not be hand-tuned, and
    records why it chose what it did.

Registration
------------
Nothing here is registered by name.  An executor is constructed by the
placement policy from a specification, not resolved through the component
registry, because there are four of them and they are framework-owned.  A
user supplying their own satisfies the ``Executor`` protocol and passes it to
the job-set runner directly.
"""

from __future__ import annotations

from .base import Executor, ResultSummary, WorkFailure, WorkItem, WorkResult, execute_item
from .gpus import GpuExecutor, visible_device_ids
from .local import LocalExecutor
from .policy import Placement, choose_placement, describe_machine
from .processes import THREAD_VARIABLES, ProcessExecutor, configure_worker

__all__ = [
    "THREAD_VARIABLES",
    "Executor",
    "GpuExecutor",
    "LocalExecutor",
    "Placement",
    "ProcessExecutor",
    "ResultSummary",
    "WorkFailure",
    "WorkItem",
    "WorkResult",
    "choose_placement",
    "configure_worker",
    "describe_machine",
    "execute_item",
    "visible_device_ids",
]
