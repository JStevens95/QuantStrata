"""
Sequential, in-process execution.

The reference implementation. Every other executor is held to reproducing it
exactly, which is what parity level 5 asserts, so this one is written for
obviousness rather than for speed: it is a loop.

That is also why it is the right thing to debug with. A failure under a
process pool arrives as text, in a traceback from another process, with no
way to attach a debugger. The same job under this executor fails where it
happened, in the caller's process, with the caller's breakpoints intact --
and because placement cannot change results, a failure here is the same
failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.runtime.logging import get_logger
from .base import WorkResult, execute_item

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import WorkItem

__all__ = ["LocalExecutor"]

_LOGGER = get_logger(__name__)


class LocalExecutor:
    """
    Runs every item in the calling process, one after another.

    Holds no state, so one instance may be reused across sets.
    """

    def map(self, items: Sequence[WorkItem[object, object]]) -> list[WorkResult[object]]:
        """
        Run every item in order.

        Parameters
        ----------
        items
            The work to do.

        Returns
        -------
        list of WorkResult
            One result per item, in input order -- which here is also
            completion order, since there is nothing to reorder.
        """
        _LOGGER.info("running %d item(s) sequentially in this process", len(items))
        return [execute_item(item) for item in items]

    @property
    def description(self) -> str:
        """
        Describe where work runs.

        Returns
        -------
        str
            ``local (sequential, in-process)``.
        """
        return "local (sequential, in-process)"
