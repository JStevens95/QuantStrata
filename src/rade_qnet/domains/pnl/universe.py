"""
Which instruments a column refers to.

A replication problem has two sides. The **elementary** instruments are the
liquid things a desk can actually trade -- the hedging basis. The **target**
instruments are the illiquid or structured positions whose P&L is being
replicated. A model predicts the second from the first, and a universe is the
record of which is which, in which order.

Why this is not a model's concern
---------------------------------
The ordering is the contract between a matrix column and a real instrument. A
prediction is a vector of numbers; without a universe it is a vector of
numbers about nothing. Getting the order wrong does not raise -- it produces
plausible predictions attributed to the wrong instruments, which is the
single most expensive mistake available in this problem and the hardest to
see.

That contract is a property of the *problem*, not of whatever model happens
to be solving it. A ridge regression over the same portfolio has exactly the
same universe. Keeping it in the flagship model's state file would mean any
second model either imported from the flagship -- making a baseline depend on
the thing it is a baseline for -- or defined its own, at which point two
definitions of "column three" exist and nothing checks they agree.

``models`` may import ``domains``; ``domains`` may not import ``models``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Universe"]


@dataclass(frozen=True, slots=True)
class Universe:
    """
    Which instruments the feature and target columns refer to.

    Frozen because a universe is the thing every downstream index is
    interpreted against. A fitted state, a prediction and a report all read
    positions out of it, and a universe that could be reordered after a fit
    would silently reattribute every one of them.

    Parameters
    ----------
    elementary_ids
        The elementary instruments, **in column order** and already reduced
        to the selected basis. Already reduced, because the alternative --
        carrying the full set and a separate index of survivors -- means
        every consumer has to apply the reduction itself, and one that
        forgets produces results that are wrong rather than absent.
    target_ids
        The target instruments, in column order.
    """

    elementary_ids: tuple[str, ...]
    target_ids: tuple[str, ...]

    @property
    def n_elementary(self) -> int:
        """
        How many elementary instruments survived basis selection.

        Returns
        -------
        int
            The count.
        """
        return len(self.elementary_ids)

    @property
    def n_targets(self) -> int:
        """
        How many target instruments are predicted.

        Returns
        -------
        int
            The count.
        """
        return len(self.target_ids)

    @property
    def instrument_ids(self) -> tuple[str, ...]:
        """
        Every instrument, elementary block first.

        The row order of the combined attribute matrix, which is why the
        concatenation happens here rather than at each call site: two places
        choosing the same order by convention is a convention that will
        eventually be broken by someone who did not know it existed.

        Returns
        -------
        tuple of str
            Elementary identifiers followed by target identifiers.
        """
        return self.elementary_ids + self.target_ids

    def position_of(self, instrument_id: str) -> int:
        """
        Return an instrument's row in the combined attribute matrix.

        Parameters
        ----------
        instrument_id
            The identifier.

        Returns
        -------
        int
            Its position.

        Raises
        ------
        KeyError
            If the universe does not contain it. Raised rather than
            returning ``-1``, which indexes the last row perfectly happily
            and would attribute a prediction to whichever instrument
            happened to be there.
        """
        try:
            return self.instrument_ids.index(instrument_id)
        except ValueError:
            raise KeyError(
                f"{instrument_id!r} is not in this universe of "
                f"{self.n_elementary} elementary and {self.n_targets} target instrument(s)"
            ) from None
