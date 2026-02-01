from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FxSwap:
    """
    FX forward instrument.

    Parameters
    ----------
    NOT IMPLEMENTED!

    Notes
    -----
    This instrument definition stays generic and stable even as pricing models evolve.
    """