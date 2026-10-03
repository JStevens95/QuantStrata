"""
What the boosted-tree model computes.

Why a tree baseline earns its place
------------------------------------
It is the model most likely to win. On tabular problems of the size a desk
actually has -- hundreds to tens of thousands of rows, dozens of features,
no spatial or sequential structure an architecture exploits -- boosted trees
beat neural networks routinely, and consistently enough that the burden of
proof sits with the network.

So this is not a straw man included for completeness. If the flagship cannot
beat it, that is the single most useful thing a comparison can report.

What it is handed, and why that matters when reading the result
-----------------------------------------------------------------
A sequence window arrives here flattened: one column per timestep-feature
pair. The tree therefore cannot know that column 7 and column 19 are the
same quantity at different lags, and has to rediscover any such relationship
from the data. That is a real handicap on a genuinely sequential problem and
no handicap at all on one where the recent past is just a set of numbers.

Which of those is true for a given book is exactly what comparing this
against ``lstm_tabular`` answers, and it is why both exist rather than one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Imported from the engine package rather than the library, because
# importing the package is what registers the engine -- and on macOS it is
# also what forces Torch's OpenMP runtime to load first. See
# `PHASE_6_ADDITIONAL_ENGINES.md` §8.2: the reverse order deadlocks with no
# error at all.
from ...engines.xgboost import BoosterModel

if TYPE_CHECKING:
    from .spec import XgbTabularSpec

__all__ = ["build"]


def build(settings: XgbTabularSpec) -> BoosterModel:
    """
    Return a holder for the booster training will produce.

    A booster is *returned* by ``xgboost.train`` rather than constructed and
    then fitted, so what can be built before training is an empty slot. That
    is what :class:`~rade_xl.engines.xgboost.engine.BoosterModel` is: the
    framework needs an object to hand to ``prepare`` and to save at the end,
    and the library does not provide one until the fit is over.

    The holder is created empty because this model declares no settings of
    its own -- see :mod:`rade_xl.models.xgb_tabular.spec` for why, and for
    what went wrong when it did. Everything the booster is configured with
    comes from the training spec, through the engine.

    Parameters
    ----------
    settings
        The validated model settings. Empty, and accepted anyway so that
        the signature does not have to change the day this model acquires
        one.

    Returns
    -------
    BoosterModel
        Unfitted, carrying no overrides.
    """
    del settings
    return BoosterModel()
