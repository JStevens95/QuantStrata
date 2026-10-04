"""
Where the boosted-tree model's data comes from, and what it requires of it.

Every model package has a ``data.py`` and it always answers two questions:

``REQUIRES``
    What the model consumes -- the input contract, checked against whatever
    the build produced before the model is constructed.
``data_module``
    Where the data comes from and how it is prepared.

Both live here, including when the answer is "the framework's own", because
the alternative is that a reader has to know which of two shapes a model is
before they know where to look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.contract.requirement import InputRequirement
from ...sources.dataset.tabular import TabularDataModule

if TYPE_CHECKING:
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["REQUIRES", "data_module"]

#: The boosted-tree model accepts anything.
#:
#: Not an omission. Every input is flattened into one design matrix before
#: it reaches the booster, so the number of blocks, their names and their
#: ranks do not change what this model computes.
#:
#: Worth knowing when reading a comparison against the flagship: a tree
#: handed a flattened window cannot tell that column 7 and column 19 are
#: the same quantity at different lags, and has to rediscover any such
#: relationship from the data. That is a real handicap on a sequential
#: problem and none at all otherwise -- and it is a consequence of this
#: declaration, not a detail of the engine.
#:
#: Declared explicitly rather than left unset, because "this model has no
#: constraints" and "nobody wrote the constraints down" look identical from
#: the outside and mean very different things. A reader of this line knows
#: the question was asked.
REQUIRES = InputRequirement.unconstrained()


def data_module(spec: SupervisedRunSpec) -> TabularDataModule:
    """
    Return the data module that builds this model's dataset.

    The framework's own, unmodified: a boosted-tree model reads a table, and
    a table is what :class:`~rade_qnet.sources.dataset.tabular.TabularDataModule`
    already knows how to load, split, scale, window and batch.

    A real deployment usually replaces this. The moment the data lives in
    a store rather than a CSV, the loader is yours -- and the change is to
    subclass ``DataModule`` here and return that instead, with nothing else
    in the package moving.

    Parameters
    ----------
    spec
        The validated run specification. Unused: the module reads its
        settings from ``spec.source`` when the framework calls it, which
        keeps this a pure constructor.

    Returns
    -------
    TabularDataModule
        The framework's own, with nothing added.
    """
    del spec
    return TabularDataModule()
