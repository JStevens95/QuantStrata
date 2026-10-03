"""
Where the recurrent baseline's data comes from, and what it requires of it.

This package is the reason the input contract exists, so the declaration
below is worth reading alongside :mod:`rade_xl.core.contract.requirement`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.contract.requirement import InputRequirement, RequiredInput
from ...sources.dataset.module import TabularDataModule

if TYPE_CHECKING:
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["REQUIRES", "data_module"]

#: Exactly one dynamic input, of rank 3 or rank 2, under any name.
#:
#: Each of those three clauses is load-bearing.
#:
#: *Exactly one*, because a recurrence reads one sequence. Before this was
#: declared, the forward pass scanned the batch and took the first tensor
#: it found, so a build emitting two feature blocks trained on whichever
#: one the dictionary happened to yield first -- reordering the build
#: changed the model and nothing reported it.
#:
#: *Rank 3 or 2*, because a window is ``(samples, timesteps, features)``
#: and a source with no sequence transform configured produces
#: ``(samples, features)``. The second is accepted and read as a window of
#: length one: that degrades this to a one-step recurrence, which is simply
#: a worse model and a better outcome than a run that cannot start on data
#: a user already has. Rank 4 is refused rather than quietly reshaped.
#:
#: *Any name*, because the input's name is the source's choice. A model
#: that hard-coded ``features`` would work against every fixture here and
#: fail against the first user whose block is called something else.
REQUIRES = InputRequirement(
    dynamic=(
        RequiredInput(
            rank=(3, 2),
            description="the feature window the recurrence reads",
        ),
    ),
)


def data_module(spec: SupervisedRunSpec) -> TabularDataModule:
    """
    Return the data module that builds this model's dataset.

    The framework's own. The sequence window comes from the source spec's
    transforms, so this model needs no data code of its own -- which is
    what keeps it tier 1 despite writing its own network.

    Parameters
    ----------
    spec
        The validated run specification. Unused.

    Returns
    -------
    TabularDataModule
        The framework's own, with nothing added.
    """
    del spec
    return TabularDataModule()
