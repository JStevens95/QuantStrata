"""
The hybrid network's tuning pipeline.

It discards proposals the architecture cannot build, and changes nothing
else.

Why that is worth a file
------------------------
Two of this model's layers reshape a width into heads::

    fusion:    width -> (fusion_heads,    width // fusion_heads)
    attention: width -> (attention_heads, width // attention_heads)

Neither reshape is meaningful unless the width divides exactly, so both
blocks raise :class:`~rade_qnet.core.runtime.errors.ContractError` at
construction when it does not. That is the right behaviour -- the
alternative is dropping ``width % heads`` features per head silently -- but
it makes a joint search over ``units`` and ``*_heads`` wasteful in a way the
framework cannot see.

Consider ``units in {16, 24, 32}`` crossed with ``fusion_heads in {1, 3, 4}``.
Four of the nine cells cannot build. A grid search spends 44% of its budget
discovering that, and each discovery costs a resolve, a model construction
and a traceback rather than a cheap arithmetic check. A random search over a
wider range does proportionally worse and, unlike the grid, does not even
fail the same way twice.

The framework cannot prevent this on its own. :class:`TuneSpec` validates
that a proposal *parses* -- that every path exists on the spec and every
value has the right type and range, which is Phase 5's cure for defect 7 --
and that is as far as a general validator can go. Whether two
independently-valid numbers are compatible *with each other* is a statement
about this architecture, and it lives with this architecture.

Why these are dropped rather than rejected
-------------------------------------------
An infeasible proposal is not a user error. A user who writes
``units: [16, 24, 32]`` and ``fusion_heads: [1, 3, 4]`` has asked a perfectly
reasonable question; it simply has fewer than nine answers. Failing the
search would force them to enumerate the feasible cells by hand, which is
both tedious and exactly the arithmetic this file already does.

So the proposals are filtered and the count is logged. What is *not* allowed
is a search that silently becomes empty: if the filter removes everything,
the space and the architecture genuinely contradict each other, and that is
a user error worth stopping for.

Why a pipeline override rather than a spec validator
------------------------------------------------------
``HybridModelSpec`` could reject the combination in a ``model_validator``,
and then every trial would fail fast instead of failing late. That is a
worse outcome, not a better one: the trial would still be *spent*. The
budget is the scarce thing in a search, not the seconds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ....core.runtime.errors import ContractError, SpecError
from ....orchestration.pipelines.search import expand
from ....orchestration.pipelines.tune import TunePipeline
from ..spec import HybridModelSpec

if TYPE_CHECKING:
    from typing import Any

    from ....core.spec.run import RunSpec

__all__ = ["MODEL_NAME", "HybridTunePipeline", "is_buildable"]

_LOGGER = logging.getLogger(__name__)

#: The blocks whose width is reshaped into heads, as ``(block, head field)``.
#:
#: Written as data rather than as two ``if`` statements so that a third
#: multi-head block added to the network is a one-line change here, and so
#: that forgetting to make it is visible as an omission from a list rather
#: than as an absence of code.
_HEADED_BLOCKS: tuple[tuple[str, str], ...] = (
    ("fusion", "fusion_heads"),
    ("attention", "attention_heads"),
)

#: The registered name this pipeline belongs to, used to recognise a run
#: spec it is entitled to have an opinion about.
MODEL_NAME = "hybrid_gnn_rnn"


def is_buildable(spec: RunSpec) -> bool:
    """
    Report whether this architecture can be constructed as specified.

    Mirrors the checks the layers themselves perform, deliberately: the
    layer is the authority and this is a cheap preview of its answer. The
    duplication is bounded -- two divisibility tests -- and the cost of
    them disagreeing is a trial that this function admitted and the model
    then refused, which surfaces as an ordinary failed trial rather than
    as a wrong number. The reverse duplication, where this function is the
    only check, is the one that would be dangerous.

    The model's parameters are parsed here rather than read off the spec,
    because a run spec carries a :class:`ComponentRef` -- a name and a bag
    of raw values -- and the typed spec that knows how a block's width is
    resolved does not exist until the pipeline's ``resolve`` stage. Reading
    ``params`` directly would mean reimplementing the defaulting rules that
    turn ``units`` into a per-block width, which is precisely the kind of
    second opinion this function is trying not to be.

    Parameters
    ----------
    spec
        A candidate run specification, already parsed and validated.

    Returns
    -------
    bool
        True if every multi-head block's width divides by its head count.

    Raises
    ------
    SpecError
        If the model parameters do not parse at all, which is a different
        question with a different answer: the trial is not infeasible, the
        request is malformed, and silently dropping it would turn a typo
        into a smaller search rather than an error.
    """
    reference = spec.model
    # Guards against being handed another model's run spec, which happens
    # if this pipeline is attached to the wrong definition. Answering True
    # is correct: this function knows of no reason that spec cannot build.
    if getattr(reference, "name", None) != MODEL_NAME:
        return True

    try:
        model = HybridModelSpec.model_validate(dict(reference.params))
    except ValidationError as error:
        raise SpecError(
            f"the proposed model parameters do not parse: {error}"
        ) from error

    return all(
        model.width(block) % getattr(model, field) == 0
        for block, field in _HEADED_BLOCKS
    )


class HybridTunePipeline(TunePipeline):
    """
    The framework's tuning pipeline, minus the trials that cannot build.

    ``propose`` is a stage rather than a hook, so this is a tier-3
    override -- but an additive one: it calls the framework's stage and
    filters the result, never replacing it. The stage sequence itself is
    untouched, and the framework's own validation still runs first, which
    is what keeps a mistyped search path a user error rather than a
    silently smaller search.
    """

    def propose(self) -> list[dict[str, Any]]:
        """
        Return the framework's proposals with the infeasible ones removed.

        Called after ``super()``, never instead of it, so the framework's
        own validation still runs first: a proposal that does not parse is
        rejected as a user error before this file sees it, and only
        proposals that are individually valid reach the compatibility test.

        Each surviving proposal is resolved to a run spec a second time --
        the base resolved it to validate it and kept only the flat form.
        Re-resolving costs a spec construction per trial and saves passing
        a parallel list of specs through a hook signature that every other
        model would then have to understand. The budget this file protects
        is measured in trials, not in parses.

        Returns
        -------
        list of dict
            One flat proposal per surviving trial.

        Raises
        ------
        SpecError
            If a proposal's model parameters do not parse. The framework's
            own check cannot see this: a run spec carries the model as a
            name and an untyped bag of values, and the bag is not validated
            against this model's schema until the ``resolve`` stage -- by
            which point the trial has been spent. Extending defect 7's cure
            to reach model parameters is a side-benefit of having to parse
            them here anyway.
        ContractError
            If no proposal in the whole space can be built, which means the
            search space and the architecture contradict each other.
        """
        proposals = super().propose()
        feasible = [
            proposal
            for index, proposal in enumerate(proposals)
            if is_buildable(self.spec.run_spec_for(expand(proposal), trial=index))
        ]

        discarded = len(proposals) - len(feasible)
        if not feasible:
            raise ContractError(
                f"none of the {len(proposals)} proposed trial(s) can be built: in "
                f"every one, a block's width is not divisible by its head count. "
                f"The search space and the architecture contradict each other, so "
                f"running it would produce a search with no trials rather than a "
                f"search with no winner. Check 'units' against "
                f"{[field for _, field in _HEADED_BLOCKS]}"
            )
        if discarded:
            _LOGGER.info(
                "discarded %d of %d proposed trial(s) whose width does not divide "
                "by its head count; %d remain",
                discarded,
                len(proposals),
                len(feasible),
            )
        return feasible
