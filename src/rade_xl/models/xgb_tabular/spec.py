"""
What the boosted-tree model needs to know about itself: nothing.

An empty spec is a real and correct answer, not a placeholder
---------------------------------------------------------------
Two schemas describe a run: ``model.params``, validated here, and
``training``, validated by the engine's own
:class:`~rade_xl.core.spec.training.XGBoostTrainingSpec`. The question
every model must answer is which settings belong on which side, and the
test is simple:

    Would changing it make this a *different model*, or the *same model
    trained differently*?

For this model the answer is always the second. It has no architecture of
its own -- it is the engine's booster over a flat table -- and
``XGBoostTrainingSpec`` already describes that booster completely: depth,
learning rate, subsample, regularisation, objective, round budget, early
stopping. There is nothing left for the model to declare.

So the class is empty, and the file still exists, because the layout is the
same at every tier and "this model has no settings" is worth saying once
and explicitly rather than leaving a reader to infer it from an absence.

Why this is not merely tidy
----------------------------
The earlier version of this file re-declared ``max_depth``,
``min_child_weight`` and ``reg_lambda`` with the same defaults as the
training spec. The engine merges the two at ``engine.py`` with
``{**booster_params(spec), **model.params}`` -- model last, model wins --
so a user who set ``training.max_depth: 12`` trained at depth 6 and was
told nothing. Both numbers were plausible, the run succeeded, and the only
symptom was a model slightly worse than it should have been.

That is the failure mode duplicated settings always have, and it is why
the ownership question above is the first one to answer when writing a
spec rather than a detail to settle later.
"""

from __future__ import annotations

from ...core.spec.base import Spec

__all__ = ["XgbTabularSpec"]


class XgbTabularSpec(Spec):
    """
    No model-level settings; the engine's training spec owns them all.

    Kept as a class rather than omitted so that the model's declaration in
    ``register.py`` reads the same as every other model's, and so that a
    specification carrying an unexpected ``model.params`` key is rejected
    rather than silently ignored.
    """
