"""
The work a pipeline stage does, when more than one pipeline does it.

Why this is not part of ``pipelines``
--------------------------------------
It was, and that was the problem.  ``pipelines`` held five ``Pipeline``
subclasses and four modules that were not pipelines at all, so a folder named
for a noun contained things that were not that noun and ``ls`` stopped being
an answer to "what can this framework do?".

The split is along a real line rather than a tidy one.  A *pipeline* is a
named lifecycle a user can invoke: train, evaluate, infer, tune, reinforce.
A *stage* is a step inside one, and the four modules here are the steps that
more than one lifecycle performs identically.  Evaluation and inference both
reload a bundle; training, evaluation and tuning all turn a prepared dataset
into something scoreable.

Shared as functions, not as a base class
-----------------------------------------
That choice is older than this package and worth restating, because the
alternative looks cheaper every time.  ``EvaluatePipeline`` and
``InferPipeline`` share their reload path, and a common base class would
express that in fewer lines.  It would also mean that changing how inference
handles an unseen entity could silently change what evaluation reports, with
nothing in either file to suggest it might.  A function has one caller
relationship and it is visible at the call site.

Modules
-------
``resolve.py``
    Which pipeline class runs a given model's lifecycle, honouring a model's
    own overrides.  Read by the public API and by a job unit.
``reload.py``
    A model, its fitted state and its signature, back from a bundle
    directory.  Read by evaluation and inference, and the one place that
    refuses an interactive bundle to a supervised reader by name.
``scoring.py``
    Turning a prepared dataset into a ``BatchSource`` for one split,
    collecting targets, and aligning predictions back to the rows they came
    from.  Read by training, evaluation and tuning.
``search.py``
    Proposing hyper-parameter trials from a tuning specification, by grid or
    at random, and expanding a flat trial into nested specification blocks.
    Read by tuning only -- it lives here because it is a *stage*, not because
    it is shared, and because leaving it in ``tune.py`` made that module the
    largest pipeline by a third.
"""

__all__: tuple[str, ...] = ()
