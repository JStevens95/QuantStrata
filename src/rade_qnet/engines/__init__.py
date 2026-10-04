"""
How optimisation actually happens -- one adapter per library.

An engine is the only place in the framework that is allowed to know about a
specific training library.  It answers a fixed set of questions for its
library: how is a model instantiated, how is data materialised into the
library's native form, how is a fit run, how is a checkpoint written, how is a
prediction produced.

Because the pipelines above talk only to this interface, adding a backend
changes no pipeline.

Engines are framework-owned, models are not
-------------------------------------------
That is the one asymmetry worth knowing before planning work against this
package.  A *model* plugs in entirely from outside: import it and it is
registered, with no edit anywhere in ``rade_qnet`` (``test_extensibility.py``
proves a third-party model gets the whole lifecycle).  An *engine* does not.
Besides the sub-package and the registration line, a new engine needs its own
training-spec type added to the ``TrainingSpec`` union in
``core.spec.training`` -- the union is discriminated on a ``Literal`` engine
name, so a fourth name does not validate until it is declared there.

That is deliberate.  Each engine's settings get their own validated type,
which is what stops a one-shot fit being configured with gradient-descent
options, and it is a stronger guarantee than a free-form settings mapping
could give.  The cost is that a backend cannot arrive from outside the
distribution.  Stated here because the test suite makes it look otherwise:
its synthetic engines register under the name ``sklearn``, reusing a
discriminator that already exists, so none of them exercises a genuinely new
engine name.  ``test_extensibility.py`` covers the refusal explicitly.

One vocabulary, however large the engine
-----------------------------------------
Every engine package draws its filenames from one closed set, enforced by
``tests/rade_qnet/engines/test_engine_layout.py`` exactly as the model layout
test governs ``models``.  ``engine.py`` is required; ``materialise.py``,
``loaders.py`` and ``predictor.py`` are the other three verbs; ``training/``,
``learners/`` and ``hardware/`` are the parts of *fit* large enough to need
their own package.  Nothing else is permitted a name.

The set is optional down to a single file on purpose, so that the shape of a
package reports what its library owns.  A reader can tell from ``ls`` that a
boosted fit has no loop to configure and no device to choose.

Sub-packages
------------
``torch``
    PyTorch.  The richest engine, and the only one that needs all three
    sub-packages: gradient-based drivers, learners for both supervised and
    reinforcement learning, mixed precision, distributed training and
    lazy-parameter materialisation.
``xgboost``
    Gradient-boosted trees.  One module, because the fit is one call.
``sklearn``
    scikit-learn estimators, for baselines and sanity checks.  One module,
    for the same reason.

Modules
-------
``loaders.py``
    Draining a ``BatchSource`` into the single matrix a one-shot fit needs.
    Pure NumPy, and shared.  It lived in ``sklearn`` until the xgboost engine
    imported it from there, which made installing one backend depend on the
    presence of another; the layout test now refuses a cross-engine import
    outright.

Modules
-------
``base.py``
    The ``Engine`` protocol every adapter satisfies, ``EngineCapabilities``
    (what a pipeline may ask of an engine) and ``ModelHandle`` (a prepared
    model and its apparatus).  Scheduled for Phase 1 and delivered in Phase 2
    instead: Phase 1 built the pipeline skeleton against the *model
    definition*, so nothing consumed an engine interface and writing one would
    have meant designing a contract with no consumer -- the risk that phase's
    own risk table names as its largest.  [Phase 2]

Dependency rule
---------------
May import: ``core``, ``sources``.
May not import: ``orchestration``, ``models``, ``analysis``.
An engine never decides *where* it runs -- that is ``orchestration.compute`` --
and never writes a report.
"""

__all__: tuple[str, ...] = ()
